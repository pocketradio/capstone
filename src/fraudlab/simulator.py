"""Closed-loop and immediate-feedback oracle replay with action-dependent labels."""

from __future__ import annotations

from collections import defaultdict
from dataclasses import asdict
from pathlib import Path
import json

import numpy as np
import pandas as pd

from .bandit import BanditObservation, LinearContextualBandit, make_context
from .config import BanditConfig, CostConfig, FeedbackConfig
from .metrics import ACTION_NAMES, APPROVE, BLOCK, REVIEW, classification_metrics, cost_summary, decision_cost
from .policy import no_review_actions, threshold_actions


def _schedule_release(action, label, day, feedback: FeedbackConfig, rng):
    """Return release day or None according to the configured feedback process."""
    if action == BLOCK:
        return (None if feedback.blocked_label_delay_days is None
                else int(day + feedback.blocked_label_delay_days))
    if action == REVIEW:
        return int(day + feedback.review_delay_days)
    if int(label) == 0:
        return int(day + feedback.maturity_days)
    if rng.random() < feedback.fraud_censor_probability:
        return None
    delay = int(rng.integers(feedback.fraud_delay_min_days, feedback.fraud_delay_max_days + 1))
    return int(day + min(feedback.maturity_days, delay))


def _release_feedback(policy, queue, current_day, immediate=False, released_by_action=None):
    """Deliver all due outcomes, updating the policy only after release."""
    if immediate:
        return
    due_days = [day for day in queue if day <= current_day]
    for day in sorted(due_days):
        for record in queue.pop(day):
            policy.update(record["observation"], record["reward"])
            if released_by_action is not None:
                released_by_action[record["action"]] += 1


def run_bandit_simulation(
    rows: pd.DataFrame,
    probabilities,
    mode: str,
    costs: CostConfig | None = None,
    feedback: FeedbackConfig | None = None,
    bandit_config: BanditConfig | None = None,
    seed: int | None = None,
    amount_scale: float | None = None,
) -> tuple[pd.DataFrame, dict]:
    """Replay the held-out stream without exposing unavailable labels to the learner."""
    costs = costs or CostConfig()
    feedback = feedback or FeedbackConfig()
    bandit_config = bandit_config or BanditConfig()
    if mode not in {"closed_loop", "oracle"}:
        raise ValueError("mode must be 'closed_loop' or 'oracle'")
    required = {"isFraud", "TransactionAmt", "transaction_day"}
    missing = required - set(rows.columns)
    if missing:
        raise ValueError(f"Simulation rows are missing columns: {sorted(missing)}")
    if len(rows) != len(probabilities):
        raise ValueError("Probability vector must match simulation rows")
    frame = rows.reset_index(drop=True)
    p = np.clip(np.asarray(probabilities, dtype=np.float64), 0.0, 1.0)
    labels = frame["isFraud"].to_numpy(dtype=np.int8)
    amounts = frame["TransactionAmt"].to_numpy(dtype=np.float64)
    days = frame["transaction_day"].to_numpy(dtype=np.int64)
    used_seed = bandit_config.seed if seed is None else int(seed)
    cfg = BanditConfig(**{**asdict(bandit_config), "seed": used_seed})
    policy = LinearContextualBandit(cfg)
    amount_scale = float(amount_scale or np.quantile(amounts, 0.95))
    delay_seed = feedback.seed + (used_seed - bandit_config.seed)
    rng = np.random.default_rng(delay_seed)
    feedback_queue = defaultdict(list)
    from collections import Counter
    released_by_action = Counter()
    records = []

    for day in np.unique(days):
        _release_feedback(policy, feedback_queue, int(day), immediate=(mode == "oracle"),
                          released_by_action=released_by_action)
        indices = np.flatnonzero(days == day)
        cap = int(np.floor(costs.review_cap * len(indices)))
        reviews_used = 0
        for ix in indices:
            remaining = max(cap - reviews_used, 0)
            remaining_fraction = remaining / max(cap, 1)
            context = make_context(p[ix], amounts[ix], remaining_fraction, amount_scale)
            observation = policy.act(context, review_available=(remaining > 0))
            action = observation.action
            if action == REVIEW:
                reviews_used += 1
            reward_cost = decision_cost(int(action), int(labels[ix]), float(amounts[ix]), costs)
            reward = -float(reward_cost)
            reveal_day = _schedule_release(int(action), int(labels[ix]), int(day), feedback, rng)
            if mode == "oracle":
                # The oracle sees the result after choosing, never before its action.
                policy.update(observation, reward)
                released_by_action[ACTION_NAMES[int(action)]] += 1
                visible_day = int(day)
            else:
                visible_day = reveal_day
                if reveal_day is not None:
                    feedback_queue[int(reveal_day)].append({
                        "observation": observation, "reward": reward,
                        "action": ACTION_NAMES[int(action)],
                    })
            records.append(
                {
                    "row_position": int(ix), "day": int(day), "action_id": int(action),
                    "action": ACTION_NAMES[int(action)], "action_probability": observation.behavior_probability,
                    "target_action_probability": observation.target_probability,
                    "label_release_day": visible_day,
                    "evaluator_feedback_scheduled_day": reveal_day,
                    "cost_points": reward_cost,
                    "policy_updates_so_far": policy.update_count,
                }
            )

    decisions = pd.DataFrame.from_records(records)
    stats = cost_summary(decisions["action_id"], labels, amounts, costs, days=days)
    stats.update(classification_metrics(labels, p))
    actions = decisions["action_id"].to_numpy(dtype=np.int8)
    scheduled = decisions["evaluator_feedback_scheduled_day"]
    blocked = actions == BLOCK
    released_by_end = scheduled.notna().to_numpy() & (scheduled.fillna(np.inf).to_numpy() <= int(days.max()))
    pending_at_end = scheduled.notna().to_numpy() & (scheduled.fillna(-np.inf).to_numpy() > int(days.max()))
    # In closed loop, releases are inferred from their scheduled day and horizon;
    # this counts delivered events even when multiple events mature on a date without a transaction.
    delivered_count = (int(policy.update_count) if mode == "oracle" else int(np.count_nonzero(released_by_end)))
    stats.update(
        {
            "mode": mode,
            "seed": used_seed,
            "feedback_updates": int(policy.update_count),
            "blocked_labels_permanently_censored": int(np.count_nonzero(blocked & scheduled.isna().to_numpy()))
            if mode == "closed_loop" else 0,
            "blocked_labels_scheduled": int(np.count_nonzero(blocked & scheduled.notna().to_numpy())),
            "blocked_labels_released_by_horizon": int(np.count_nonzero(blocked & released_by_end)),
            "blocked_labels_pending_at_end": int(np.count_nonzero(blocked & pending_at_end)),
            "approved_fraud_labels_permanently_censored": int(np.count_nonzero(
                (decisions["action_id"].to_numpy() == APPROVE) & (labels == 1)
                & decisions["evaluator_feedback_scheduled_day"].isna().to_numpy()
            )) if mode == "closed_loop" else 0,
            "feedback_events_pending_at_end": int(sum(len(v) for v in feedback_queue.values())),
            "feedback_events_scheduled": int(scheduled.notna().sum()),
            "feedback_events_released_by_horizon": delivered_count,
            "feedback_updates_by_action": dict(released_by_action),
            "feedback_config": asdict(feedback),
            "bandit_config": asdict(cfg),
        }
    )
    return decisions, stats


def run_threshold_baselines(rows, probabilities, costs: CostConfig, threshold_config: dict | None = None) -> dict[str, dict]:
    """Evaluate simple baselines on known labels without changing the learner stream."""
    frame = rows.reset_index(drop=True)
    p = np.asarray(probabilities, dtype=np.float64)
    y = frame["isFraud"].to_numpy(dtype=np.int8)
    a = frame["TransactionAmt"].to_numpy(dtype=np.float64)
    days = frame["transaction_day"].to_numpy(dtype=np.int64)
    result = {}
    always_approve = np.full(len(frame), APPROVE, dtype=np.int8)
    always_block = np.full(len(frame), BLOCK, dtype=np.int8)
    no_review = no_review_actions(p, a, costs)
    result["always_approve"] = cost_summary(always_approve, y, a, costs, days=days)
    result["always_block"] = cost_summary(always_block, y, a, costs, days=days)
    result["no_review"] = cost_summary(no_review, y, a, costs, days=days)
    if threshold_config:
        actions = np.empty(len(frame), dtype=np.int8)
        for day in np.unique(days):
            ix = np.flatnonzero(days == day)
            actions[ix] = threshold_actions(
                p[ix], a[ix], threshold_config["t_review"], threshold_config["t_block"], costs
            )
        result["validation_tuned_thresholds"] = cost_summary(actions, y, a, costs, days=days)
    return result


def save_simulation_outputs(decisions: pd.DataFrame, metrics: dict, output_dir: str | Path, stem: str):
    """Persist decisions and metrics for one model/mode run."""
    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)
    decisions.to_csv(out / f"{stem}_decisions.csv", index=False)
    (out / f"{stem}_metrics.json").write_text(json.dumps(metrics, indent=2), encoding="utf-8")

