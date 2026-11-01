"""Cost accounting and model/simulation metrics."""

from __future__ import annotations

import numpy as np
import pandas as pd

from .config import CostConfig


APPROVE = 0
REVIEW = 1
BLOCK = 2
ACTION_NAMES = {APPROVE: "approve", REVIEW: "review", BLOCK: "block"}


def decision_cost(action: int, label: int, amount: float, costs: CostConfig) -> float:
    """Return the report's realized point cost for one decision."""
    if action == APPROVE:
        return float(amount) if int(label) == 1 else 0.0
    if action == REVIEW:
        return float(costs.review_cost)
    if action == BLOCK:
        return 0.0 if int(label) == 1 else float(costs.false_decline_rate * amount)
    raise ValueError(f"Unknown action id: {action}")


def decision_costs(actions, labels, amounts, costs: CostConfig) -> np.ndarray:
    """Vectorized report cost table for a batch of decisions."""
    actions = np.asarray(actions, dtype=np.int8)
    labels = np.asarray(labels, dtype=np.int8)
    amounts = np.asarray(amounts, dtype=np.float64)
    if not (len(actions) == len(labels) == len(amounts)):
        raise ValueError("Actions, labels, and amounts must have equal length")
    result = np.zeros(len(actions), dtype=np.float64)
    approve = actions == APPROVE
    block = actions == BLOCK
    review = actions == REVIEW
    result[approve & (labels == 1)] = amounts[approve & (labels == 1)]
    result[block & (labels == 0)] = costs.false_decline_rate * amounts[block & (labels == 0)]
    result[review] = costs.review_cost
    if np.any(~(approve | block | review)):
        raise ValueError("Action vector contains an unknown action id")
    return result


def expected_action_cost(probability: float, amount: float, costs: CostConfig) -> dict[int, float]:
    """Compute expected costs from the calibrated risk score and cost table."""
    p = float(np.clip(probability, 0.0, 1.0))
    amount = float(max(amount, 0.0))
    return {
        APPROVE: p * amount,
        REVIEW: float(costs.review_cost),
        BLOCK: (1.0 - p) * costs.false_decline_rate * amount,
    }


def cost_summary(actions, labels, amounts, costs: CostConfig, days=None) -> dict:
    """Compute total cost and operational review metrics from hidden outcomes."""
    actions = np.asarray(actions, dtype=np.int8)
    labels = np.asarray(labels, dtype=np.int8)
    amounts = np.asarray(amounts, dtype=np.float64)
    total = float(decision_costs(actions, labels, amounts, costs).sum())
    review_count = int(np.count_nonzero(actions == REVIEW))
    if days is None:
        cap = int(np.floor(costs.review_cap * len(actions)))
    else:
        day_values = np.asarray(days)
        if len(day_values) != len(actions):
            raise ValueError("Days must match the decision vector")
        cap = int(sum(np.floor(costs.review_cap * np.count_nonzero(day_values == day)) for day in np.unique(day_values)))
    return {
        "total_cost": float(total),
        "transactions": int(len(actions)),
        "review_count": review_count,
        "review_rate": float(review_count / len(actions)) if len(actions) else 0.0,
        "review_cap_count": cap,
        "review_cap_utilization": float(review_count / cap) if cap else 0.0,
        "approve_count": int(np.count_nonzero(actions == APPROVE)),
        "block_count": int(np.count_nonzero(actions == BLOCK)),
        "approve_genuine_correct": int(np.count_nonzero((actions == APPROVE) & (labels == 0))),
        "approve_fraud_missed": int(np.count_nonzero((actions == APPROVE) & (labels == 1))),
        "block_fraud_correct": int(np.count_nonzero((actions == BLOCK) & (labels == 1))),
        "block_genuine_false_decline": int(np.count_nonzero((actions == BLOCK) & (labels == 0))),
        "review_fraud": int(np.count_nonzero((actions == REVIEW) & (labels == 1))),
        "review_genuine": int(np.count_nonzero((actions == REVIEW) & (labels == 0))),
    }


def classification_metrics(labels, probabilities) -> dict:
    """Calculate ranking, threshold, and probability-calibration metrics."""
    y = np.asarray(labels, dtype=np.int8)
    p = np.clip(np.asarray(probabilities, dtype=np.float64), 0.0, 1.0)
    pred = p >= 0.5
    positives = int(y.sum())
    negatives = int(len(y) - positives)
    if positives == 0 or negatives == 0:
        pr_auc = roc_auc = float("nan")
    else:
        order = np.argsort(-p, kind="mergesort")
        sorted_y = y[order]
        sorted_desc = p[order]
        group_ends = np.flatnonzero(np.r_[sorted_desc[1:] != sorted_desc[:-1], True]) + 1
        tp_at_threshold = np.cumsum(sorted_y)[group_ends - 1]
        group_positives = np.diff(np.r_[0, tp_at_threshold])
        pr_auc = float(np.sum((tp_at_threshold / group_ends) * group_positives) / positives)
        # Average ranks give the standard tie-corrected Mann-Whitney ROC-AUC.
        rank_order = np.argsort(p, kind="mergesort")
        sorted_p = p[rank_order]
        group_starts = np.r_[0, np.flatnonzero(sorted_p[1:] != sorted_p[:-1]) + 1]
        group_ends_ascending = np.r_[group_starts[1:], len(y)]
        average_ranks = (group_starts + 1 + group_ends_ascending) / 2.0
        ranks = np.empty(len(y), dtype=np.float64)
        ranks[rank_order] = np.repeat(average_ranks, group_ends_ascending - group_starts)
        roc_auc = float((ranks[y == 1].sum() - positives * (positives + 1) / 2) / (positives * negatives))
    tp = int(np.count_nonzero(pred & (y == 1)))
    fp = int(np.count_nonzero(pred & (y == 0)))
    fn = int(np.count_nonzero(~pred & (y == 1)))
    precision = tp / (tp + fp) if tp + fp else 0.0
    recall = tp / (tp + fn) if tp + fn else 0.0
    f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
    bins = np.linspace(0.0, 1.0, 11)
    ece = 0.0
    for lo, hi in zip(bins[:-1], bins[1:]):
        mask = (p >= lo) & (p <= hi if hi == 1.0 else p < hi)
        if np.any(mask):
            ece += float(mask.mean() * abs(p[mask].mean() - y[mask].mean()))
    return {
        "pr_auc": pr_auc,
        "roc_auc": roc_auc,
        "precision_at_0_5": float(precision),
        "recall_at_0_5": float(recall),
        "f1_at_0_5": float(f1),
        "brier_score": float(np.mean((p - y) ** 2)),
        "ece_10_bins": float(ece),
    }

