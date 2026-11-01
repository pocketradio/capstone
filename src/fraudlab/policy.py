"""Cost-sensitive threshold baselines with a strict daily review cap."""

from __future__ import annotations

import itertools

import numpy as np

from .config import CostConfig
from .metrics import APPROVE, BLOCK, REVIEW, decision_costs


def apply_review_cap(actions, probabilities, amounts, costs: CostConfig, cap_count: int | None = None):
    """Mask excess reviews; send them to the lower expected-cost non-review action."""
    actions = np.asarray(actions, dtype=np.int8).copy()
    p = np.asarray(probabilities, dtype=np.float64)
    amounts = np.asarray(amounts, dtype=np.float64)
    cap_count = int(np.floor(costs.review_cap * len(actions))) if cap_count is None else max(int(cap_count), 0)
    review_pos = np.flatnonzero(actions == REVIEW)
    if len(review_pos) > cap_count:
        # Keep the highest-risk reviews; then fall back to the cheapest expected non-review action.
        keep = review_pos[np.argsort(p[review_pos], kind="stable")[-cap_count:]] if cap_count else np.array([], dtype=int)
        rejected = np.setdiff1d(review_pos, keep, assume_unique=True)
        approve_cost = p[rejected] * amounts[rejected]
        block_cost = (1.0 - p[rejected]) * costs.false_decline_rate * amounts[rejected]
        actions[rejected] = np.where(block_cost < approve_cost, BLOCK, APPROVE).astype(np.int8)
    return actions


def threshold_actions(probabilities, amounts, t_review: float, t_block: float, costs: CostConfig):
    """Apply approve/review/block score zones and enforce a daily cap."""
    p = np.asarray(probabilities, dtype=np.float64)
    amounts = np.asarray(amounts, dtype=np.float64)
    if t_review > t_block:
        raise ValueError("t_review must be <= t_block")
    actions = np.full(len(p), APPROVE, dtype=np.int8)
    actions[p >= t_block] = BLOCK
    actions[(p >= t_review) & (p < t_block)] = REVIEW
    return apply_review_cap(actions, p, amounts, costs)


def tune_threshold_policy(probabilities, labels, amounts, days, costs: CostConfig, grid_size: int = 41) -> dict:
    """Grid-search two thresholds on validation data with a per-day review cap."""
    p = np.asarray(probabilities, dtype=np.float64)
    y = np.asarray(labels, dtype=np.int8)
    a = np.asarray(amounts, dtype=np.float64)
    days = np.asarray(days)
    if not (len(p) == len(y) == len(a) == len(days)):
        raise ValueError("Validation arrays must have equal length")
    grid = np.unique(np.quantile(p, np.linspace(0.0, 1.0, grid_size)))
    day_groups = [np.flatnonzero(days == day) for day in np.unique(days)]
    best = None
    for t_review, t_block in itertools.combinations_with_replacement(grid, 2):
        total = 0.0
        for ix in day_groups:
            actions = threshold_actions(p[ix], a[ix], float(t_review), float(t_block), costs)
            total += float(decision_costs(actions, y[ix], a[ix], costs).sum())
        candidate = {"t_review": float(t_review), "t_block": float(t_block), "validation_cost": float(total)}
        if best is None or candidate["validation_cost"] < best["validation_cost"]:
            best = candidate
    return best or {"t_review": 1.0, "t_block": 1.0, "validation_cost": float("nan")}


def no_review_actions(probabilities, amounts, costs: CostConfig):
    """Choose between approval and blocking using expected cost from calibrated risk."""
    p = np.asarray(probabilities, dtype=np.float64)
    amounts = np.asarray(amounts, dtype=np.float64)
    approve_cost = p * amounts
    block_cost = (1.0 - p) * costs.false_decline_rate * amounts
    return np.where(block_cost < approve_cost, BLOCK, APPROVE).astype(np.int8)

