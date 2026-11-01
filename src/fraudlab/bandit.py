"""Delayed-reward linear contextual policy-gradient bandit."""

from __future__ import annotations

from dataclasses import dataclass
import numpy as np

from .config import BanditConfig
from .metrics import APPROVE, BLOCK, REVIEW


N_ACTIONS = 3


def make_context(probability: float, amount: float, remaining_review_fraction: float, amount_scale: float) -> np.ndarray:
    """Encode the agreed context: calibrated risk, log amount, and remaining capacity."""
    scale = max(float(amount_scale), 1.0)
    return np.asarray(
        [1.0, float(np.clip(probability, 0.0, 1.0)), float(np.log1p(max(amount, 0.0)) / np.log1p(scale)),
         float(np.clip(remaining_review_fraction, 0.0, 1.0))],
        dtype=np.float64,
    )


@dataclass
class BanditObservation:
    """Decision record retained until its reward is released by the simulator."""

    context: np.ndarray
    action: int
    behavior_probability: float
    target_probability: float
    propensity: float
    valid_actions: np.ndarray


class LinearContextualBandit:
    """Softmax policy-gradient contextual bandit with epsilon exploration."""

    def __init__(self, config: BanditConfig | None = None):
        self.config = config or BanditConfig()
        self.weights = np.zeros((N_ACTIONS, 4), dtype=np.float64)
        self.rng = np.random.default_rng(self.config.seed)
        self.reward_baseline = 0.0
        self.baseline_seen = 0
        self.update_count = 0

    def _softmax(self, context: np.ndarray, valid_actions: np.ndarray) -> np.ndarray:
        logits = self.weights @ context
        logits = logits[valid_actions]
        logits = logits - np.max(logits)
        exps = np.exp(np.clip(logits, -40.0, 40.0))
        probs_valid = exps / exps.sum()
        probs = np.zeros(N_ACTIONS, dtype=np.float64)
        probs[valid_actions] = probs_valid
        return probs

    def act(self, context: np.ndarray, review_available: bool) -> BanditObservation:
        """Sample a legal action and log its behavior and target propensities."""
        valid = np.asarray([APPROVE, BLOCK] if not review_available else [APPROVE, REVIEW, BLOCK], dtype=np.int64)
        target = self._softmax(context, valid)
        uniform = np.zeros(N_ACTIONS, dtype=np.float64)
        uniform[valid] = 1.0 / len(valid)
        eps = float(np.clip(self.config.exploration_epsilon, 0.0, 1.0))
        behavior = (1.0 - eps) * target + eps * uniform
        action = int(self.rng.choice(np.arange(N_ACTIONS), p=behavior))
        return BanditObservation(
            context=context.copy(), action=action,
            behavior_probability=float(behavior[action]),
            target_probability=float(target[action]),
            propensity=float(behavior[action]),
            valid_actions=valid.copy(),
        )

    def update(self, observation: BanditObservation, reward: float) -> None:
        """Apply a clipped, importance-weighted REINFORCE update when reward arrives."""
        action = int(observation.action)
        self.baseline_seen += 1
        if self.baseline_seen == 1:
            self.reward_baseline = float(reward)
        else:
            rate = float(self.config.baseline_rate)
            self.reward_baseline = (1.0 - rate) * self.reward_baseline + rate * float(reward)
        advantage = (float(reward) - self.reward_baseline) / max(self.config.advantage_scale, 1e-9)
        advantage = float(np.clip(advantage, -self.config.advantage_clip, self.config.advantage_clip))
        logits = self.weights @ observation.context
        valid = observation.valid_actions
        target = np.zeros(N_ACTIONS, dtype=np.float64)
        valid_logits = logits[valid] - np.max(logits[valid])
        target[valid] = np.exp(np.clip(valid_logits, -40.0, 40.0))
        target /= target.sum()
        ratio = min(
            float(target[action] / max(observation.behavior_probability, 1e-12)),
            self.config.importance_ratio_clip,
        )
        grad = -target[:, None] * observation.context[None, :]
        grad[action] += observation.context
        grad *= advantage * ratio
        norm = float(np.linalg.norm(grad))
        if norm > self.config.gradient_norm_clip:
            grad *= self.config.gradient_norm_clip / norm
        self.weights += self.config.learning_rate * grad
        self.update_count += 1

