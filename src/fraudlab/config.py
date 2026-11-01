"""Shared experiment settings and serializable configuration helpers."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from pathlib import Path
import json


@dataclass(frozen=True)
class SplitConfig:
    """Chronological fractions; validation is carved from the training period."""

    train_period_fraction: float = 0.60
    fit_fraction_of_train_period: float = 0.80

    @property
    def fit_fraction(self) -> float:
        return self.train_period_fraction * self.fit_fraction_of_train_period

    @property
    def validation_fraction(self) -> float:
        return self.train_period_fraction - self.fit_fraction


@dataclass(frozen=True)
class CostConfig:
    """Costs in points, matching the Phase 1 report."""

    false_decline_rate: float = 0.10
    review_cost: float = 5.0
    review_cap: float = 0.03


@dataclass(frozen=True)
class FeedbackConfig:
    """Explicit starting assumptions for feedback absent from IEEE-CIS."""

    review_delay_days: int = 1
    maturity_days: int = 30
    fraud_delay_min_days: int = 1
    fraud_delay_max_days: int = 30
    fraud_censor_probability: float = 0.20
    # None preserves the original assumption that blocked outcomes never arrive.
    blocked_label_delay_days: int | None = None
    seed: int = 42

    def __post_init__(self):
        if self.review_delay_days < 0 or self.maturity_days < 0:
            raise ValueError("Feedback delays must be non-negative")
        if self.blocked_label_delay_days is not None and self.blocked_label_delay_days < 0:
            raise ValueError("Blocked-label delay must be non-negative or None")
        if self.fraud_delay_min_days < 0 or self.fraud_delay_max_days < self.fraud_delay_min_days:
            raise ValueError("Fraud delay range must be non-negative and ordered")
        if not 0.0 <= self.fraud_censor_probability <= 1.0:
            raise ValueError("Fraud censor probability must be between 0 and 1")


@dataclass(frozen=True)
class BanditConfig:
    """Linear-softmax contextual policy training settings."""

    learning_rate: float = 0.02
    exploration_epsilon: float = 0.05
    advantage_scale: float = 100.0
    advantage_clip: float = 5.0
    gradient_norm_clip: float = 5.0
    importance_ratio_clip: float = 10.0
    baseline_rate: float = 0.05
    seed: int = 42


@dataclass(frozen=True)
class ExperimentConfig:
    """Top-level configuration saved with every experiment."""

    split: SplitConfig = SplitConfig()
    costs: CostConfig = CostConfig()
    feedback: FeedbackConfig = FeedbackConfig()
    bandit: BanditConfig = BanditConfig()
    historical_phase1_cost: float = 720345.0
    random_seed: int = 42

    def to_dict(self) -> dict:
        """Return a JSON-friendly nested dictionary."""
        return asdict(self)


def save_config(config: ExperimentConfig, path: str | Path) -> None:
    """Save the complete experiment configuration as readable JSON."""
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(json.dumps(config.to_dict(), indent=2), encoding="utf-8")

