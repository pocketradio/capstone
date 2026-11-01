"""Dataset loading, chronological split, and dataset profiling."""

from __future__ import annotations

from pathlib import Path
import json
import os

import numpy as np
import pandas as pd

from .config import SplitConfig


TARGET = "isFraud"
TIME_COLUMN = "TransactionDT"
AMOUNT_COLUMN = "TransactionAmt"
ID_COLUMN = "TransactionID"


def find_transaction_file(data_dir: str | Path | None = None) -> Path:
    """Resolve train_transaction.csv from an explicit path or common notebook mounts."""
    candidates = []
    if data_dir is None:
        env_dir = os.environ.get("IEEE_CIS_DATA_DIR")
        if env_dir:
            data_dir = env_dir
    if data_dir is not None:
        root = Path(data_dir)
        candidates.extend([root / "train_transaction.csv", root])
    candidates.extend(
        [
            Path("data/raw/train_transaction.csv"),
            Path("/kaggle/input/ieee-fraud-detection/train_transaction.csv"),
            Path("/kaggle/input/ieee-cis-fraud-detection/train_transaction.csv"),
            Path("/kaggle/input/competitions/ieee-fraud-detection/train_transaction.csv"),
        ]
    )
    for candidate in candidates:
        if candidate.is_file() and candidate.name.lower() == "train_transaction.csv":
            return candidate
    for root in (Path("/kaggle/input"),):
        if root.exists():
            # Competition sources are mounted one level deeper than datasets:
            # /kaggle/input/competitions/<competition-slug>/...
            found = list(root.rglob("train_transaction.csv"))
            if found:
                return found[0]
    searched = "\n".join(str(p) for p in candidates)
    raise FileNotFoundError(
        "Could not locate train_transaction.csv. Set IEEE_CIS_DATA_DIR to the "
        f"directory containing it. Checked:\n{searched}"
    )


def load_transactions(data_dir: str | Path | None = None) -> pd.DataFrame:
    """Load, validate, and stably sort the labeled IEEE-CIS training table."""
    path = find_transaction_file(data_dir)
    frame = pd.read_csv(path, low_memory=False)
    required = {TARGET, TIME_COLUMN, AMOUNT_COLUMN}
    missing = required - set(frame.columns)
    if missing:
        raise ValueError(f"Dataset is missing required columns: {sorted(missing)}")
    frame = frame.sort_values([TIME_COLUMN, ID_COLUMN], kind="mergesort").reset_index(drop=True)
    frame[TARGET] = frame[TARGET].astype("int8")
    return frame


def chronological_positions(frame: pd.DataFrame, config: SplitConfig | None = None) -> dict[str, np.ndarray]:
    """Return positional indices for fit, validation, and final simulation periods."""
    config = config or SplitConfig()
    if not 0 < config.fit_fraction < config.train_period_fraction < 1:
        raise ValueError("Split fractions must satisfy 0 < fit < training period < 1")
    n_rows = len(frame)
    fit_end = int(n_rows * config.fit_fraction)
    train_period_end = int(n_rows * config.train_period_fraction)
    if min(fit_end, train_period_end - fit_end, n_rows - train_period_end) <= 0:
        raise ValueError("Dataset is too small for the requested chronological split")
    return {
        "fit": np.arange(0, fit_end, dtype=np.int64),
        "validation": np.arange(fit_end, train_period_end, dtype=np.int64),
        "simulation": np.arange(train_period_end, n_rows, dtype=np.int64),
    }


def split_summary(frame: pd.DataFrame, positions: dict[str, np.ndarray]) -> pd.DataFrame:
    """Summarize row counts, times, and fraud rates without changing the split."""
    records = []
    for name, pos in positions.items():
        part = frame.iloc[pos]
        records.append(
            {
                "split": name,
                "rows": len(part),
                "first_time": float(part[TIME_COLUMN].min()),
                "last_time": float(part[TIME_COLUMN].max()),
                "first_day": int(part[TIME_COLUMN].min() // 86400) + 1,
                "last_day": int(part[TIME_COLUMN].max() // 86400) + 1,
                "fraud_count": int(part[TARGET].sum()),
                "fraud_rate": float(part[TARGET].mean()),
            }
        )
    return pd.DataFrame(records)


def profile_dataset(frame: pd.DataFrame, output_dir: str | Path | None = None) -> dict:
    """Inspect missingness, dtypes, categorical cardinality, and key distributions."""
    from .features import infer_categorical_columns

    categorical = infer_categorical_columns(frame)
    type_rows = []
    category_rows = []
    for column in frame.columns:
        values = frame[column]
        type_rows.append(
            {
                "column": column,
                "dtype": str(values.dtype),
                "missing_count": int(values.isna().sum()),
                "missing_rate": float(values.isna().mean()),
                "n_unique": int(values.nunique(dropna=True)),
                "categorical_for_models": column in categorical,
            }
        )
        if column in categorical:
            top = values.astype("string").fillna("__MISSING__").value_counts(normalize=True, dropna=False).head(20)
            category_rows.extend(
                {"column": column, "value": str(value), "proportion": float(proportion)}
                for value, proportion in top.items()
            )
    profile = {
        "rows": int(len(frame)),
        "columns": int(frame.shape[1]),
        "fraud_count": int(frame[TARGET].sum()),
        "fraud_rate": float(frame[TARGET].mean()),
        "categorical_columns": categorical,
        "amount_quantiles": {
            str(q): float(frame[AMOUNT_COLUMN].quantile(q)) for q in (0.5, 0.75, 0.95, 0.99, 1.0)
        },
        "feature_profile": type_rows,
    }
    if output_dir is not None:
        out = Path(output_dir)
        out.mkdir(parents=True, exist_ok=True)
        (out / "dataset_profile.json").write_text(json.dumps(profile, indent=2), encoding="utf-8")
        pd.DataFrame(type_rows).to_csv(out / "feature_profile.csv", index=False)
        pd.DataFrame(category_rows).to_csv(out / "categorical_top_values.csv", index=False)
        weekly = frame.assign(week=(frame[TIME_COLUMN] // (7 * 86400)).astype(int)).groupby("week")[TARGET].agg(
            transactions="size", frauds="sum", fraud_rate="mean"
        )
        weekly.to_csv(out / "weekly_fraud_rate.csv")
    return profile

