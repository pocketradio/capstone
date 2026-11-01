"""Leakage-safe IEEE-CIS feature construction and portable artifact I/O.

The candidate encodings adapt public competition feature ideas to a temporal
replay: every history statistic is computed from strictly earlier timestamps.
"""

from __future__ import annotations

from pathlib import Path
import json

import numpy as np
import pandas as pd

from .data import AMOUNT_COLUMN, ID_COLUMN, TARGET, TIME_COLUMN


FEATURE_BLOCK_ORDER = ("frequency", "uid", "aggregates", "amount_parts", "identity")
FEATURE_SOURCES = {
    "control": "Existing project features: transaction day/hour, log amount, prior counts and amount means.",
    "frequency": "Winner-inspired frequency features for card/address/email identifiers and combinations.",
    "uid": "Winner-inspired card1 + addr1 + D1-derived transaction-day UID, used only for prior counts.",
    "aggregates": "Winner-inspired group statistics (prior amount and D9 mean/std) for card/address/email/UID.",
    "amount_parts": "Winner-inspired amount dollar and cents components.",
    "identity": "Separate identity/device fields joined one-to-one by TransactionID; no target-derived fields.",
}

SEMANTIC_CATEGORICAL = {
    "ProductCD", "card1", "card2", "card3", "card4", "card5", "card6",
    "addr1", "addr2", "P_emaildomain", "R_emaildomain",
    *(f"M{i}" for i in range(1, 10)),
}


def infer_categorical_columns(frame: pd.DataFrame) -> list[str]:
    """Infer categorical fields from string dtype and known IEEE-CIS categories."""
    by_type = set(frame.select_dtypes(include=["object", "category", "string"]).columns)
    return sorted((by_type | (SEMANTIC_CATEGORICAL & set(frame.columns))) - {TARGET})


def _group_codes(data: pd.DataFrame, keys: list[str]) -> pd.Series:
    """Build entity codes while ensuring missing key values never form entities."""
    valid = data[keys].notna().all(axis=1)
    codes = pd.Series(-1, index=data.index, dtype="int64")
    if valid.any():
        codes.loc[valid] = data.loc[valid].groupby(keys, sort=False, observed=True).ngroup().to_numpy()
    return codes


def _prior_count(data: pd.DataFrame, keys: list[str]) -> np.ndarray:
    """Number of prior rows for an entity; equal timestamps are one batch."""
    entity = _group_codes(data, keys)
    valid = entity.ge(0)
    if not valid.any():
        return np.zeros(len(data), dtype=np.float32)
    table = pd.DataFrame({"entity": entity[valid].to_numpy(), "time": data.loc[valid, TIME_COLUMN].to_numpy()})
    table = table.groupby(["entity", "time"], sort=True).size().rename("batch").reset_index()
    table["before"] = table.groupby("entity", sort=False)["batch"].cumsum() - table["batch"]
    index = pd.MultiIndex.from_arrays([entity[valid].to_numpy(), data.loc[valid, TIME_COLUMN].to_numpy()], names=["entity", "time"])
    lookup = table.set_index(["entity", "time"])["before"]
    out = np.zeros(len(data), dtype=np.float32)
    out[np.flatnonzero(valid.to_numpy())] = lookup.reindex(index).to_numpy(dtype=np.float32)
    return out


def _prior_numeric_stats(data: pd.DataFrame, keys: list[str], value_column: str, prefix: str) -> dict[str, np.ndarray]:
    """Prior-only mean and sample standard deviation, excluding timestamp ties."""
    if value_column not in data:
        return {}
    values = pd.to_numeric(data[value_column], errors="coerce").to_numpy(dtype=np.float64)
    entity = _group_codes(data, keys)
    valid = entity.ge(0).to_numpy() & np.isfinite(values)
    result = {
        f"{prefix}_{value_column}_mean_before": np.full(len(data), np.nan, dtype=np.float32),
        f"{prefix}_{value_column}_std_before": np.full(len(data), np.nan, dtype=np.float32),
    }
    if not valid.any():
        return result
    table = pd.DataFrame({
        "entity": entity.to_numpy()[valid],
        "time": data.loc[valid, TIME_COLUMN].to_numpy(),
        "value": values[valid],
    })
    table["value_sq"] = table["value"] * table["value"]
    table = table.groupby(["entity", "time"], sort=True).agg(
        count=("value", "count"), sum=("value", "sum"), sum_sq=("value_sq", "sum")
    )
    group = table.groupby(level=0, sort=False)
    for col in ("count", "sum", "sum_sq"):
        table[f"prior_{col}"] = group[col].cumsum() - table[col]
    n = table["prior_count"].to_numpy(dtype=np.float64)
    total = table["prior_sum"].to_numpy(dtype=np.float64)
    total_sq = table["prior_sum_sq"].to_numpy(dtype=np.float64)
    mean = np.divide(total, n, out=np.full(len(table), np.nan), where=n > 0)
    variance = np.divide(total_sq - np.divide(total * total, n, out=np.zeros(len(table)), where=n > 0),
                         n - 1, out=np.full(len(table), np.nan), where=n > 1)
    table["prior_mean"] = mean
    table["prior_std"] = np.sqrt(np.maximum(variance, 0.0))
    index = pd.MultiIndex.from_arrays([entity.to_numpy()[valid], data.loc[valid, TIME_COLUMN].to_numpy()], names=["entity", "time"])
    lookup_mean = table["prior_mean"]
    lookup_std = table["prior_std"]
    positions = np.flatnonzero(valid)
    result[f"{prefix}_{value_column}_mean_before"][positions] = lookup_mean.reindex(index).to_numpy(dtype=np.float32)
    result[f"{prefix}_{value_column}_std_before"][positions] = lookup_std.reindex(index).to_numpy(dtype=np.float32)
    return result


def add_causal_features(frame: pd.DataFrame) -> tuple[pd.DataFrame, list[str], dict[str, list[str]], list[str]]:
    """Create control and candidate feature blocks using strictly prior timestamps.

    Returns (rows, categorical columns, block-to-column map, control columns).
    Feature computation never reads the target column.
    """
    if TARGET not in frame:
        raise ValueError(f"Expected {TARGET} in source frame; labels are retained only for evaluation")
    data = frame.sort_values([TIME_COLUMN, ID_COLUMN], kind="mergesort").reset_index(drop=True).copy()
    seconds = pd.to_numeric(data[TIME_COLUMN], errors="coerce")
    if seconds.isna().any():
        raise ValueError(f"{TIME_COLUMN} must be present to enforce chronological history")
    seconds = seconds.astype("float64")
    amount = pd.to_numeric(data[AMOUNT_COLUMN], errors="coerce")
    data["transaction_day"] = (seconds // 86400).astype("int32") + 1
    data["transaction_hour"] = ((seconds % 86400) // 3600).astype("int8")
    data["log_amount"] = np.log1p(amount.clip(lower=0))

    block_columns: dict[str, list[str]] = {name: [] for name in FEATURE_BLOCK_ORDER}
    # Control block preserves existing project features, but fixes timestamp ties.
    base_specs = [("card1", ["card1"]), ("card1_addr1", ["card1", "addr1"]),
                  ("email", ["P_emaildomain"])]
    for prefix, keys in base_specs:
        if all(k in data for k in keys):
            count_name = f"{prefix}_transactions_before"
            mean_name = f"{prefix}_amount_mean_before"
            data[count_name] = _prior_count(data, keys)
            data[mean_name] = _prior_numeric_stats(data, keys, AMOUNT_COLUMN, prefix)[f"{prefix}_{AMOUNT_COLUMN}_mean_before"]

    # Distinct frequency candidates beyond the control's card1/card1+addr1/email.
    frequency_specs = [
        ("card2", ["card2"]), ("card3", ["card3"]), ("addr1", ["addr1"]),
        ("card1_addr1_email", ["card1", "addr1", "P_emaildomain"]),
    ]
    for prefix, keys in frequency_specs:
        if all(k in data for k in keys):
            name = f"{prefix}_transactions_before"
            data[name] = _prior_count(data, keys)
            block_columns["frequency"].append(name)

    # UID from card1, addr1, and the transaction day inferred from D1.
    uid_key = None
    if all(k in data for k in ("card1", "addr1", "D1")):
        d1 = pd.to_numeric(data["D1"], errors="coerce")
        uid_day = data["transaction_day"].astype("float64") - d1
        valid_uid = data[["card1", "addr1"]].notna().all(axis=1) & d1.notna()
        uid_key = pd.DataFrame({"card1": data["card1"], "addr1": data["addr1"], "uid_day": uid_day.round(0)}, index=data.index)
        uid_key.loc[~valid_uid, "uid_day"] = np.nan
        uid_count = _prior_count(pd.concat([data[[TIME_COLUMN]], uid_key], axis=1), ["card1", "addr1", "uid_day"])
        data["uid_transactions_before"] = uid_count
        block_columns["uid"].append("uid_transactions_before")

    # Aggregate families. Base control already includes amount means for card1,
    # card1+addr1, and email, so candidates add their std and all D9 stats.
    aggregate_specs = [("card1", ["card1"]), ("card1_addr1", ["card1", "addr1"]),
                       ("card1_addr1_email", ["card1", "addr1", "P_emaildomain"])]
    for prefix, keys in aggregate_specs:
        if not all(k in data for k in keys):
            continue
        for column in (AMOUNT_COLUMN, "D9"):
            stats = _prior_numeric_stats(data, keys, column, prefix)
            for name, values in stats.items():
                if column == AMOUNT_COLUMN and name.endswith("_mean_before"):
                    continue
                data[name] = values
                block_columns["aggregates"].append(name)
    if uid_key is not None:
        temporary = pd.concat([data[[TIME_COLUMN]], uid_key], axis=1)
        for column in (AMOUNT_COLUMN, "D9"):
            stats = _prior_numeric_stats(temporary.assign(**{column: data[column]}), ["card1", "addr1", "uid_day"], column, "uid")
            for name, values in stats.items():
                data[name] = values
                block_columns["aggregates"].append(name)

    valid_amount = amount.notna()
    dollars = "amount_dollars"
    cents = "amount_cents"
    data[dollars] = np.floor(amount).where(valid_amount).astype("float32")
    data[cents] = np.mod(np.round((amount - np.floor(amount)) * 100), 100).where(valid_amount).astype("float32")
    block_columns["amount_parts"].extend([dollars, cents])

    categoricals = infer_categorical_columns(data)
    excluded = {TARGET, ID_COLUMN, TIME_COLUMN}
    feature_columns = [c for c in data.columns if c not in excluded]
    control_columns = [c for c in feature_columns if c not in {v for cols in block_columns.values() for v in cols}]
    for column in categoricals:
        data[column] = data[column].astype("string").fillna("__MISSING__").astype(str)
    return data[feature_columns + [TARGET, TIME_COLUMN, ID_COLUMN]], categoricals, block_columns, control_columns


def save_feature_artifact(frame: pd.DataFrame, categorical_columns: list[str], output_dir: str | Path,
                          feature_blocks: dict[str, list[str]], control_columns: list[str]) -> Path:
    """Save engineered rows and metadata, including candidate-block provenance."""
    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)
    target = out / "ieee_cis_features.parquet"
    frame.to_parquet(target, index=False)
    (out / "feature_metadata.json").write_text(json.dumps({
        "categorical_columns": categorical_columns,
        "feature_columns": [c for c in frame.columns if c not in {TARGET, TIME_COLUMN, ID_COLUMN}],
        "control_columns": control_columns,
        "feature_blocks": feature_blocks,
        "feature_sources": FEATURE_SOURCES,
        "history_rule": "Strictly earlier TransactionDT; same-timestamp rows do not contribute to each other.",
        "target_usage": "isFraud excluded from all feature construction and encodings.",
    }, indent=2), encoding="utf-8")
    return target


def join_identity_features(
    transaction_features: pd.DataFrame,
    identity_frame: pd.DataFrame,
    categorical_columns: list[str],
    feature_blocks: dict[str, list[str]],
    control_columns: list[str],
) -> tuple[pd.DataFrame, list[str], dict[str, list[str]], list[str]]:
    """Left-join identity fields without changing transaction rows or order.

    Identity is kept as its own optional feature block. The presence indicator
    distinguishes an absent identity record from missing fields within a record.
    """
    if ID_COLUMN not in transaction_features or ID_COLUMN not in identity_frame:
        raise ValueError(f"Both tables must contain {ID_COLUMN}")
    if identity_frame[ID_COLUMN].duplicated().any():
        raise ValueError(f"Identity data contains duplicate {ID_COLUMN} values")
    identity_columns = [c for c in identity_frame.columns if c != ID_COLUMN]
    collisions = set(identity_columns) & set(transaction_features.columns)
    if collisions:
        raise ValueError(f"Identity columns collide with transaction features: {sorted(collisions)}")

    identity = identity_frame[[ID_COLUMN] + identity_columns].copy()
    identity["identity_present"] = np.int8(1)
    joined = transaction_features.merge(identity, on=ID_COLUMN, how="left", sort=False, validate="one_to_one")
    if len(joined) != len(transaction_features):
        raise AssertionError("Identity join changed the transaction row count")
    if not np.array_equal(joined[ID_COLUMN].to_numpy(), transaction_features[ID_COLUMN].to_numpy()):
        raise AssertionError("Identity join changed transaction order")
    joined["identity_present"] = joined["identity_present"].fillna(0).astype("int8")

    cats = infer_categorical_columns(joined)
    for column in cats:
        joined[column] = joined[column].astype("string").fillna("__MISSING__").astype(str)
    blocks = {name: list(values) for name, values in feature_blocks.items()}
    blocks.setdefault("identity", [])
    blocks["identity"] = identity_columns + ["identity_present"]
    # `control_columns` intentionally remains transaction-only.
    return joined, cats, blocks, list(control_columns)


def load_feature_artifact(path: str | Path) -> tuple[pd.DataFrame, dict]:
    """Load engineered rows and their feature recipe metadata."""
    source = Path(path)
    frame = pd.read_parquet(source)
    metadata = json.loads((source.parent / "feature_metadata.json").read_text(encoding="utf-8"))
    return frame, metadata


def feature_columns_for_recipe(metadata: dict, blocks: list[str] | tuple[str, ...] = ()) -> list[str]:
    """Return base control columns plus the requested candidate blocks."""
    unknown = set(blocks) - set(metadata["feature_blocks"])
    if unknown:
        raise ValueError(f"Unknown feature blocks: {sorted(unknown)}")
    columns = list(metadata["control_columns"])
    for block in blocks:
        columns.extend(metadata["feature_blocks"][block])
    return list(dict.fromkeys(columns))
