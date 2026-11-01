"""Secondary one-class anomaly baselines for the approved R2/R3 recipes."""

from __future__ import annotations

from pathlib import Path
import json
import time

import numpy as np
import pandas as pd

from .config import ExperimentConfig
from .data import ID_COLUMN, TARGET, chronological_positions
from .features import feature_columns_for_recipe, load_feature_artifact
from .metrics import classification_metrics
from .policy import tune_threshold_policy
from .pipeline import SOTA_RECIPE_BLOCKS


def train_anomaly_candidate(
    feature_path: str | Path,
    recipe: str,
    model_name: str,
    output_dir: str | Path,
    config: ExperimentConfig | None = None,
) -> dict:
    """Fit IsolationForest or SGDOneClassSVM on genuine fit rows only.

    Numeric features are selected from the approved recipe and imputed/scaled
    using fit-period statistics. The one-class models do not consume fraud
    labels during fitting; validation labels are used only for score calibration
    and the cost threshold selection.
    """
    if recipe not in {"r2_full_causal", "r3_full_causal_identity"}:
        raise ValueError("One-class candidates are restricted to R2 and R3")
    config = config or ExperimentConfig()
    data, metadata = load_feature_artifact(feature_path)
    positions = chronological_positions(data, config.split)
    columns = feature_columns_for_recipe(metadata, SOTA_RECIPE_BLOCKS[recipe])
    cats = set(metadata["categorical_columns"])
    numeric = [c for c in columns if c not in cats]
    if not numeric:
        raise ValueError("No numeric features available for anomaly model")

    x_fit = data.iloc[positions["fit"]][numeric].apply(pd.to_numeric, errors="coerce")
    medians = x_fit.median().replace([np.inf, -np.inf], np.nan).fillna(0.0)
    arrays = {}
    for name, pos in positions.items():
        values = data.iloc[pos][numeric].apply(pd.to_numeric, errors="coerce").replace([np.inf, -np.inf], np.nan)
        arrays[name] = values.fillna(medians).to_numpy(dtype=np.float32)
    y = data[TARGET].to_numpy(dtype=np.int8)
    fit_y = y[positions["fit"]]
    genuine_positions = np.flatnonzero(fit_y == 0)
    from sklearn.preprocessing import StandardScaler

    started = time.time()
    scaler = StandardScaler()
    fit_x = scaler.fit_transform(arrays["fit"][genuine_positions]).astype(np.float32, copy=False)
    eval_x = {name: scaler.transform(values).astype(np.float32, copy=False)
              for name, values in arrays.items() if name != "fit"}
    if model_name == "isolation_forest":
        from sklearn.ensemble import IsolationForest

        estimator = IsolationForest(
            n_estimators=200, max_samples="auto", contamination="auto",
            random_state=config.random_seed, n_jobs=-1,
        )
    elif model_name == "sgd_one_class_svm":
        from sklearn.linear_model import SGDOneClassSVM

        estimator = SGDOneClassSVM(
            nu=0.01, fit_intercept=True, shuffle=True, random_state=config.random_seed,
            max_iter=100, tol=1e-3, learning_rate="optimal", average=True,
        )
    else:
        raise ValueError("model_name must be isolation_forest or sgd_one_class_svm")
    estimator.fit(fit_x)
    val_score = -np.asarray(estimator.decision_function(eval_x["validation"]), dtype=np.float64)
    sim_score = -np.asarray(estimator.decision_function(eval_x["simulation"]), dtype=np.float64)
    from sklearn.isotonic import IsotonicRegression

    val_y = y[positions["validation"]]
    calibrator = IsotonicRegression(y_min=0.0, y_max=1.0, out_of_bounds="clip")
    calibrator.fit(val_score, val_y)
    val_prob = np.clip(calibrator.predict(val_score), 0.0, 1.0)
    sim_prob = np.clip(calibrator.predict(sim_score), 0.0, 1.0)
    val_rows = data.iloc[positions["validation"]]
    thresholds = tune_threshold_policy(
        val_prob, val_y, val_rows["TransactionAmt"].to_numpy(),
        val_rows["transaction_day"].to_numpy(), config.costs,
    )
    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)
    import joblib

    joblib.dump({"estimator": estimator, "scaler": scaler, "medians": medians.to_dict(),
                 "numeric_features": numeric, "calibrator": calibrator}, out / f"{model_name}_model.joblib")
    val = data.iloc[positions["validation"]][[ID_COLUMN, "TransactionDT", "transaction_day", "TransactionAmt", TARGET]].copy()
    val["probability"] = val_prob
    val.to_csv(out / f"{model_name}_validation_predictions.csv", index=False)
    sim = data.iloc[positions["simulation"]][[ID_COLUMN, "TransactionDT", "transaction_day", "TransactionAmt", TARGET]].copy()
    sim["probability"] = sim_prob
    sim.to_parquet(out / f"{model_name}_simulation_predictions.parquet", index=False)
    (out / f"{model_name}_thresholds.json").write_text(json.dumps(thresholds, indent=2), encoding="utf-8")
    result = {
        "model": model_name, "recipe": recipe, "seed": config.random_seed,
        "training_rows": int(len(genuine_positions)), "numeric_features": int(len(numeric)),
        "validation_metrics": classification_metrics(val_y, val_prob),
        "validation_total_cost": thresholds["validation_cost"], "thresholds": thresholds,
        "runtime_seconds": time.time() - started, "target_used_for_fit": False,
    }
    (out / "candidate_metadata.json").write_text(json.dumps(result, indent=2, default=str), encoding="utf-8")
    return result
