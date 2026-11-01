"""CPU model fitting, validation-only calibration, and prediction artifacts."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import json

import numpy as np
import pandas as pd

from .metrics import classification_metrics


WEIGHT_STRATEGIES = ("none", "sqrt", "balanced")


@dataclass
class FittedScoreModel:
    """A fitted base estimator plus its validation-trained calibrator."""

    name: str
    weight_strategy: str
    estimator: object
    calibrator: object
    categorical_columns: list[str]
    numeric_columns: list[str]
    encoder_state: dict
    feature_columns: list[str]
    validation_metrics: dict


def _require(package: str, import_name: str | None = None):
    """Import an optional training dependency with an actionable error."""
    import importlib

    try:
        return importlib.import_module(import_name or package)
    except ImportError as exc:
        raise ImportError(
            f"Training {package} requires its optional package. Install requirements.txt first."
        ) from exc


def _weight_value(strategy: str, y: np.ndarray) -> float:
    """Return positive-class weight for a requested strategy."""
    positives = max(int(np.count_nonzero(y == 1)), 1)
    negatives = max(int(np.count_nonzero(y == 0)), 1)
    ratio = negatives / positives
    if strategy == "none":
        return 1.0
    if strategy == "sqrt":
        return float(np.sqrt(ratio))
    if strategy == "balanced":
        return float(ratio)
    raise ValueError(f"Unsupported class-weight strategy: {strategy}")


def _prepare_native_frames(X_fit, X_eval, cat_columns, native_category=True):
    """Apply train-fitted category vocabularies and numeric imputations."""
    X_fit = X_fit.copy()
    X_eval = X_eval.copy()
    cat_columns = list(cat_columns)
    numeric = [c for c in X_fit.columns if c not in cat_columns]
    state = {"category_levels": {}, "numeric_medians": {}}
    for column in cat_columns:
        levels = sorted(X_fit[column].fillna("__MISSING__").astype(str).unique().tolist())
        if "__UNKNOWN__" not in levels:
            levels.append("__UNKNOWN__")
        state["category_levels"][column] = levels
        for frame in (X_fit, X_eval):
            values = frame[column].fillna("__MISSING__").astype(str)
            values = values.where(values.isin(levels), "__UNKNOWN__")
            frame[column] = pd.Categorical(values, categories=levels) if native_category else values.astype(str)
    for column in numeric:
        med = float(pd.to_numeric(X_fit[column], errors="coerce").median())
        if not np.isfinite(med):
            med = 0.0
        state["numeric_medians"][column] = med
        for frame in (X_fit, X_eval):
            frame[column] = pd.to_numeric(frame[column], errors="coerce").fillna(med).astype("float32")
    return X_fit, X_eval, numeric, state


def _prepare_xgb_frames(X_fit, frames, cat_columns):
    """One-hot low-cardinality categories; omit high-cardinality raw IDs.

    Their time-safe prior-count features are already part of the selected recipe.
    """
    X_fit = X_fit.copy()
    cat_columns = list(cat_columns)
    numeric = [c for c in X_fit.columns if c not in cat_columns]
    state = {"category_levels": {}, "numeric_medians": {}, "output_columns": []}
    for c in numeric:
        med = float(pd.to_numeric(X_fit[c], errors="coerce").median())
        if not np.isfinite(med):
            med = 0.0
        state["numeric_medians"][c] = med
    for c in cat_columns:
        values = X_fit[c].fillna("__MISSING__").astype(str)
        levels = sorted(values.unique().tolist())
        state["category_levels"][c] = levels
        state.setdefault("low_cardinality", {})[c] = len(levels) <= 20
    def transform(frame):
        result = pd.DataFrame(index=frame.index)
        for c in numeric:
            result[c] = pd.to_numeric(frame[c], errors="coerce").fillna(state["numeric_medians"][c]).astype("float32")
        for c in cat_columns:
            values = frame[c].fillna("__MISSING__").astype(str)
            levels = state["category_levels"][c]
            if state["low_cardinality"][c]:
                for i, value in enumerate(levels):
                    result[f"cat_{cat_columns.index(c)}_{i}"] = (values == value).astype("int8")
        return result
    fit_out = transform(X_fit)
    state["output_columns"] = fit_out.columns.tolist()
    transformed = [fit_out]
    for frame in frames:
        value = transform(frame).reindex(columns=state["output_columns"], fill_value=0)
        transformed.append(value)
    return transformed, state


def _make_estimator(name: str, weight_strategy: str, positive_weight: float, categories: list[str], seed: int):
    """Construct a CPU estimator for one model and class-weight candidate."""
    if name == "lightgbm":
        lgb = _require("lightgbm")
        return lgb.LGBMClassifier(
            objective="binary", metric="None", n_estimators=1600, learning_rate=0.035, num_leaves=31,
            max_depth=-1, min_child_samples=80, subsample=0.85, subsample_freq=1,
            colsample_bytree=0.85, reg_lambda=1.0, scale_pos_weight=positive_weight,
            random_state=seed, deterministic=True, force_col_wise=True,
            n_jobs=-1, device_type="cpu", verbosity=-1,
        )
    if name == "xgboost":
        xgb = _require("xgboost")
        return xgb.XGBClassifier(
            objective="binary:logistic", eval_metric="aucpr", n_estimators=1200,
            learning_rate=0.04, max_depth=8, min_child_weight=10, subsample=0.85,
            colsample_bytree=0.85, reg_lambda=1.0, scale_pos_weight=positive_weight,
            tree_method="hist", n_jobs=-1, random_state=seed, early_stopping_rounds=100,
        )
    if name == "catboost":
        cat = _require("catboost")
        return cat.CatBoostClassifier(
            loss_function="Logloss", eval_metric="PRAUC:type=Classic", iterations=1600,
            learning_rate=0.04, depth=8, l2_leaf_reg=5.0,
            class_weights=[1.0, positive_weight], random_seed=seed,
            thread_count=-1, verbose=False, allow_writing_files=False,
        )
    raise ValueError(f"Unknown model name: {name}")


def fit_score_model(
    name: str,
    X_fit: pd.DataFrame,
    y_fit,
    X_validation: pd.DataFrame,
    y_validation,
    categorical_columns: list[str],
    output_dir: str | Path,
    seed: int = 42,
    weight_strategies=WEIGHT_STRATEGIES,
    progress=None,
    progress_label: str = "",
) -> tuple[FittedScoreModel, pd.DataFrame]:
    """Select weighting on validation PR-AUC and fit a validation-only isotonic calibrator."""
    from sklearn.isotonic import IsotonicRegression
    from sklearn.metrics import average_precision_score
    import joblib

    name = name.lower()
    y_fit = np.asarray(y_fit, dtype=np.int8)
    y_validation = np.asarray(y_validation, dtype=np.int8)
    categorical_columns = [c for c in categorical_columns if c in X_fit.columns]
    numeric_columns = [c for c in X_fit.columns if c not in categorical_columns]
    X_fit = X_fit.reset_index(drop=True)
    X_validation = X_validation.reset_index(drop=True)

    if name == "xgboost":
        (fit_x, val_x), encoder_state = _prepare_xgb_frames(X_fit, [X_validation], categorical_columns)
        model_cat_columns = []
    else:
        fit_x, val_x, numeric_columns, encoder_state = _prepare_native_frames(
            X_fit, X_validation, categorical_columns, native_category=(name == "lightgbm")
        )
        model_cat_columns = categorical_columns

    results = []
    candidates = []
    def lightgbm_pr_auc(y_true, y_score):
        """Provide a version-stable validation PR-AUC metric for LightGBM."""
        return "pr_auc", float(average_precision_score(y_true, y_score)), True

    for strategy in weight_strategies:
        if progress is not None:
            progress.set_postfix_str(f"{progress_label} | weight={strategy}")
        weight = _weight_value(strategy, y_fit)
        estimator = _make_estimator(name, strategy, weight, model_cat_columns, seed)
        if name == "lightgbm":
            estimator.fit(
                fit_x, y_fit, categorical_feature=model_cat_columns,
                eval_set=[(val_x, y_validation)], eval_metric=lightgbm_pr_auc,
                callbacks=[_require("lightgbm").early_stopping(100, verbose=False)],
            )
        elif name == "xgboost":
            estimator.fit(
                fit_x, y_fit, eval_set=[(val_x, y_validation)], verbose=False,
            )
        else:
            estimator.fit(
                fit_x, y_fit, cat_features=model_cat_columns,
                eval_set=(val_x, y_validation), early_stopping_rounds=100,
                verbose=False,
            )
        raw = np.asarray(estimator.predict_proba(val_x)[:, 1], dtype=np.float64)
        if progress is not None:
            progress.update(1)
        score = float(average_precision_score(y_validation, raw))
        results.append({"model": name, "weight_strategy": strategy, "positive_weight": weight, "validation_pr_auc": score})
        candidates.append((score, strategy, estimator, raw, weight))

    _, best_strategy, best_estimator, val_raw, best_weight = max(candidates, key=lambda x: x[0])
    calibrator = IsotonicRegression(y_min=0.0, y_max=1.0, out_of_bounds="clip")
    calibrator.fit(val_raw, y_validation)
    val_calibrated = np.asarray(calibrator.predict(val_raw), dtype=np.float64)
    val_metrics = classification_metrics(y_validation, val_calibrated)
    fitted = FittedScoreModel(
        name=name,
        weight_strategy=best_strategy,
        estimator=best_estimator,
        calibrator=calibrator,
        categorical_columns=categorical_columns,
        numeric_columns=numeric_columns,
        encoder_state=encoder_state,
        feature_columns=X_fit.columns.tolist(),
        validation_metrics=val_metrics,
    )
    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)
    joblib.dump(fitted, out / f"{name}_model.joblib")
    (out / f"{name}_model_metadata.json").write_text(
        json.dumps({
            "model": name,
            "selected_weight_strategy": best_strategy,
            "positive_weight": best_weight,
            "categorical_columns": categorical_columns,
            "feature_columns": X_fit.columns.tolist(),
            "validation_metrics": val_metrics,
            "calibration": "isotonic fit on validation predictions only",
        }, indent=2),
        encoding="utf-8",
    )
    pd.DataFrame(results).to_csv(out / f"{name}_weight_search.csv", index=False)
    pd.DataFrame({"y_true": y_validation, "probability_raw": val_raw, "probability_calibrated": val_calibrated}).to_csv(
        out / f"{name}_validation_predictions.csv", index=False
    )
    return fitted, pd.DataFrame(results)


def predict_calibrated(model: FittedScoreModel, X: pd.DataFrame) -> np.ndarray:
    """Return calibrated scores using only the fitted model's saved preprocessing state."""
    X = X.loc[:, model.feature_columns].copy()
    if model.name == "xgboost":
        # Reuse the training encoder metadata; categories unseen during fitting map to all-zero/zero frequency.
        state = model.encoder_state
        result = pd.DataFrame(index=X.index)
        for c in model.numeric_columns:
            med = state["numeric_medians"][c]
            result[c] = pd.to_numeric(X[c], errors="coerce").fillna(med).astype("float32")
        for i, c in enumerate(model.categorical_columns):
            values = X[c].fillna("__MISSING__").astype(str)
            levels = state["category_levels"][c]
            if state["low_cardinality"][c]:
                for j, level in enumerate(levels):
                    result[f"cat_{i}_{j}"] = (values == level).astype("int8")
        X_model = result.reindex(columns=state["output_columns"], fill_value=0)
    else:
        X_model = X.copy()
        for c in model.categorical_columns:
            levels = model.encoder_state["category_levels"][c]
            values = X_model[c].fillna("__MISSING__").astype(str)
            values = values.where(values.isin(levels), "__UNKNOWN__")
            X_model[c] = pd.Categorical(values, categories=levels) if model.name == "lightgbm" else values.astype(str)
        for c, med in model.encoder_state["numeric_medians"].items():
            X_model[c] = pd.to_numeric(X_model[c], errors="coerce").fillna(med).astype("float32")
    raw = np.asarray(model.estimator.predict_proba(X_model)[:, 1], dtype=np.float64)
    return np.clip(np.asarray(model.calibrator.predict(raw), dtype=np.float64), 0.0, 1.0)

