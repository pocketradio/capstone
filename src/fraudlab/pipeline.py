"""Notebook-friendly model training, feature ablation, and simulation inputs."""

from __future__ import annotations

from pathlib import Path
import json

import numpy as np
import pandas as pd

from .config import ExperimentConfig, save_config
from .data import ID_COLUMN, TARGET, TIME_COLUMN, chronological_positions, split_summary
from .features import FEATURE_BLOCK_ORDER, feature_columns_for_recipe, load_feature_artifact
from .metrics import classification_metrics
from .models import WEIGHT_STRATEGIES, fit_score_model, predict_calibrated
from .policy import tune_threshold_policy


def _fit_recipe(data, metadata, blocks, model_name, positions, config, output_dir, progress=None, recipe_label=None):
    feature_cols = feature_columns_for_recipe(metadata, blocks)
    X = data.loc[:, feature_cols]
    y = data[TARGET].to_numpy(dtype=np.int8)
    val_ix, sim_ix = positions["validation"], positions["simulation"]
    recipe_name = recipe_label or ("control" if not blocks else "plus_".join(blocks))
    recipe_dir = Path(output_dir) / "recipes" / recipe_name
    model_dir = recipe_dir / "models"
    model, weight_search = fit_score_model(
        name=model_name,
        X_fit=X.iloc[positions["fit"]], y_fit=y[positions["fit"]],
        X_validation=X.iloc[val_ix], y_validation=y[val_ix],
        categorical_columns=metadata["categorical_columns"], output_dir=model_dir,
        seed=config.random_seed,
        progress=progress,
        progress_label=f"{model_name}/{recipe_name}",
        catboost_task_type=config.catboost_task_type,
    )
    val_prob = predict_calibrated(model, X.iloc[val_ix])
    sim_prob = predict_calibrated(model, X.iloc[sim_ix])
    val_days = data.iloc[val_ix]["transaction_day"].to_numpy()
    thresholds = tune_threshold_policy(
        val_prob, y[val_ix], data.iloc[val_ix]["TransactionAmt"].to_numpy(), val_days, config.costs,
    )
    pred_dir = recipe_dir / "predictions"
    pred_dir.mkdir(parents=True, exist_ok=True)
    val_pred = data.iloc[val_ix][[ID_COLUMN, TIME_COLUMN, "transaction_day", "TransactionAmt", TARGET]].copy()
    val_pred["probability"] = val_prob
    val_pred.to_csv(pred_dir / f"{model_name}_validation_predictions.csv", index=False)
    sim_pred = data.iloc[sim_ix][[ID_COLUMN, TIME_COLUMN, "transaction_day", "TransactionAmt", TARGET]].copy()
    sim_pred["probability"] = sim_prob
    sim_pred.to_parquet(pred_dir / f"{model_name}_simulation_predictions.parquet", index=False)
    (recipe_dir / f"{model_name}_thresholds.json").write_text(json.dumps(thresholds, indent=2), encoding="utf-8")
    return {
        "model": model_name, "recipe": recipe_name, "blocks": list(blocks),
        "weight_search": weight_search.to_dict(orient="records"),
        "validation_metrics": classification_metrics(y[val_ix], val_prob),
        "validation_total_cost": thresholds["validation_cost"], "thresholds": thresholds,
        "model_path": str(model_dir / f"{model_name}_model.joblib"),
        "simulation_predictions_path": str(pred_dir / f"{model_name}_simulation_predictions.parquet"),
        "thresholds_path": str(recipe_dir / f"{model_name}_thresholds.json"),
    }


SOTA_RECIPE_BLOCKS = {
    "r0_control": (),
    "r1_uid": ("uid",),
    "r2_full_causal": ("frequency", "uid", "aggregates", "amount_parts"),
    "r3_full_causal_identity": ("frequency", "uid", "aggregates", "amount_parts", "identity"),
}


def train_recipe_model_from_artifacts(
    transaction_feature_path: str | Path,
    identity_feature_path: str | Path | None,
    recipe: str,
    model_name: str,
    output_dir: str | Path,
    config: ExperimentConfig | None = None,
) -> dict:
    """Train one fixed model/feature recipe for the parallel experiment grid."""
    if recipe not in SOTA_RECIPE_BLOCKS:
        raise ValueError(f"Unknown recipe {recipe!r}; expected one of {sorted(SOTA_RECIPE_BLOCKS)}")
    config = config or ExperimentConfig()
    blocks = SOTA_RECIPE_BLOCKS[recipe]
    is_identity = "identity" in blocks
    source = identity_feature_path if is_identity else transaction_feature_path
    if source is None:
        raise ValueError(f"Recipe {recipe} requires the identity-joined feature artifact")
    data, metadata = load_feature_artifact(source)
    positions = chronological_positions(data, config.split)
    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)
    split_summary(data, positions).to_csv(out / "split_summary.csv", index=False)
    save_config(config, out / "experiment_config.json")

    from tqdm import tqdm

    from .models import WEIGHT_STRATEGIES

    with tqdm(
        total=len(WEIGHT_STRATEGIES), desc=f"{model_name}/{recipe}", unit="fit",
        bar_format="{desc}: {percentage:3.0f}%|{bar}| {n_fmt}/{total_fmt} {postfix}",
        mininterval=30, dynamic_ncols=False, leave=True,
    ) as progress:
        result = _fit_recipe(
            data, metadata, list(blocks), model_name, positions, config, out,
            progress=progress, recipe_label=recipe,
        )
    (out / "candidate_metadata.json").write_text(json.dumps({
        "model": model_name,
        "recipe": recipe,
        "blocks": list(blocks),
        "feature_artifact": str(source),
        "seed": config.random_seed,
        "validation_metrics": result["validation_metrics"],
        "validation_total_cost": result["validation_total_cost"],
        "thresholds": result["thresholds"],
        "weight_search": result["weight_search"],
    }, indent=2), encoding="utf-8")
    return result


def train_models_from_artifact(
    feature_path: str | Path,
    output_dir: str | Path,
    model_names=("lightgbm",),
    config: ExperimentConfig | None = None,
    run_feature_ablation: bool = True,
) -> dict[str, dict]:
    """Train control/candidate LightGBM recipes, then freeze the selected recipe.

    Feature blocks are added in documented order and retained only when the
    validation-tuned decision cost falls (ties require improved validation PR-AUC).
    The final simulation period is never consulted for recipe selection.
    """
    config = config or ExperimentConfig()
    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)
    data, metadata = load_feature_artifact(feature_path)
    positions = chronological_positions(data, config.split)
    split_summary(data, positions).to_csv(out / "split_summary.csv", index=False)
    save_config(config, out / "experiment_config.json")
    results = {}

    from tqdm import tqdm

    available_blocks = sum(bool(metadata["feature_blocks"].get(block)) for block in FEATURE_BLOCK_ORDER)
    recipe_count = (1 + available_blocks) if run_feature_ablation else len(model_names)
    progress = tqdm(total=recipe_count * len(WEIGHT_STRATEGIES), desc="Training fits", unit="fit")

    selected_blocks = []
    selection_rows = []
    if run_feature_ablation:
        if "lightgbm" not in model_names:
            raise ValueError("Feature ablation must run with LightGBM before other models.")
        current = _fit_recipe(data, metadata, [], "lightgbm", positions, config, out, progress)
        current_cost = current["validation_total_cost"]
        current_pr = current["validation_metrics"]["pr_auc"]
        selection_rows.append({"step": "control", "candidate_block": "", "accepted": True,
                               "validation_total_cost": current_cost, "validation_pr_auc": current_pr,
                               "blocks": ""})
        final_result = current
        for block in FEATURE_BLOCK_ORDER:
            if not metadata["feature_blocks"].get(block):
                selection_rows.append({"step": block, "candidate_block": block, "accepted": False,
                                       "reason": "No source columns available", "blocks": ",".join(selected_blocks)})
                continue
            candidate_blocks = selected_blocks + [block]
            candidate = _fit_recipe(data, metadata, candidate_blocks, "lightgbm", positions, config, out, progress)
            cand_cost = candidate["validation_total_cost"]
            cand_pr = candidate["validation_metrics"]["pr_auc"]
            accepted = cand_cost < current_cost - 1e-9 or (abs(cand_cost - current_cost) <= 1e-9 and cand_pr > current_pr)
            selection_rows.append({"step": block, "candidate_block": block, "accepted": accepted,
                                   "validation_total_cost": cand_cost, "validation_pr_auc": cand_pr,
                                   "blocks": ",".join(candidate_blocks)})
            if accepted:
                selected_blocks = candidate_blocks
                current_cost, current_pr, final_result = cand_cost, cand_pr, candidate
        selected = {"model": "lightgbm", "recipe": final_result["recipe"], "blocks": selected_blocks,
                    "selection_rule": "Sequential forward selection on validation-tuned cost; equal cost accepted only if PR-AUC improves.",
                    "validation_total_cost": current_cost, "validation_pr_auc": current_pr,
                    "feature_sources": metadata["feature_sources"]}
        (out / "selected_feature_recipe.json").write_text(json.dumps(selected, indent=2), encoding="utf-8")
        pd.DataFrame(selection_rows).to_csv(out / "feature_ablation.csv", index=False)
        results["feature_selection"] = selected

    if not run_feature_ablation:
        selection_path = out / "selected_feature_recipe.json"
        if not selection_path.exists():
            raise FileNotFoundError("Run LightGBM feature ablation before training additional models.")
        selected = json.loads(selection_path.read_text(encoding="utf-8"))
        selected_blocks = selected["blocks"]

    # For initial LightGBM ablation, use the already-fitted selected recipe.
    for name in model_names:
        if run_feature_ablation and name == "lightgbm":
            result = final_result
        else:
            result = _fit_recipe(data, metadata, selected_blocks, name, positions, config, out, progress)
        results[name] = result
    (out / "training_summary.json").write_text(json.dumps(results, indent=2), encoding="utf-8")
    progress.close()
    return results


def load_simulation_input(feature_path: str | Path, prediction_path: str | Path, config=None):
    """Load only held-out rows and saved predictions in aligned transaction order."""
    config = config or ExperimentConfig()
    data, _ = load_feature_artifact(feature_path)
    positions = chronological_positions(data, config.split)["simulation"]
    rows = data.iloc[positions].reset_index(drop=True)
    predictions = pd.read_parquet(prediction_path).reset_index(drop=True)
    if len(rows) != len(predictions):
        raise ValueError("Saved predictions do not align with the simulation split")
    if ID_COLUMN in predictions and ID_COLUMN in rows:
        if not np.array_equal(predictions[ID_COLUMN].to_numpy(), rows[ID_COLUMN].to_numpy()):
            raise ValueError("Saved predictions are not in the simulation transaction order")
    return rows, predictions["probability"].to_numpy(dtype=np.float64)
