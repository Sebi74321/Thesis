"""CLI runner for the A0/A1/A2/A3/A4/A5 weighted-retraining experiment."""

from __future__ import annotations

import argparse
import hashlib
from importlib import metadata as importlib_metadata
import json
import os
import re
import shutil
import tempfile
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Iterable, List

import numpy as np
import pandas as pd

from .augmentation import create_uniform_augmentation, create_weighted_augmentation
from .data_split import create_data_splits
from .detector import train_detector
from .diagnostics import baseline_detector_diagnostics
from .evaluation import _cat, evaluate_variant
from .io_utils import atomic_write_csv, atomic_write_json, combined_sha256, file_sha256
from .mixed_utility import (
    evaluate_mixed_utility_curve,
    evaluate_real_only_baseline,
    summarize_mixed_utility,
)
from .priority_diagnostics import (
    feature_exclusion_sensitivity,
    prioritized_feature_diagnostics,
    top_shap_feature_variant_metrics,
)
from .scoring import (
    build_region_definitions,
    correlation_groups,
    compute_feature_components,
    compute_feature_priority,
    compute_row_weights,
    weight_diagnostics,
)
from .utility_reporting import save_ablation_heatmap_artifacts

VALID_VARIANTS = ("A0", "A1", "A2", "A3", "A4", "A5")
VALID_GENERATORS = ("ctabgan_plus", "ctgan", "dp_cgan")


def _dp_transformer_context(
    train: pd.DataFrame,
    train_indices: Iterable[int],
    categorical_columns: Iterable[str],
    source_data_sha256: str,
) -> Dict[str, Any]:
    """Describe everything that makes a fitted DP-CGAN transformer reusable."""
    serialized_indices = json.dumps(
        [int(index) for index in train_indices], separators=(",", ":")
    ).encode("utf-8")
    try:
        package_version = importlib_metadata.version("dp-cgans")
    except importlib_metadata.PackageNotFoundError:
        package_version = "unavailable"
    return {
        "source_data_sha256": str(source_data_sha256),
        "train_indices_sha256": hashlib.sha256(serialized_indices).hexdigest(),
        "training_rows": int(len(train)),
        "columns": [str(column) for column in train.columns],
        "dtypes": {str(column): str(dtype) for column, dtype in train.dtypes.items()},
        "categorical_columns": [str(column) for column in categorical_columns],
        "dp_cgans_version": package_version,
    }


def _atomic_copy_file(source: Path, destination: Path) -> None:
    """Copy a binary artifact without exposing a partial destination file."""
    if not source.is_file():
        raise FileNotFoundError(f"Required source artifact does not exist: {source}")
    destination.parent.mkdir(parents=True, exist_ok=True)
    fd, temporary = tempfile.mkstemp(
        prefix=destination.name, suffix=".tmp", dir=destination.parent
    )
    os.close(fd)
    try:
        shutil.copyfile(source, temporary)
        os.replace(temporary, destination)
    finally:
        if os.path.exists(temporary):
            os.unlink(temporary)


def _freeze_or_validate_dp_transformer(
    source: Path,
    frozen: Path,
    manifest_path: Path,
    context: Dict[str, Any],
) -> Path:
    """Create once, then strictly validate, the run-scoped A0 transformer."""
    if frozen.is_file() or manifest_path.is_file():
        if not frozen.is_file() or not manifest_path.is_file():
            raise RuntimeError(
                "DP-CGAN transformer cache is incomplete; both the transformer "
                "and compatibility manifest are required"
            )
        stored = json.loads(manifest_path.read_text(encoding="utf-8"))
        stored_context = stored.get("compatibility")
        if stored_context != context:
            raise ValueError(
                "DP-CGAN transformer compatibility mismatch: dataset split, schema, "
                "categorical columns, or package version changed"
            )
        actual_hash = file_sha256(frozen)
        if stored.get("transformer_sha256") != actual_hash:
            raise ValueError("Frozen DP-CGAN transformer hash does not match its manifest")
        return frozen.resolve()

    _atomic_copy_file(source, frozen)
    atomic_write_json(
        manifest_path,
        {
            "source_variant": "A0",
            "reuse_scope": "A1-A5_within_this_ablation_run",
            "compatibility": context,
            "transformer_sha256": file_sha256(frozen),
            "transformer_path": str(frozen.resolve()),
        },
    )
    return frozen.resolve()


def _numeric_metric_delta(current: Dict[str, Any], reference: Dict[str, Any]) -> Dict[str, float]:
    """Return numeric metric differences shared by two result dictionaries."""
    result: Dict[str, float] = {}
    for key, value in current.items():
        reference_value = reference.get(key)
        if (
            isinstance(value, (int, float))
            and not isinstance(value, bool)
            and isinstance(reference_value, (int, float))
            and not isinstance(reference_value, bool)
        ):
            result[key] = float(value) - float(reference_value)
    return result


def _build_controlled_deltas(
    summary: pd.DataFrame,
    metrics_by_variant: Dict[str, Dict[str, Any]],
    real_only_utility: pd.DataFrame,
    utility_tasks: List[Dict[str, Any]],
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Attach deltas using only A0 and real-data references."""
    summary = summary.copy()
    delta_rows: List[Dict[str, Any]] = []
    updates_by_variant: Dict[str, Dict[str, Any]] = {
        str(variant): {} for variant in summary["variant"]
    }

    if "A0" in metrics_by_variant:
        a0 = metrics_by_variant["A0"]
        for variant, current in metrics_by_variant.items():
            differences = _numeric_metric_delta(current, a0)
            delta_rows.append({
                "comparison": f"{variant}-A0",
                "variant": variant,
                "control": "A0",
                "reference_type": "synthetic_baseline",
                **{f"delta_{key}": value for key, value in differences.items()},
            })
            updates = updates_by_variant.setdefault(variant, {})
            updates["comparison_vs_A0"] = f"{variant}-A0"
            for key, value in differences.items():
                updates[f"delta_{key}_vs_A0"] = value

    if not real_only_utility.empty:
        excluded = {
            "synthetic_fraction", "synthetic_share_of_training", "repeat", "seed",
            "real_training_rows", "synthetic_training_rows", "training_rows",
        }
        for variant, current in metrics_by_variant.items():
            real_row: Dict[str, Any] = {
                "comparison": f"{variant}-REAL",
                "variant": variant,
                "control": "REAL",
                "reference_type": "real_data_utility_baseline",
            }
            updates = updates_by_variant.setdefault(variant, {})
            updates["comparison_vs_real"] = f"{variant}-REAL"
            for task in utility_tasks:
                task_name = str(task["name"])
                task_rows = real_only_utility[
                    real_only_utility["utility_task"].astype(str) == task_name
                ]
                if task_rows.empty:
                    continue
                numeric_means = task_rows.select_dtypes(include="number").mean()
                for metric, reference_value in numeric_means.items():
                    if metric in excluded:
                        continue
                    result_key = f"utility_{task_name}_{metric}"
                    current_value = current.get(result_key)
                    if not isinstance(current_value, (int, float)) or isinstance(current_value, bool):
                        continue
                    delta = float(current_value) - float(reference_value)
                    real_row[f"delta_{result_key}"] = delta
                    updates[f"real_baseline_{result_key}"] = float(reference_value)
                    updates[f"delta_{result_key}_vs_real"] = delta
            delta_rows.append(real_row)

    # Add the many derived columns in one operation. Repeated ``.loc`` writes
    # insert one pandas block per column and cause severe frame fragmentation.
    updates = pd.DataFrame(
        [updates_by_variant.get(str(variant), {}) for variant in summary["variant"]],
        index=summary.index,
    )
    if not updates.empty:
        summary = pd.concat(
            [summary.drop(columns=updates.columns, errors="ignore"), updates], axis=1
        )

    return summary, pd.DataFrame(delta_rows)


def _load_config(path: Path) -> Dict[str, Any]:
    path = path.resolve()
    with path.open(encoding="utf-8") as handle:
        config = json.load(handle)
    base_name = config.pop("base_config", None)
    if base_name:
        base = _load_config(path.parent / str(base_name))
        generator_override = config.pop("generator", None)
        base.update(config)
        if generator_override is not None:
            base["generator"] = generator_override
        config = base
    required = {"data_path", "target_col", "categorical_cols", "generator"}
    missing = required - set(config)
    if missing:
        raise ValueError(f"Config is missing required fields: {sorted(missing)}")
    generator_name = str(config.get("generator_name", "ctabgan_plus")).strip().lower()
    if generator_name not in VALID_GENERATORS:
        raise ValueError(
            f"Unknown generator_name {generator_name!r}; choose from {VALID_GENERATORS}"
        )
    if generator_name == "dp_cgan" and config["generator"].get("private", False) is not False:
        raise ValueError("dp_cgan is a non-private baseline; set generator.private=false")
    if generator_name == "dp_cgan":
        reuse_transformer = config["generator"].get(
            "reuse_transformer_across_variants", False
        )
        if not isinstance(reuse_transformer, bool):
            raise ValueError(
                "generator.reuse_transformer_across_variants must be true or false"
            )
        if reuse_transformer and config["generator"].get("saved_transformer"):
            raise ValueError(
                "Ablation-managed transformer reuse requires saved_transformer=null; "
                "A0 must fit the frozen transformer from real_train"
            )
    config["generator_name"] = generator_name
    config["generator"].setdefault(
        "categorical_columns", list(config["categorical_cols"])
    )
    return config


def _code_hash(project_root: Path) -> str:
    paths = list((project_root / "xai_reweighting").glob("*.py"))
    paths.extend(
        [
            project_root / "model" / "synthesizer" / "ctabgan_synthesizer.py",
            project_root / "model" / "pipeline" / "data_preparation.py",
        ]
    )
    return combined_sha256(path for path in paths if path.exists())


def _fingerprint(config: Dict[str, Any], data_hash: str, code_hash: str) -> str:
    payload = json.dumps(
        {"config": config, "data_hash": data_hash, "code_hash": code_hash}, sort_keys=True
    ).encode()
    return hashlib.sha256(payload).hexdigest()


def _adapter(config, device, seed, progress, progress_label, work_dir=None):
    from .generator_adapters import create_generator

    return create_generator(
        config.get("generator_name", "ctabgan_plus"),
        dict(config["generator"]),
        device=device,
        seed=seed,
        deterministic=config.get("deterministic", True),
        allow_tf32=config.get("allow_tf32", False),
        progress=progress,
        progress_label=progress_label,
        work_dir=work_dir,
    )


def _prepare_variant_data(variant, train, priorities, definitions, weighting, seed, output_dir):
    if variant == "A0":
        return train.copy(deep=True), {}, None
    gamma = float(weighting["gamma"])
    if variant == "A1":
        augmented, counts = create_uniform_augmentation(train, gamma, seed)
        return augmented, counts, None

    priority = priorities[variant]
    weights, contributions = compute_row_weights(
        train,
        priority,
        definitions,
        alpha=float(weighting["alpha"]),
        w_max=float(weighting["w_max"]),
    )
    diagnostics = weight_diagnostics(weights, float(weighting["w_max"]))
    alpha = float(weighting["alpha"])
    diagnostics["warning_fraction_capped"] = diagnostics["fraction_capped"] > 0.10
    diagnostics["warning_mean_outside_recommended_range"] = not (
        1.0 + 0.10 * alpha <= diagnostics["mean"] <= 1.0 + 0.30 * alpha
    )
    diagnostics["warning_median_far_from_one"] = diagnostics["50%"] > 1.0 + 0.25 * alpha
    diagnostics["warning_nearly_uniform"] = diagnostics["std"] < 1e-6
    atomic_write_json(output_dir / f"row_weight_summary_{variant}.json", diagnostics)
    atomic_write_csv(
        output_dir / f"row_weights_{variant}.csv",
        pd.concat([weights, contributions], axis=1),
    )
    augmented, counts = create_weighted_augmentation(train, weights, gamma, seed)
    return augmented, counts, diagnostics


def _training_diagnostics(history: pd.DataFrame) -> Dict[str, Any]:
    if history.empty:
        return {"status": "unavailable", "epochs_completed": 0}
    loss_columns = [
        column
        for column in history.columns
        if column not in {"epoch", "elapsed_seconds"} and history[column].notna().any()
    ]
    tail_rows = max(1, int(len(history) * 0.2))
    diagnostics: Dict[str, Any] = {
        "status": "no_instability_detected",
        "epochs_completed": int(len(history)),
        "tail_epochs": int(tail_rows),
        "losses": {},
    }
    unstable_tail = False
    extreme_loss = False
    for column in loss_columns:
        values = history[column].dropna().to_numpy(dtype=float)
        tail = values[-tail_rows:]
        x = np.arange(len(tail), dtype=float)
        slope = float(np.polyfit(x, tail, 1)[0]) if len(tail) > 1 else 0.0
        scale = float(np.mean(np.abs(tail)))
        relative_std = float(np.std(tail) / max(scale, 1e-8))
        maximum = float(np.max(np.abs(values)))
        diagnostics["losses"][column] = {
            "initial": float(values[0]),
            "final": float(values[-1]),
            "tail_mean": float(np.mean(tail)),
            "tail_std": float(np.std(tail)),
            "tail_relative_std": relative_std,
            "tail_slope_per_epoch": slope,
            "max_absolute": maximum,
        }
        if column not in {"discriminator_real", "discriminator_fake", "generator"}:
            unstable_tail = unstable_tail or relative_std > 2.0
        extreme_loss = extreme_loss or maximum > 1e4
    diagnostics["warning_unstable_tail"] = unstable_tail
    diagnostics["warning_extreme_loss"] = extreme_loss
    if unstable_tail or extreme_loss:
        diagnostics["status"] = "review"
    return diagnostics


def _atomic_checkpoint(model, path: Path) -> bool:
    save = getattr(model, "save_checkpoint", None)
    if save is None:
        return False
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, temporary = tempfile.mkstemp(prefix=path.name, suffix=".tmp", dir=path.parent)
    os.close(fd)
    try:
        save(Path(temporary))
        os.replace(temporary, path)
    finally:
        if os.path.exists(temporary):
            os.unlink(temporary)
    return True


def _save_training_artifacts(
    model,
    variant: str,
    output_dir: Path,
    *,
    generator_name: str = "ctabgan_plus",
    generator_config: Dict[str, Any] | None = None,
    training_rows: int | None = None,
) -> Dict[str, Any]:
    history = getattr(model, "training_history", pd.DataFrame())
    if not isinstance(history, pd.DataFrame):
        history = pd.DataFrame(history)
    diagnostics = _training_diagnostics(history)
    atomic_write_csv(output_dir / f"training_history_{variant}.csv", history)

    mixture = []
    columns = list(getattr(model, "columns", []) or [])
    for raw in getattr(model, "mixture_diagnostics", []):
        item = dict(raw)
        index = int(item["column_index"])
        item["feature"] = columns[index] if 0 <= index < len(columns) else None
        mixture.append(item)
    atomic_write_json(output_dir / f"mixture_diagnostics_{variant}.json", mixture)
    decimals = getattr(model, "decimals", None)
    if decimals:
        atomic_write_json(output_dir / f"measurement_precision_{variant}.json", decimals)
    numeric_constraints = getattr(model, "numeric_constraints", None)
    if numeric_constraints:
        atomic_write_json(
            output_dir / f"numeric_postprocessing_{variant}.json", numeric_constraints
        )
    diagnostics["mixture_all_converged"] = (
        all(item.get("converged", False) for item in mixture) if mixture else None
    )
    convergence_warnings = list(getattr(model, "convergence_warnings", []))
    diagnostics["convergence_warnings"] = convergence_warnings
    diagnostics["convergence_warning_count"] = len(convergence_warnings)
    diagnostics["generator_name"] = generator_name
    if generator_name == "dp_cgan":
        diagnostics["differential_privacy_enabled"] = False
        diagnostics["backend_mode"] = "non_private_baseline"
        diagnostics["transformer_reused"] = bool(
            getattr(model, "transformer_reused", False)
        )
        reused_path = getattr(model, "saved_transformer_path", None)
        diagnostics["saved_transformer"] = (
            str(reused_path) if reused_path is not None else None
        )
        diagnostics["saved_transformer_sha256"] = (
            file_sha256(Path(reused_path))
            if reused_path is not None and Path(reused_path).is_file()
            else None
        )
    if training_rows is not None:
        diagnostics["training_rows"] = int(training_rows)
    settings = generator_config or {}
    for key in ("epochs", "batch_size", "discriminator_steps", "pac"):
        if key in settings:
            diagnostics[f"{key}_configured"] = settings[key]
    if training_rows is not None and settings.get("batch_size"):
        steps_per_epoch = max(int(training_rows) // int(settings["batch_size"]), 1)
        epochs = int(settings.get("epochs", 0))
        diagnostics["steps_per_epoch"] = steps_per_epoch
        diagnostics["generator_updates"] = steps_per_epoch * epochs
        if "discriminator_steps" in settings:
            diagnostics["discriminator_updates"] = (
                diagnostics["generator_updates"] * int(settings["discriminator_steps"])
            )
    atomic_write_json(
        output_dir / f"convergence_warnings_{variant}.json", convergence_warnings
    )
    upstream_stdout = str(getattr(model, "upstream_stdout", "") or "")
    if upstream_stdout:
        log_path = output_dir / f"upstream_training_{variant}.log"
        fd, temporary = tempfile.mkstemp(prefix=log_path.name, suffix=".tmp", dir=output_dir)
        try:
            with os.fdopen(fd, "w", encoding="utf-8") as handle:
                handle.write(upstream_stdout)
            os.replace(temporary, log_path)
        finally:
            if os.path.exists(temporary):
                os.unlink(temporary)
    atomic_write_json(output_dir / f"training_diagnostics_{variant}.json", diagnostics)
    return diagnostics


def _fit_and_save_training(
    model,
    data: pd.DataFrame,
    variant: str,
    output_dir: Path,
    *,
    generator_name: str = "ctabgan_plus",
    generator_config: Dict[str, Any] | None = None,
):
    try:
        model.fit(data)
    except Exception:
        _save_training_artifacts(
            model,
            variant,
            output_dir,
            generator_name=generator_name,
            generator_config=generator_config,
            training_rows=len(data),
        )
        raise
    diagnostics = _save_training_artifacts(
        model,
        variant,
        output_dir,
        generator_name=generator_name,
        generator_config=generator_config,
        training_rows=len(data),
    )
    suffix = ".pt" if generator_name == "ctabgan_plus" else ".pkl"
    diagnostics["checkpoint_saved"] = _atomic_checkpoint(
        model, output_dir / f"model_checkpoint_{variant}{suffix}"
    )
    atomic_write_json(output_dir / f"training_diagnostics_{variant}.json", diagnostics)
    return diagnostics


def _save_postprocessing_diagnostics(model, output_dir: Path, variant: str) -> None:
    diagnostics = getattr(model, "last_postprocessing_diagnostics", None)
    if diagnostics:
        atomic_write_json(
            output_dir / f"postprocessing_diagnostics_{variant}.json", diagnostics
        )


def _save_discriminator_shap_artifacts(
    model,
    real_probe: pd.DataFrame,
    synthetic_probe: pd.DataFrame,
    output_dir: Path,
    variant: str,
    config: Dict[str, Any],
    generator_name: str,
    seed: int,
) -> Dict[str, Any] | None:
    """Evaluate CTAB+'s internal discriminator snapshots when configured."""
    shap_config = config.get("discriminator_shap", {})
    if not shap_config.get("enabled", False) or generator_name != "ctabgan_plus":
        return None

    from .discriminator_shap import evaluate_discriminator_snapshots

    excluded = list(shap_config.get("exclude_features", []))
    if shap_config.get("exclude_target", False):
        excluded.append(str(config["target_col"]))
    excluded = list(dict.fromkeys(excluded))
    evaluation = evaluate_discriminator_snapshots(
        model,
        real_probe,
        synthetic_probe,
        background_size=int(shap_config.get("background_size", 50)),
        explain_size=int(shap_config.get("explain_size", 100)),
        test_size=float(config.get("detector", {}).get("test_size", 0.3)),
        condition_samples=int(shap_config.get("condition_samples", 8)),
        seed=int(shap_config.get("seed", seed)),
        exclude_features=excluded,
    )
    trajectory = evaluation.trajectory
    csv_name = f"discriminator_shap_{variant}.csv"
    atomic_write_csv(output_dir / csv_name, trajectory)
    metrics_name = f"discriminator_snapshot_metrics_{variant}.csv"
    predictions_name = f"discriminator_snapshot_predictions_{variant}.csv"
    atomic_write_csv(output_dir / metrics_name, evaluation.metrics)
    atomic_write_csv(output_dir / predictions_name, evaluation.predictions)
    metadata = {
        "variant": variant,
        "backend": generator_name,
        "method": "shap.GradientExplainer",
        "model": "ctabgan_plus_internal_discriminator",
        "epochs": sorted(int(epoch) for epoch in evaluation.metrics["epoch"].unique()),
        "background_rows": (
            int(trajectory["background_rows"].iloc[0]) if not trajectory.empty else 0
        ),
        "maximum_explained_rows_per_group": int(
            shap_config.get("explain_size", 100)
        ),
        "seed": int(shap_config.get("seed", seed)),
        "excluded_features": excluded,
        "probe": "balanced_real_audit_and_variant_synthetic_audit",
        "temporal_interpretation": (
            "historical_discriminator_snapshots_evaluated_against_a_fixed_"
            "final_generator_probe"
        ),
        "probe_split": "calibration_and_stratified_holdout",
        "threshold": "calibration_only_youden_j",
        "primary_scope": "correct_synthetic_holdout_only",
        "outcome_groups": [
            "correct_real",
            "false_real_as_synthetic",
            "correct_synthetic",
            "false_synthetic_as_real",
        ],
        "conditional_vector": "marginalized_valid_sampled_conditions",
        "condition_samples": int(shap_config.get("condition_samples", 8)),
        "aggregation": "signed_encoded_sum_then_mean_absolute_by_original_feature",
        "artifact": csv_name,
        "metrics_artifact": metrics_name,
        "predictions_artifact": predictions_name,
    }
    atomic_write_json(output_dir / f"discriminator_shap_{variant}.json", metadata)
    return metadata


def _save_discriminator_detector_comparison(
    output_dir: Path,
    variant: str,
    detector_importance: pd.Series,
    detector_signed: pd.Series,
    top_k: int,
) -> None:
    """Compare like-scoped post-hoc and final-snapshot feature importance."""
    trajectory_path = output_dir / f"discriminator_shap_{variant}.csv"
    if not trajectory_path.is_file():
        return
    trajectory = pd.read_csv(trajectory_path)
    if trajectory.empty or "primary_scope" not in trajectory:
        return
    primary = trajectory[
        trajectory["primary_scope"].astype(str).str.lower().isin({"true", "1"})
    ].copy()
    if primary.empty:
        return
    final_epoch = int(pd.to_numeric(primary["epoch"]).max())
    primary = primary[pd.to_numeric(primary["epoch"]) == final_epoch].copy()
    snapshot = primary.set_index("feature")
    features = list(
        dict.fromkeys(
            [*detector_importance.index.astype(str), *snapshot.index.astype(str)]
        )
    )
    comparison = pd.DataFrame({"feature": features})
    comparison["detector_importance_share"] = comparison["feature"].map(
        detector_importance.astype(float)
    ).fillna(0.0)
    comparison["detector_mean_signed_shap"] = comparison["feature"].map(
        detector_signed.astype(float)
    )
    comparison["snapshot_importance_share"] = comparison["feature"].map(
        pd.to_numeric(snapshot["importance_share"], errors="coerce")
    ).fillna(0.0)
    comparison["snapshot_mean_signed_shap"] = comparison["feature"].map(
        pd.to_numeric(snapshot["mean_signed_shap"], errors="coerce")
    )
    signed_valid = (
        comparison["detector_mean_signed_shap"].notna()
        & comparison["snapshot_mean_signed_shap"].notna()
        & ~np.isclose(comparison["detector_mean_signed_shap"], 0.0)
        & ~np.isclose(comparison["snapshot_mean_signed_shap"], 0.0)
    )
    comparison["signed_direction_agreement"] = np.where(
        signed_valid,
        np.sign(comparison["detector_mean_signed_shap"])
        == np.sign(comparison["snapshot_mean_signed_shap"]),
        np.nan,
    )
    comparison["detector_rank"] = comparison[
        "detector_importance_share"
    ].rank(method="min", ascending=False).astype(int)
    comparison["snapshot_rank"] = comparison[
        "snapshot_importance_share"
    ].rank(method="min", ascending=False).astype(int)
    comparison["rank_difference_snapshot_minus_detector"] = (
        comparison["snapshot_rank"] - comparison["detector_rank"]
    )
    comparison = comparison.sort_values(
        ["detector_rank", "snapshot_rank", "feature"], kind="mergesort"
    ).reset_index(drop=True)
    name = f"discriminator_detector_shap_comparison_{variant}.csv"
    atomic_write_csv(output_dir / name, comparison)

    count = min(int(top_k), len(comparison))
    detector_top = set(comparison.nsmallest(count, "detector_rank")["feature"])
    snapshot_top = set(comparison.nsmallest(count, "snapshot_rank")["feature"])
    union = detector_top | snapshot_top
    spearman = comparison["detector_importance_share"].corr(
        comparison["snapshot_importance_share"], method="spearman"
    )
    atomic_write_json(
        output_dir / f"discriminator_detector_shap_comparison_{variant}.json",
        {
            "variant": variant,
            "snapshot_epoch": final_epoch,
            "shared_scope": "correct_synthetic_holdout_only",
            "detector_method": "TreeSHAP",
            "snapshot_method": "GradientExplainer",
            "spearman_importance_correlation": (
                float(spearman) if np.isfinite(spearman) else None
            ),
            "top_k": count,
            "top_k_overlap_count": len(detector_top & snapshot_top),
            "top_k_jaccard": float(len(detector_top & snapshot_top) / len(union))
            if union
            else 0.0,
            "signed_direction_agreement_rate": (
                float(
                    pd.to_numeric(
                        comparison.loc[
                            signed_valid, "signed_direction_agreement"
                        ],
                        errors="coerce",
                    ).mean()
                )
                if signed_valid.any()
                else None
            ),
            "artifact": name,
        },
    )


def run_experiment(
    config: Dict[str, Any],
    project_root: Path,
    stage: str,
    device_spec: str,
    variants: Iterable[str],
    output_override: Path | None = None,
    resume: bool = False,
    smoke: bool = False,
    adapter_factory=None,
    progress: str = "auto",
) -> Path:
    variants = [v.upper() for v in variants]
    if progress not in {"auto", "on", "off"}:
        raise ValueError("progress must be auto, on, or off")
    invalid = sorted(set(variants) - set(VALID_VARIANTS))
    if invalid:
        raise ValueError(f"Unknown variants: {invalid}")
    if stage not in {"val", "test"}:
        raise ValueError("stage must be val or test")
    if stage == "test" and smoke:
        raise ValueError("--smoke cannot be used with --stage test")
    if stage == "test" and not config.get("frozen", False):
        raise ValueError("Test evaluation requires config field frozen=true")

    config = json.loads(json.dumps(config))
    config["stage"] = stage
    config["device"] = device_spec
    config["variants"] = variants
    config["smoke"] = bool(smoke)
    seed = int(config.get("seed", 42))
    generator_name = str(config.get("generator_name", "ctabgan_plus")).lower()
    if (
        generator_name == "ctabgan_plus"
        and config.get("discriminator_shap", {}).get("enabled", False)
    ):
        snapshot_shap = config["discriminator_shap"]
        detector_shap = config.get("detector", {})
        snapshot_frequency = config["generator"].get("snapshot_frq")
        epochs = int(config["generator"]["epochs"])
        if (
            isinstance(snapshot_frequency, bool)
            or not isinstance(snapshot_frequency, int)
            or snapshot_frequency <= 0
            or snapshot_frequency > epochs
        ):
            raise ValueError(
                "Enabled discriminator_shap requires generator.snapshot_frq to be "
                "a positive integer no greater than generator.epochs"
            )
        if int(snapshot_shap.get("explain_size", 100)) != int(
            detector_shap.get("shap_max_rows", 100)
        ):
            raise ValueError(
                "Comparable detector and discriminator SHAP require equal "
                "detector.shap_max_rows and discriminator_shap.explain_size"
            )
        if bool(snapshot_shap.get("exclude_target", False)):
            raise ValueError(
                "Comparable detector and discriminator SHAP must use the same "
                "feature set; discriminator_shap.exclude_target must be false"
            )
        if int(snapshot_shap.get("seed", seed)) != seed:
            raise ValueError(
                "Comparable detector and discriminator SHAP must use the run seed"
            )
    if smoke:
        config["generator"]["epochs"] = int(config.get("smoke_epochs", 1))
        config["generator"]["batch_size"] = int(config.get("smoke_batch_size", 64))
        if (
            str(config.get("generator_name", "ctabgan_plus")).lower() == "ctabgan_plus"
            and config.get("discriminator_shap", {}).get("enabled", False)
        ):
            configured_frequency = config["generator"].get("snapshot_frq")
            config["generator"]["snapshot_frq"] = min(
                int(configured_frequency or config["generator"]["epochs"]),
                int(config["generator"]["epochs"]),
            )
        config.setdefault("detector", {})["n_estimators"] = 20
        config["detector"]["shap_max_rows"] = 100
        config.setdefault("baseline_diagnostics", {})["n_estimators"] = 20
        config.setdefault("priority_diagnostics", {})["n_estimators"] = 20
        config.setdefault("feature_exclusion_sensitivity", {})["n_estimators"] = 20
        config.setdefault("evaluation", {})["n_estimators"] = 20
        mixed_smoke = config.setdefault("mixed_utility", {})
        mixed_smoke["repeats"] = 1
        mixed_smoke["n_estimators"] = 20
        mixed_smoke["additive_fractions"] = [0.0, 1.0]
        mixed_smoke["replacement_fractions"] = [0.0, 1.0]

    data_path = (project_root / config["data_path"]).resolve()
    data_hash, code_hash = file_sha256(data_path), _code_hash(project_root)
    fingerprint = _fingerprint(config, data_hash, code_hash)
    dataset_name = str(config.get("dataset_name", data_path.stem)).strip().lower()
    dataset_name = re.sub(r"[^a-z0-9_-]+", "_", dataset_name).strip("_") or "dataset"
    generator_tag = "ctabgan" if generator_name == "ctabgan_plus" else generator_name
    run_name = f"{dataset_name}_{generator_tag}_{stage}_seed{seed}_{fingerprint[:10]}"
    output_dir = output_override or (project_root / config.get("results_dir", "results") / run_name)
    output_dir = output_dir.resolve()

    manifest_path = output_dir / "manifest.json"
    if output_dir.exists() and not resume:
        raise FileExistsError(f"Output directory exists; pass --resume to continue: {output_dir}")
    if resume:
        if not manifest_path.exists():
            raise FileNotFoundError("Cannot resume without manifest.json")
        previous_manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        if previous_manifest.get("fingerprint") != fingerprint:
            raise ValueError("Resume fingerprint mismatch: config, data, or code changed")
    output_dir.mkdir(parents=True, exist_ok=True)

    if adapter_factory is None:
        from .device import device_manifest, resolve_device, seed_everything

        device = resolve_device(device_spec)
        seed_everything(seed, config.get("deterministic", True), config.get("allow_tf32", False))
        runtime_manifest = device_manifest(
            device, config.get("deterministic", True), config.get("allow_tf32", False)
        )
    else:
        device = device_spec
        runtime_manifest = {
            "device": str(device_spec),
            "precision": "test-double",
            "deterministic": True,
            "allow_tf32": False,
        }
    started = time.time()
    manifest = {
        "status": "running",
        "started_at": datetime.now(timezone.utc).isoformat(),
        "fingerprint": fingerprint,
        "data_sha256": data_hash,
        "code_sha256": code_hash,
        "smoke": smoke,
        "progress": progress,
        "generator_name": generator_name,
        "dp_cgan_transformer_reuse": bool(
            generator_name == "dp_cgan"
            and config["generator"].get(
                "reuse_transformer_across_variants", False
            )
        ),
        **runtime_manifest,
    }
    atomic_write_json(manifest_path, manifest)
    atomic_write_json(output_dir / "config.json", config)

    data = pd.read_csv(data_path)
    if smoke:
        sample_size = min(int(config.get("smoke_rows", 500)), len(data))
        data = data.groupby(config["target_col"], group_keys=False).sample(
            frac=sample_size / len(data), random_state=seed
        )
        if len(data) > sample_size:
            data = data.sample(sample_size, random_state=seed)
        data = data.reset_index(drop=True)
    continuous = config.get(
        "continuous_cols", [c for c in data.columns if c not in config["categorical_cols"]]
    )
    split_cfg = config.get("split", {})
    splits = create_data_splits(data, config["target_col"], seed=seed, **split_cfg)
    atomic_write_json(output_dir / "split_indices.json", splits.indices)
    real_eval = splits.val if stage == "val" else splits.test
    reuse_dp_transformer = bool(
        adapter_factory is None
        and generator_name == "dp_cgan"
        and config["generator"].get("reuse_transformer_across_variants", False)
    )
    frozen_dp_transformer = (
        output_dir / "backend_work" / "shared" / "fitted_transformer_A0.pkl"
    )
    dp_transformer_manifest = output_dir / "dp_cgan_transformer_manifest.json"
    dp_transformer_context = _dp_transformer_context(
        splits.train,
        splits.indices["train"],
        config["generator"]["categorical_columns"],
        data_hash,
    )
    if adapter_factory is None:
        def make_adapter(variant):
            variant_config = json.loads(json.dumps(config))
            if reuse_dp_transformer and variant != "A0":
                if not frozen_dp_transformer.is_file():
                    raise FileNotFoundError(
                        "The frozen A0 DP-CGAN transformer is unavailable; A0 must "
                        "complete before a weighted variant can start"
                    )
                variant_config["generator"]["saved_transformer"] = str(
                    frozen_dp_transformer.resolve()
                )
            return _adapter(
                variant_config,
                device,
                seed,
                progress,
                f"{generator_name} {variant} GAN training",
                output_dir / "backend_work" / variant,
            )
    else:
        def make_adapter(variant):
            return adapter_factory()

    baseline_audit_path = output_dir / "baseline_synthetic_audit.csv"
    baseline_eval_path = output_dir / "synthetic_A0.csv"
    baseline_model = None
    training_diagnostics_by_variant: Dict[str, Dict[str, Any]] = {}
    if resume and baseline_audit_path.exists() and baseline_eval_path.exists():
        if (
            generator_name == "ctabgan_plus"
            and config.get("discriminator_shap", {}).get("enabled", False)
            and not (output_dir / "discriminator_shap_A0.csv").is_file()
        ):
            raise ValueError(
                "Cannot resume A0 discriminator SHAP because the in-memory snapshots "
                "were not persisted; start a new run to produce this artifact"
            )
        baseline_audit = pd.read_csv(baseline_audit_path)
        baseline_eval = pd.read_csv(baseline_eval_path)
        training_path = output_dir / "training_diagnostics_A0.json"
        if training_path.exists():
            training_diagnostics_by_variant["A0"] = json.loads(
                training_path.read_text(encoding="utf-8")
            )
    else:
        baseline_model = make_adapter("A0")
        training_diagnostics_by_variant["A0"] = _fit_and_save_training(
            baseline_model,
            splits.train,
            "A0",
            output_dir,
            generator_name=generator_name,
            generator_config=config["generator"],
        )
        baseline_audit = baseline_model.sample(len(splits.audit))
        raw_audit = getattr(baseline_model, "last_raw_sample", None)
        if raw_audit is not None:
            atomic_write_csv(output_dir / "baseline_synthetic_audit_raw.csv", raw_audit)
        baseline_eval = baseline_model.sample(len(splits.train))
        raw_eval = getattr(baseline_model, "last_raw_sample", None)
        if raw_eval is not None:
            atomic_write_csv(output_dir / "synthetic_raw_A0.csv", raw_eval)
        _save_postprocessing_diagnostics(baseline_model, output_dir, "A0")
        atomic_write_csv(baseline_audit_path, baseline_audit)
        atomic_write_csv(baseline_eval_path, baseline_eval)
        _save_discriminator_shap_artifacts(
            baseline_model,
            splits.audit,
            baseline_audit,
            output_dir,
            "A0",
            config,
            generator_name,
            seed,
        )

    if reuse_dp_transformer:
        _freeze_or_validate_dp_transformer(
            output_dir / "backend_work" / "A0" / "fitted_transformer.pkl",
            frozen_dp_transformer,
            dp_transformer_manifest,
            dp_transformer_context,
        )

    detector_cfg = config.get("detector", {})
    audit_result = train_detector(
        splits.audit,
        baseline_audit,
        config["categorical_cols"],
        seed=seed,
        n_estimators=int(detector_cfg.get("n_estimators", 300)),
        test_size=float(detector_cfg.get("test_size", 0.3)),
        shap_max_rows=int(detector_cfg.get("shap_max_rows", 100)),
        shap_scope=str(
            detector_cfg.get(
                "shap_scope", "correct_synthetic_holdout_only"
            )
        ),
        n_jobs=int(config.get("n_jobs", -1)),
    )
    components = compute_feature_components(
        splits.audit,
        baseline_audit,
        audit_result.shap_importance,
        config["categorical_cols"],
        continuous,
        target_col=config["target_col"],
    )
    atomic_write_csv(output_dir / "baseline_feature_components.csv", components)
    atomic_write_json(output_dir / "baseline_detector_metrics.json", audit_result.metrics)
    detector_shap = pd.DataFrame(
        {
            "feature": audit_result.shap_importance.index.astype(str),
            "importance_share": audit_result.shap_importance.to_numpy(dtype=float),
            "mean_signed_shap": audit_result.shap_signed.reindex(
                audit_result.shap_importance.index
            ).to_numpy(dtype=float),
        }
    ).sort_values(
        ["importance_share", "feature"],
        ascending=[False, True],
        kind="mergesort",
    ).reset_index(drop=True)
    detector_shap["rank"] = np.arange(1, len(detector_shap) + 1, dtype=int)
    atomic_write_csv(output_dir / "baseline_detector_shap.csv", detector_shap)
    _save_discriminator_detector_comparison(
        output_dir,
        "A0",
        audit_result.shap_importance,
        audit_result.shap_signed,
        int(config.get("weighting", {}).get("top_k", 5)),
    )
    diagnostics_cfg = config.get("baseline_diagnostics", {})
    if diagnostics_cfg.get("enabled", True):
        ranking, conditional_features, conditional_categories, conditional_detectors = (
            baseline_detector_diagnostics(
                splits.audit,
                baseline_audit,
                components,
                config["target_col"],
                config["categorical_cols"],
                continuous,
                top_n=int(diagnostics_cfg.get("top_n", 10)),
                seed=seed,
                n_estimators=int(
                    diagnostics_cfg.get(
                        "n_estimators", detector_cfg.get("n_estimators", 300)
                    )
                ),
                n_jobs=int(config.get("n_jobs", -1)),
                detector_fn=train_detector,
            )
        )
        atomic_write_csv(output_dir / "baseline_detector_feature_ranking.csv", ranking)
        atomic_write_csv(
            output_dir / "baseline_conditional_feature_diagnostics.csv", conditional_features
        )
        atomic_write_csv(
            output_dir / "baseline_conditional_category_frequencies.csv", conditional_categories
        )
        atomic_write_csv(
            output_dir / "baseline_conditional_detector_metrics.csv", conditional_detectors
        )
    definitions = build_region_definitions(
        splits.train, splits.audit, baseline_audit, config["categorical_cols"], continuous
    )
    atomic_write_json(
        output_dir / "underrepresented_regions.json",
        {feature: definition.to_dict() for feature, definition in definitions.items()},
    )

    weighting = {"alpha": 4.0, "gamma": 0.6, "top_k": 5, "w_max": 3.0}
    weighting.update(config.get("weighting", {}))
    group_mapping = correlation_groups(
        splits.audit,
        continuous,
        float(weighting.get("correlation_threshold", 0.65)),
    )
    atomic_write_json(output_dir / "priority_correlation_groups.json", group_mapping)
    priorities = {
        variant: compute_feature_priority(
            components,
            variant,
            int(weighting["top_k"]),
            feature_groups=group_mapping if weighting.get("correlation_aware_selection", False) else None,
            max_per_group=(
                int(weighting.get("max_per_correlation_group", 1))
                if weighting.get("correlation_aware_selection", False)
                else None
            ),
            exclude_features=weighting.get("exclude_features", []),
        )
        for variant in ("A2", "A3", "A4", "A5")
    }
    for variant, priority in priorities.items():
        atomic_write_csv(output_dir / f"feature_scores_{variant}.csv", priority)

    priority_diagnostic_cfg = config.get("priority_diagnostics", {})
    if priority_diagnostic_cfg.get("enabled", True):
        priority_summary, priority_spikes, priority_detectors, priority_correlations = (
            prioritized_feature_diagnostics(
                splits.train,
                splits.audit,
                baseline_audit,
                priorities,
                config["target_col"],
                config["categorical_cols"],
                continuous,
                baseline_detector_auc=float(audit_result.metrics["detector_auc"]),
                seed=seed,
                n_estimators=int(priority_diagnostic_cfg.get("n_estimators", 100)),
                n_jobs=int(config.get("n_jobs", -1)),
            )
        )
        atomic_write_csv(output_dir / "prioritized_feature_diagnostics.csv", priority_summary)
        atomic_write_csv(output_dir / "prioritized_feature_value_spikes.csv", priority_spikes)
        atomic_write_csv(output_dir / "prioritized_feature_detector_ablation.csv", priority_detectors)
        atomic_write_csv(output_dir / "prioritized_feature_correlations.csv", priority_correlations)

    evaluation_cfg = config.get("evaluation", {})
    utility_tasks = config.get("utility_tasks") or [
        {
            "name": "mortality",
            "balance": "imbalanced",
            "task_type": "classification",
            "target_col": config["target_col"],
            "positive_label": "1",
        }
    ]
    atomic_write_json(
        output_dir / "utility_tasks.json",
        {
            "decision_rule": "random_forest_argmax",
            "threshold_tuning": False,
            "tasks": [
                {
                    **task,
                    "train_class_frequencies": _cat(splits.train[task["target_col"]]).value_counts(normalize=True).to_dict(),
                    "evaluation_class_frequencies": _cat(real_eval[task["target_col"]]).value_counts(normalize=True).to_dict(),
                }
                for task in utility_tasks
            ],
        },
    )
    exclusion_cfg = config.get("feature_exclusion_sensitivity", {})
    if exclusion_cfg.get("enabled", True):
        detector_sensitivity, utility_sensitivity = feature_exclusion_sensitivity(
            splits.audit,
            baseline_audit,
            real_eval,
            baseline_eval,
            priorities,
            group_mapping,
            config["categorical_cols"],
            utility_tasks,
            seed=seed,
            n_estimators=int(exclusion_cfg.get("n_estimators", 100)),
            n_jobs=int(config.get("n_jobs", -1)),
        )
        atomic_write_csv(
            output_dir / "feature_family_detector_sensitivity.csv", detector_sensitivity
        )
        atomic_write_csv(
            output_dir / "feature_family_utility_sensitivity.csv", utility_sensitivity
        )
    mixed_cfg = config.get("mixed_utility", {})
    mixed_enabled = bool(mixed_cfg.get("enabled", False))
    mixed_results: List[pd.DataFrame] = []
    real_only_path = output_dir / "utility_real_only_baseline.csv"
    if resume and real_only_path.exists():
        real_only_utility = pd.read_csv(real_only_path)
    else:
        task_frames = []
        baseline_repeats = int(
            mixed_cfg.get("repeats", 3)
            if mixed_enabled
            else evaluation_cfg.get("real_baseline_repeats", 1)
        )
        baseline_estimators = int(
            mixed_cfg.get("n_estimators", evaluation_cfg.get("n_estimators", 300))
            if mixed_enabled
            else evaluation_cfg.get("n_estimators", 300)
        )
        for task in utility_tasks:
            task_frame = evaluate_real_only_baseline(
                splits.train,
                real_eval,
                task["target_col"],
                config["categorical_cols"],
                positive_label=str(task["positive_label"]),
                balance=str(task.get("balance", "imbalanced")),
                repeats=baseline_repeats,
                seed=seed,
                n_estimators=baseline_estimators,
                n_jobs=int(config.get("n_jobs", -1)),
                progress=progress,
            )
            task_frame.insert(0, "utility_task", task["name"])
            task_frame.insert(1, "target_balance", task.get("balance", "unspecified"))
            task_frames.append(task_frame)
        real_only_utility = pd.concat(task_frames, ignore_index=True)
        atomic_write_csv(real_only_path, real_only_utility)

    def mixed_utility_for_variant(variant: str, synthetic: pd.DataFrame) -> pd.DataFrame:
        path = output_dir / f"utility_mixture_{variant}.csv"
        if resume and path.exists():
            return pd.read_csv(path)
        task_results = []
        for task in utility_tasks:
            task_baseline = real_only_utility[
                real_only_utility["utility_task"] == task["name"]
            ].drop(columns=["utility_task", "target_balance"])
            result = evaluate_mixed_utility_curve(
                variant, splits.train, synthetic, real_eval, task["target_col"],
                config["categorical_cols"], task_baseline,
                positive_label=str(task["positive_label"]),
                balance=str(task.get("balance", "imbalanced")),
                additive_fractions=mixed_cfg.get("additive_fractions", [0.0, 0.25, 0.5, 1.0]),
                replacement_fractions=mixed_cfg.get("replacement_fractions", [0.0, 0.25, 0.5, 0.75, 1.0]),
                repeats=int(mixed_cfg.get("repeats", 3)), seed=seed,
                n_estimators=int(mixed_cfg.get("n_estimators", evaluation_cfg.get("n_estimators", 300))),
                n_jobs=int(config.get("n_jobs", -1)), progress=progress,
            )
            result.insert(1, "utility_task", task["name"])
            result.insert(2, "target_balance", task.get("balance", "unspecified"))
            task_results.append(result)
        result = pd.concat(task_results, ignore_index=True)
        result.insert(1, "generator_name", generator_name)
        atomic_write_csv(path, result)
        return result

    rows: List[Dict[str, Any]] = []
    metrics_by_variant: Dict[str, Dict[str, Any]] = {}
    feature_details_by_variant: Dict[str, pd.DataFrame] = {}
    for variant in variants:
        complete_marker = output_dir / f".{variant}.complete"
        metrics_path = output_dir / f"metrics_{variant}.json"
        if resume and complete_marker.exists() and metrics_path.exists():
            if (
                generator_name == "ctabgan_plus"
                and config.get("discriminator_shap", {}).get("enabled", False)
                and not (output_dir / f"discriminator_shap_{variant}.csv").is_file()
            ):
                raise ValueError(
                    f"Cannot resume {variant} discriminator SHAP because the in-memory "
                    "snapshots were not persisted; start a new run to produce this artifact"
                )
            metrics = json.loads(metrics_path.read_text(encoding="utf-8"))
            metrics_by_variant[variant] = metrics
            rows.append({"variant": variant, **metrics})
            feature_path = output_dir / f"feature_metrics_{variant}.csv"
            if feature_path.exists():
                feature_details_by_variant[variant] = pd.read_csv(feature_path)
            if mixed_enabled:
                synthetic_path = output_dir / f"synthetic_{variant}.csv"
                mixed_results.append(
                    mixed_utility_for_variant(variant, pd.read_csv(synthetic_path))
                )
            continue

        if variant == "A0":
            synthetic = baseline_eval
            selection_counts = {}
            diagnostics = None
        else:
            retrain, selection_counts, diagnostics = _prepare_variant_data(
                variant, splits.train, priorities, definitions, weighting, seed, output_dir
            )
            model = make_adapter(variant)
            training_diagnostics_by_variant[variant] = _fit_and_save_training(
                model,
                retrain,
                variant,
                output_dir,
                generator_name=generator_name,
                generator_config=config["generator"],
            )
            synthetic = model.sample(len(splits.train))
            raw_synthetic = getattr(model, "last_raw_sample", None)
            if raw_synthetic is not None:
                atomic_write_csv(output_dir / f"synthetic_raw_{variant}.csv", raw_synthetic)
            _save_postprocessing_diagnostics(model, output_dir, variant)
            atomic_write_csv(output_dir / f"synthetic_{variant}.csv", synthetic)
            _save_discriminator_shap_artifacts(
                model,
                splits.audit,
                synthetic.iloc[: len(splits.audit)].reset_index(drop=True),
                output_dir,
                variant,
                config,
                generator_name,
                seed,
            )
        atomic_write_json(output_dir / f"augmentation_counts_{variant}.json", selection_counts)

        metrics, details = evaluate_variant(
            splits.train,
            real_eval,
            synthetic,
            config["target_col"],
            config["categorical_cols"],
            continuous,
            utility_tasks=utility_tasks,
            seed=seed,
            n_jobs=int(config.get("n_jobs", -1)),
            n_estimators=int(evaluation_cfg.get("n_estimators", 300)),
            privacy_max_reference_rows=evaluation_cfg.get("privacy_max_reference_rows"),
            privacy_max_query_rows=evaluation_cfg.get("privacy_max_query_rows"),
        )
        metrics["variant"] = variant
        metrics["generator_name"] = generator_name
        metrics["training_rows"] = len(splits.train) if variant == "A0" else len(retrain)
        metrics["synthetic_rows"] = len(synthetic)
        if diagnostics:
            metrics["weight_mean"] = diagnostics["mean"]
            metrics["weight_fraction_capped"] = diagnostics["fraction_capped"]
        training_audit = training_diagnostics_by_variant.get(variant, {})
        metrics["training_stability_status"] = training_audit.get("status", "unavailable")
        metrics["training_warning_unstable_tail"] = training_audit.get(
            "warning_unstable_tail"
        )
        metrics["mixture_all_converged"] = training_audit.get("mixture_all_converged")
        atomic_write_json(metrics_path, metrics)
        atomic_write_csv(output_dir / f"feature_metrics_{variant}.csv", details)
        feature_details_by_variant[variant] = details
        atomic_write_json(complete_marker, {"status": "complete"})
        if mixed_enabled:
            mixed_results.append(mixed_utility_for_variant(variant, synthetic))
        metrics_by_variant[variant] = metrics
        rows.append(metrics)

    summary, delta_rows = _build_controlled_deltas(
        pd.DataFrame(rows), metrics_by_variant, real_only_utility, utility_tasks
    )
    atomic_write_csv(output_dir / "ablation_summary.csv", summary)
    atomic_write_csv(output_dir / "ablation_deltas.csv", delta_rows)
    mixed_all = (
        pd.concat(mixed_results, ignore_index=True)
        if mixed_enabled and mixed_results
        else None
    )
    save_ablation_heatmap_artifacts(
        summary,
        real_only_utility,
        utility_tasks,
        output_dir,
        mixture_results=mixed_all,
    )
    ranking_path = output_dir / "baseline_detector_feature_ranking.csv"
    if ranking_path.exists() and feature_details_by_variant:
        atomic_write_csv(
            output_dir / "top_shap_feature_variant_metrics.csv",
            top_shap_feature_variant_metrics(
                pd.read_csv(ranking_path), feature_details_by_variant
            ),
        )
    if mixed_all is not None:
        atomic_write_csv(output_dir / "utility_mixture_results.csv", mixed_all)
        atomic_write_csv(
            output_dir / "utility_mixture_summary.csv",
            summarize_mixed_utility(mixed_all),
        )
        atomic_write_json(
            output_dir / "utility_mixture_manifest.json",
            {
                "evaluation_split": stage,
                "generator_name": generator_name,
                "utility_tasks": utility_tasks,
                "utility_protocol": {
                    "decision_rule": "random_forest_argmax",
                    "threshold_tuning": False,
                    "training_prevalence": "real_train for mortality; 0.5 for mortality_balanced",
                    "evaluation_prevalence": "unchanged real validation/test prevalence",
                    "raw_prevalence_diagnostics": True,
                },
                "variants": variants,
                "additive_fractions": mixed_cfg.get(
                    "additive_fractions", [0.0, 0.25, 0.5, 1.0]
                ),
                "replacement_fractions": mixed_cfg.get(
                    "replacement_fractions", [0.0, 0.25, 0.5, 0.75, 1.0]
                ),
                "repeats": int(mixed_cfg.get("repeats", 3)),
                "n_estimators": int(
                    mixed_cfg.get(
                        "n_estimators", evaluation_cfg.get("n_estimators", 300)
                    )
                ),
                "fraction_semantics": {
                    "additive": "synthetic rows divided by len(real_train)",
                    "replacement": "synthetic rows divided by fixed total training rows",
                },
            },
        )
    manifest.update(
        {
            "status": "complete",
            "completed_at": datetime.now(timezone.utc).isoformat(),
            "elapsed_seconds": time.time() - started,
            "variants_completed": variants,
            "dp_cgan_transformer_manifest": (
                "dp_cgan_transformer_manifest.json"
                if dp_transformer_manifest.is_file()
                else None
            ),
            "discriminator_shap_variants": [
                variant
                for variant in variants
                if (output_dir / f"discriminator_shap_{variant}.csv").is_file()
            ],
        }
    )
    atomic_write_json(manifest_path, manifest)
    return output_dir


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True, type=Path)
    parser.add_argument("--stage", choices=("val", "test"), default="val")
    parser.add_argument("--device", default="auto")
    parser.add_argument("--seed", type=int)
    parser.add_argument("--output-dir", type=Path)
    parser.add_argument("--variants", default=",".join(VALID_VARIANTS))
    parser.add_argument("--smoke", action="store_true")
    parser.add_argument("--resume", action="store_true")
    parser.add_argument(
        "--progress",
        choices=("auto", "on", "off"),
        default="auto",
        help="GAN training progress: interactive bar, forced bar, or disabled",
    )
    return parser


def main(argv=None) -> int:
    args = build_parser().parse_args(argv)
    project_root = Path(__file__).resolve().parents[1]
    config = _load_config(args.config.resolve())
    if args.seed is not None:
        config["seed"] = args.seed
    output = run_experiment(
        config,
        project_root,
        args.stage,
        args.device,
        [x.strip() for x in args.variants.split(",") if x.strip()],
        output_override=args.output_dir,
        resume=args.resume,
        smoke=args.smoke,
        progress=args.progress,
    )
    print(output)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
