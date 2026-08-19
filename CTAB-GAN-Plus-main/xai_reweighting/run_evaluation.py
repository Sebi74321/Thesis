"""Recompute every evaluation artifact for a completed ablation run without GAN training."""

from __future__ import annotations

import argparse
import json
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable

import pandas as pd

from .evaluation import evaluate_variant
from .io_utils import atomic_write_csv, atomic_write_json, combined_sha256, file_sha256
from .mixed_utility import evaluate_real_only_baseline
from .run_ablation import VALID_VARIANTS, _build_controlled_deltas
from .run_diagnostics import run_existing_diagnostics
from .run_mixed_utility import run_existing_mixed_utility


_PRESERVED_METRIC_FIELDS = {
    "training_rows",
    "weight_mean",
    "weight_fraction_capped",
    "training_stability_status",
    "training_warning_unstable_tail",
    "mixture_all_converged",
}


def _read_json(path: Path) -> dict[str, Any]:
    if not path.is_file():
        raise FileNotFoundError(f"Required completed-run artifact is missing: {path}")
    return json.loads(path.read_text(encoding="utf-8"))


def _take_rows(data: pd.DataFrame, indices: dict[str, Any], name: str) -> pd.DataFrame:
    if name not in indices:
        raise ValueError(f"Persisted split indices do not contain {name!r}")
    selected = [int(value) for value in indices[name]]
    if any(value < 0 or value >= len(data) for value in selected):
        raise ValueError(f"Persisted {name!r} split contains out-of-range row indices")
    if len(selected) != len(set(selected)):
        raise ValueError(f"Persisted {name!r} split contains duplicate row indices")
    return data.iloc[selected].copy(deep=True).reset_index(drop=True)


def _resolve_variants(
    requested: Iterable[str] | None,
    config: dict[str, Any],
    manifest: dict[str, Any],
    run_dir: Path,
) -> list[str]:
    configured = (
        config.get("variants") or manifest.get("variants_completed") or VALID_VARIANTS
    )
    variants = [str(value).upper() for value in (requested or configured)]
    variants = list(dict.fromkeys(variants))
    invalid = sorted(set(variants) - set(VALID_VARIANTS))
    if invalid:
        raise ValueError(f"Unknown ablation variants: {invalid}")
    if "A0" not in variants and (run_dir / "synthetic_A0.csv").is_file():
        variants.insert(0, "A0")
    if not variants:
        raise ValueError("At least one completed variant is required")
    missing = [
        variant
        for variant in variants
        if not (run_dir / f"synthetic_{variant}.csv").is_file()
    ]
    if missing:
        raise FileNotFoundError(f"Saved synthetic datasets are missing for: {missing}")
    return variants


def _real_only_utility(
    run_dir: Path,
    config: dict[str, Any],
    real_train: pd.DataFrame,
    real_eval: pd.DataFrame,
    utility_tasks: list[dict[str, Any]],
    *,
    progress: str,
) -> pd.DataFrame:
    evaluation_cfg = config.get("evaluation", {})
    frames = []
    for task in utility_tasks:
        frame = evaluate_real_only_baseline(
            real_train,
            real_eval,
            task["target_col"],
            config["categorical_cols"],
            positive_label=str(task["positive_label"]),
            repeats=int(evaluation_cfg.get("real_baseline_repeats", 1)),
            seed=int(config.get("seed", 42)),
            n_estimators=int(evaluation_cfg.get("n_estimators", 300)),
            n_jobs=int(config.get("n_jobs", -1)),
            progress=progress,
        )
        frame.insert(0, "utility_task", task["name"])
        frame.insert(1, "target_balance", task.get("balance", "unspecified"))
        frames.append(frame)
    result = pd.concat(frames, ignore_index=True)
    atomic_write_csv(run_dir / "utility_real_only_baseline.csv", result)
    return result


def run_existing_evaluation(
    run_dir: Path,
    variants: Iterable[str] | None = None,
    *,
    progress: str = "auto",
    run_diagnostics: bool = True,
    run_mixed_utility: bool = True,
) -> Path:
    """Reevaluate saved synthetic data while leaving every generator artifact untouched."""
    if progress not in {"auto", "on", "off"}:
        raise ValueError("progress must be auto, on, or off")
    report = (
        (lambda message: print(message, flush=True))
        if progress != "off"
        else (lambda message: None)
    )
    started = time.time()
    started_at = datetime.now(timezone.utc).isoformat()
    project_root = Path(__file__).resolve().parents[1]
    run_dir = run_dir.resolve()
    config_path = run_dir / "config.json"
    config = _read_json(config_path)
    manifest = _read_json(run_dir / "manifest.json")
    if manifest.get("status") != "complete":
        raise ValueError("Full reevaluation requires a run whose manifest status is complete")
    if bool(config.get("smoke", manifest.get("smoke", False))):
        raise ValueError(
            "Completed smoke runs cannot be reevaluated because their source-row mapping "
            "was not persisted; run a new smoke experiment instead"
        )

    data_path = (project_root / config["data_path"]).resolve()
    if not data_path.is_file():
        raise FileNotFoundError(f"Configured source dataset does not exist: {data_path}")
    data_hash = file_sha256(data_path)
    if data_hash != manifest.get("data_sha256"):
        raise ValueError("Source data hash does not match the selected completed run")

    data = pd.read_csv(data_path)
    indices = _read_json(run_dir / "split_indices.json")
    stage = str(config.get("stage", "val"))
    if stage not in {"val", "test"}:
        raise ValueError(f"Unknown persisted evaluation stage: {stage}")
    real_train = _take_rows(data, indices, "train")
    real_eval = _take_rows(data, indices, stage)
    selected_variants = _resolve_variants(variants, config, manifest, run_dir)
    continuous = config.get(
        "continuous_cols",
        [column for column in data.columns if column not in config["categorical_cols"]],
    )
    utility_tasks = config.get("utility_tasks") or [
        {
            "name": "mortality",
            "balance": "imbalanced",
            "target_col": config["target_col"],
            "positive_label": "1",
        }
    ]

    reevaluation_manifest_path = run_dir / "evaluation_rerun_manifest.json"
    code_files = list((project_root / "xai_reweighting").glob("*.py"))
    rerun_manifest: dict[str, Any] = {
        "status": "running",
        "started_at": started_at,
        "training_performed": False,
        "stage": stage,
        "seed": int(config.get("seed", 42)),
        "variants": selected_variants,
        "data_sha256": data_hash,
        "evaluation_code_sha256": combined_sha256(code_files),
        "diagnostics_requested": bool(run_diagnostics),
        "mixed_utility_requested": bool(run_mixed_utility),
    }
    atomic_write_json(reevaluation_manifest_path, rerun_manifest)

    try:
        synthetic_by_variant: dict[str, pd.DataFrame] = {}
        for variant in selected_variants:
            synthetic_path = run_dir / f"synthetic_{variant}.csv"
            synthetic = pd.read_csv(synthetic_path)
            if list(synthetic.columns) != list(real_train.columns):
                raise ValueError(
                    f"{synthetic_path.name} does not preserve the real training schema"
                )
            if len(synthetic) != len(real_train):
                raise ValueError(
                    f"{synthetic_path.name} has {len(synthetic)} rows; expected {len(real_train)}"
                )
            synthetic_by_variant[variant] = synthetic
        report(f"Validated saved data for {len(selected_variants)} variants")

        mixed_enabled = bool(config.get("mixed_utility", {}).get("enabled", False))
        if run_mixed_utility and mixed_enabled:
            report("Recomputing real-only and mixed utility")
            run_existing_mixed_utility(
                run_dir,
                selected_variants,
                progress=progress,
            )
            real_only = pd.read_csv(run_dir / "utility_real_only_baseline.csv")
        else:
            real_only = _real_only_utility(
                run_dir,
                config,
                real_train,
                real_eval,
                utility_tasks,
                progress=progress,
            )

        metrics_by_variant: dict[str, dict[str, Any]] = {}
        rows = []
        evaluation_cfg = config.get("evaluation", {})
        for position, variant in enumerate(selected_variants, start=1):
            report(f"Evaluating {variant} ({position}/{len(selected_variants)})")
            synthetic = synthetic_by_variant[variant]
            metrics, details = evaluate_variant(
                real_train,
                real_eval,
                synthetic,
                config["target_col"],
                config["categorical_cols"],
                continuous,
                utility_tasks=utility_tasks,
                seed=int(config.get("seed", 42)),
                n_jobs=int(config.get("n_jobs", -1)),
                n_estimators=int(evaluation_cfg.get("n_estimators", 300)),
                privacy_max_reference_rows=evaluation_cfg.get("privacy_max_reference_rows"),
                privacy_max_query_rows=evaluation_cfg.get("privacy_max_query_rows"),
            )
            previous_path = run_dir / f"metrics_{variant}.json"
            previous = _read_json(previous_path) if previous_path.is_file() else {}
            for field in _PRESERVED_METRIC_FIELDS:
                if field in previous:
                    metrics[field] = previous[field]
            metrics["variant"] = variant
            metrics["training_rows"] = int(
                metrics.get("training_rows", len(real_train))
            )
            metrics["synthetic_rows"] = len(synthetic)
            atomic_write_json(previous_path, metrics)
            atomic_write_csv(run_dir / f"feature_metrics_{variant}.csv", details)
            metrics_by_variant[variant] = metrics
            rows.append(metrics)

        summary, deltas = _build_controlled_deltas(
            pd.DataFrame(rows), metrics_by_variant, real_only, utility_tasks
        )
        atomic_write_csv(run_dir / "ablation_summary.csv", summary)
        atomic_write_csv(run_dir / "ablation_deltas.csv", deltas)

        if run_diagnostics:
            report("Recomputing detector, priority, and feature-exclusion diagnostics")
            run_existing_diagnostics(config_path, run_dir)

        rerun_manifest.update(
            {
                "status": "complete",
                "completed_at": datetime.now(timezone.utc).isoformat(),
                "elapsed_seconds": time.time() - started,
                "artifacts_recomputed": [
                    "metrics_<variant>.json",
                    "feature_metrics_<variant>.csv",
                    "ablation_summary.csv",
                    "ablation_deltas.csv",
                    "utility_real_only_baseline.csv",
                    *(
                        [
                            "utility_mixture_<variant>.csv",
                            "utility_mixture_results.csv",
                            "utility_mixture_summary.csv",
                        ]
                        if run_mixed_utility and mixed_enabled
                        else []
                    ),
                    *(["diagnostic CSV artifacts"] if run_diagnostics else []),
                ],
            }
        )
        atomic_write_json(reevaluation_manifest_path, rerun_manifest)
    except Exception as exc:
        rerun_manifest.update(
            {
                "status": "failed",
                "failed_at": datetime.now(timezone.utc).isoformat(),
                "elapsed_seconds": time.time() - started,
                "error_type": type(exc).__name__,
                "error": str(exc),
            }
        )
        atomic_write_json(reevaluation_manifest_path, rerun_manifest)
        raise
    return run_dir


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", required=True, type=Path)
    parser.add_argument(
        "--variants",
        help="Comma-separated completed variants; defaults to all variants recorded by the run",
    )
    parser.add_argument("--progress", choices=("auto", "on", "off"), default="auto")
    parser.add_argument("--skip-diagnostics", action="store_true")
    parser.add_argument("--skip-mixed-utility", action="store_true")
    return parser


def main(argv=None) -> int:
    args = build_parser().parse_args(argv)
    variants = (
        [value.strip().upper() for value in args.variants.split(",") if value.strip()]
        if args.variants
        else None
    )
    output = run_existing_evaluation(
        args.run_dir,
        variants,
        progress=args.progress,
        run_diagnostics=not args.skip_diagnostics,
        run_mixed_utility=not args.skip_mixed_utility,
    )
    print(output)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
