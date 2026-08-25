"""Generate A0 detector diagnostics for an existing ablation run."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd

from .diagnostics import baseline_detector_diagnostics
from .io_utils import atomic_write_csv, file_sha256
from .priority_diagnostics import (
    feature_exclusion_sensitivity,
    prioritized_feature_diagnostics,
    top_shap_feature_variant_metrics,
)
from .scoring import correlation_groups
from .utility_balance import replace_legacy_gender_task


def run_existing_diagnostics(
    config_path: Path,
    run_dir: Path,
    *,
    top_n: int | None = None,
    n_estimators: int | None = None,
    n_jobs: int | None = None,
) -> Path:
    project_root = Path(__file__).resolve().parents[1]
    config = json.loads(config_path.resolve().read_text(encoding="utf-8"))
    run_dir = run_dir.resolve()
    manifest = json.loads((run_dir / "manifest.json").read_text(encoding="utf-8"))
    data_path = (project_root / config["data_path"]).resolve()
    if file_sha256(data_path) != manifest.get("data_sha256"):
        raise ValueError("Source data hash does not match the selected run")

    indices = json.loads((run_dir / "split_indices.json").read_text(encoding="utf-8"))
    data = pd.read_csv(data_path)
    real_audit = data.iloc[indices["audit"]].copy(deep=True).reset_index(drop=True)
    synthetic_audit = pd.read_csv(run_dir / "baseline_synthetic_audit.csv")
    components = pd.read_csv(run_dir / "baseline_feature_components.csv")
    diagnostic_config = config.get("baseline_diagnostics", {})
    detector_config = config.get("detector", {})
    continuous = config.get(
        "continuous_cols",
        [column for column in data.columns if column not in config["categorical_cols"]],
    )

    ranking, feature_gaps, category_gaps, detector_metrics = baseline_detector_diagnostics(
        real_audit,
        synthetic_audit,
        components,
        config["target_col"],
        config["categorical_cols"],
        continuous,
        top_n=int(top_n if top_n is not None else diagnostic_config.get("top_n", 10)),
        seed=int(config.get("seed", 42)),
        n_estimators=int(
            n_estimators
            if n_estimators is not None
            else diagnostic_config.get("n_estimators", detector_config.get("n_estimators", 300))
        ),
        n_jobs=int(n_jobs if n_jobs is not None else config.get("n_jobs", -1)),
    )
    atomic_write_csv(run_dir / "baseline_detector_feature_ranking.csv", ranking)
    atomic_write_csv(run_dir / "baseline_conditional_feature_diagnostics.csv", feature_gaps)
    atomic_write_csv(run_dir / "baseline_conditional_category_frequencies.csv", category_gaps)
    atomic_write_csv(run_dir / "baseline_conditional_detector_metrics.csv", detector_metrics)

    priorities = {}
    for variant in ("A2", "A3", "A4", "A5"):
        path = run_dir / f"feature_scores_{variant}.csv"
        if path.exists():
            priorities[variant] = pd.read_csv(path)
    if priorities:
        priority_config = config.get("priority_diagnostics", {})
        baseline_metrics_path = run_dir / "baseline_detector_metrics.json"
        baseline_auc = 0.5
        if baseline_metrics_path.exists():
            baseline_auc = float(
                json.loads(baseline_metrics_path.read_text(encoding="utf-8")).get(
                    "detector_auc", baseline_auc
                )
            )
        summary, spikes, ablation, correlations = prioritized_feature_diagnostics(
            data.iloc[indices["train"]].reset_index(drop=True),
            real_audit,
            synthetic_audit,
            priorities,
            config["target_col"],
            config["categorical_cols"],
            continuous,
            baseline_detector_auc=baseline_auc,
            seed=int(config.get("seed", 42)),
            n_estimators=int(priority_config.get("n_estimators", 100)),
            n_jobs=int(config.get("n_jobs", -1)),
        )
        atomic_write_csv(run_dir / "prioritized_feature_diagnostics.csv", summary)
        atomic_write_csv(run_dir / "prioritized_feature_value_spikes.csv", spikes)
        atomic_write_csv(run_dir / "prioritized_feature_detector_ablation.csv", ablation)
        atomic_write_csv(run_dir / "prioritized_feature_correlations.csv", correlations)
        stage = str(manifest.get("stage", "val"))
        eval_name = stage if stage in {"val", "test"} else "val"
        real_eval = data.iloc[indices[eval_name]].copy(deep=True).reset_index(drop=True)
        synthetic_eval = pd.read_csv(run_dir / "synthetic_A0.csv")
        weighting = config.get("weighting", {})
        groups = correlation_groups(
            real_audit, continuous, float(weighting.get("correlation_threshold", 0.65))
        )
        utility_tasks = replace_legacy_gender_task(config.get("utility_tasks") or [
            {"name": "mortality", "balance": "imbalanced", "target_col": config["target_col"], "positive_label": "1"},
            {"name": "mortality_balanced", "balance": "balanced", "target_col": config["target_col"], "positive_label": "1"},
        ], config["target_col"])
        sensitivity_config = config.get("feature_exclusion_sensitivity", {})
        detector_sensitivity, utility_sensitivity = feature_exclusion_sensitivity(
            real_audit, synthetic_audit, real_eval, synthetic_eval, priorities, groups,
            config["categorical_cols"], utility_tasks,
            seed=int(config.get("seed", 42)),
            n_estimators=int(sensitivity_config.get("n_estimators", 100)),
            n_jobs=int(config.get("n_jobs", -1)),
        )
        atomic_write_csv(run_dir / "feature_family_detector_sensitivity.csv", detector_sensitivity)
        atomic_write_csv(run_dir / "feature_family_utility_sensitivity.csv", utility_sensitivity)
    feature_metrics_by_variant = {
        path.stem.removeprefix("feature_metrics_"): pd.read_csv(path)
        for path in sorted(run_dir.glob("feature_metrics_A*.csv"))
    }
    if feature_metrics_by_variant:
        atomic_write_csv(
            run_dir / "top_shap_feature_variant_metrics.csv",
            top_shap_feature_variant_metrics(ranking, feature_metrics_by_variant),
        )
    return run_dir


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True, type=Path)
    parser.add_argument("--run-dir", required=True, type=Path)
    parser.add_argument("--top-n", type=int)
    parser.add_argument("--n-estimators", type=int)
    parser.add_argument("--n-jobs", type=int)
    return parser


def main(argv=None) -> int:
    args = build_parser().parse_args(argv)
    output = run_existing_diagnostics(
        args.config,
        args.run_dir,
        top_n=args.top_n,
        n_estimators=args.n_estimators,
        n_jobs=args.n_jobs,
    )
    metrics = pd.read_csv(output / "baseline_conditional_detector_metrics.csv")
    print(metrics.to_string(index=False))
    print(output)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
