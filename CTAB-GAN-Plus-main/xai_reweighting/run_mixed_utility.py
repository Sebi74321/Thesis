"""Run mixed real/synthetic utility evaluation on an existing ablation run."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd

from .io_utils import atomic_write_csv, atomic_write_json, file_sha256
from .mixed_utility import (
    evaluate_mixed_utility_curve,
    evaluate_real_only_baseline,
    summarize_mixed_utility,
)
from .utility_balance import replace_legacy_gender_task


def run_existing_mixed_utility(
    run_dir: Path,
    variants: list[str],
    *,
    repeats: int | None = None,
    n_estimators: int | None = None,
    n_jobs: int | None = None,
    progress: str = "auto",
) -> Path:
    project_root = Path(__file__).resolve().parents[1]
    run_dir = run_dir.resolve()
    config = json.loads((run_dir / "config.json").read_text(encoding="utf-8"))
    if config.get("smoke", False):
        raise ValueError(
            "Existing-run mixed utility is unavailable for smoke runs because their "
            "sampled source-row mapping is not persisted; run a new smoke experiment instead"
        )
    manifest = json.loads((run_dir / "manifest.json").read_text(encoding="utf-8"))
    indices = json.loads((run_dir / "split_indices.json").read_text(encoding="utf-8"))
    data_path = (project_root / config["data_path"]).resolve()
    if file_sha256(data_path) != manifest.get("data_sha256"):
        raise ValueError("Source data hash does not match the selected run")

    data = pd.read_csv(data_path)
    take = lambda name: data.iloc[indices[name]].copy(deep=True).reset_index(drop=True)
    real_train = take("train")
    stage = config.get("stage", "val")
    if stage not in {"val", "test"}:
        raise ValueError(f"Unknown run stage: {stage}")
    real_eval = take(stage)
    mixed_cfg = config.get("mixed_utility", {})
    evaluation_cfg = config.get("evaluation", {})
    repeat_count = int(repeats if repeats is not None else mixed_cfg.get("repeats", 3))
    tree_count = int(
        n_estimators
        if n_estimators is not None
        else mixed_cfg.get("n_estimators", evaluation_cfg.get("n_estimators", 300))
    )
    jobs = int(n_jobs if n_jobs is not None else config.get("n_jobs", -1))
    seed = int(config.get("seed", 42))
    generator_name = str(config.get("generator_name", "ctabgan_plus"))
    utility_tasks = replace_legacy_gender_task(config.get("utility_tasks") or [
        {"name": "mortality", "balance": "imbalanced", "target_col": config["target_col"], "positive_label": "1"},
        {"name": "mortality_balanced", "balance": "balanced", "target_col": config["target_col"], "positive_label": "1"},
    ], config["target_col"])

    baseline_frames = []
    for task in utility_tasks:
        baseline = evaluate_real_only_baseline(
            real_train, real_eval, task["target_col"], config["categorical_cols"],
            positive_label=str(task["positive_label"]), repeats=repeat_count,
            balance=str(task.get("balance", "imbalanced")),
            seed=seed, n_estimators=tree_count, n_jobs=jobs, progress=progress,
        )
        baseline.insert(0, "utility_task", task["name"])
        baseline.insert(1, "target_balance", task.get("balance", "unspecified"))
        baseline_frames.append(baseline)
    real_only = pd.concat(baseline_frames, ignore_index=True)
    atomic_write_csv(run_dir / "utility_real_only_baseline.csv", real_only)

    frames = []
    for variant in variants:
        synthetic_path = run_dir / f"synthetic_{variant}.csv"
        if not synthetic_path.exists():
            raise FileNotFoundError(f"Missing synthetic data for {variant}: {synthetic_path}")
        task_results = []
        synthetic = pd.read_csv(synthetic_path)
        for task in utility_tasks:
            baseline = real_only[real_only["utility_task"] == task["name"]].drop(
                columns=["utility_task", "target_balance"]
            )
            task_result = evaluate_mixed_utility_curve(
                variant, real_train, synthetic, real_eval, task["target_col"],
                config["categorical_cols"], baseline,
                positive_label=str(task["positive_label"]),
                balance=str(task.get("balance", "imbalanced")),
                additive_fractions=mixed_cfg.get("additive_fractions", [0.0, 0.25, 0.5, 1.0]),
                replacement_fractions=mixed_cfg.get("replacement_fractions", [0.0, 0.25, 0.5, 0.75, 1.0]),
                repeats=repeat_count, seed=seed, n_estimators=tree_count,
                n_jobs=jobs, progress=progress,
            )
            task_result.insert(1, "utility_task", task["name"])
            task_result.insert(2, "target_balance", task.get("balance", "unspecified"))
            task_results.append(task_result)
        result = pd.concat(task_results, ignore_index=True)
        result.insert(1, "generator_name", generator_name)
        atomic_write_csv(run_dir / f"utility_mixture_{variant}.csv", result)
        frames.append(result)

    combined = pd.concat(frames, ignore_index=True)
    atomic_write_csv(run_dir / "utility_mixture_results.csv", combined)
    atomic_write_csv(
        run_dir / "utility_mixture_summary.csv", summarize_mixed_utility(combined)
    )
    atomic_write_json(
        run_dir / "utility_mixture_manifest.json",
        {
            "stage": stage,
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
            "repeats": repeat_count,
            "n_estimators": tree_count,
            "n_jobs": jobs,
            "fraction_semantics": {
                "additive": "synthetic rows divided by len(real_train)",
                "replacement": "synthetic rows divided by fixed total training rows",
            },
            "additive_fractions": mixed_cfg.get(
                "additive_fractions", [0.0, 0.25, 0.5, 1.0]
            ),
            "replacement_fractions": mixed_cfg.get(
                "replacement_fractions", [0.0, 0.25, 0.5, 0.75, 1.0]
            ),
        },
    )
    return run_dir


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", required=True, type=Path)
    parser.add_argument("--variants", default="A0,A1,A2,A3,A4,A5")
    parser.add_argument("--repeats", type=int)
    parser.add_argument("--n-estimators", type=int)
    parser.add_argument("--n-jobs", type=int)
    parser.add_argument("--progress", choices=("auto", "on", "off"), default="auto")
    return parser


def main(argv=None) -> int:
    args = build_parser().parse_args(argv)
    variants = [value.strip().upper() for value in args.variants.split(",") if value.strip()]
    output = run_existing_mixed_utility(
        args.run_dir,
        variants,
        repeats=args.repeats,
        n_estimators=args.n_estimators,
        n_jobs=args.n_jobs,
        progress=args.progress,
    )
    print(output / "utility_mixture_summary.csv")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
