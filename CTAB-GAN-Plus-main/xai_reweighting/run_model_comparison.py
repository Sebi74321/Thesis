"""CLI runner for the RQ1 CTAB-GAN+/CTGAN/DP-CGAN baseline comparison."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import tempfile
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable

import numpy as np
import pandas as pd

from .data_split import create_data_splits
from .evaluation import _cat, evaluate_variant
from .io_utils import atomic_write_csv, atomic_write_json, combined_sha256, file_sha256
from .mixed_utility import (
    evaluate_mixed_utility_curve,
    evaluate_real_only_baseline,
    summarize_mixed_utility,
)
from .rq1_evaluation import evaluate_rq1_specific, real_real_reference
from .rq1_reporting import aggregate_results, save_comparison_plots
from .run_ablation import _training_diagnostics


VALID_MODELS = ("ctabgan_plus", "ctgan", "dp_cgan")


def _load_config(path: Path) -> dict[str, Any]:
    config = json.loads(path.read_text(encoding="utf-8"))
    if "base_config" in config:
        base_path = (path.parent / config["base_config"]).resolve()
        base = json.loads(base_path.read_text(encoding="utf-8"))
        override = {key: value for key, value in config.items() if key != "base_config"}
        base.update(override)
        config = base
    for model_name, values in list(config.get("models", {}).items()):
        if values.get("inherit_generator", False):
            inherited = dict(config.get("generator", {}))
            inherited.update({key: value for key, value in values.items() if key != "inherit_generator"})
            config["models"][model_name] = inherited
    required = {"data_path", "target_col", "categorical_cols", "models"}
    missing = required - set(config)
    if missing:
        raise ValueError(f"Config is missing required fields: {sorted(missing)}")
    unknown = set(config["models"]) - set(VALID_MODELS)
    if unknown:
        raise ValueError(f"Config contains unknown models: {sorted(unknown)}")
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


def _fingerprint(config: dict, data_hash: str, code_hash: str) -> str:
    payload = json.dumps(
        {"config": config, "data_hash": data_hash, "code_hash": code_hash}, sort_keys=True
    ).encode()
    return hashlib.sha256(payload).hexdigest()


def _safe_dataset_name(value: str) -> str:
    return re.sub(r"[^a-z0-9_-]+", "_", value.lower()).strip("_") or "dataset"


def _stratified_smoke_sample(data: pd.DataFrame, target: str, n: int, seed: int) -> pd.DataFrame:
    n = min(n, len(data))
    sampled = data.groupby(target, group_keys=False).sample(
        frac=n / len(data), random_state=seed
    )
    if len(sampled) > n:
        sampled = sampled.sample(n=n, random_state=seed)
    elif len(sampled) < n:
        remainder = data.drop(index=sampled.index).sample(n=n - len(sampled), random_state=seed)
        sampled = pd.concat([sampled, remainder])
    return sampled.reset_index(drop=True)


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


def _model_config(config: dict, model_name: str, smoke: bool) -> dict:
    values = json.loads(json.dumps(config["models"][model_name]))
    values.setdefault("categorical_columns", list(config["categorical_cols"]))
    if smoke:
        values["epochs"] = int(values.pop("smoke_epochs", config.get("smoke_epochs", 1)))
        values["batch_size"] = int(
            values.pop("smoke_batch_size", config.get("smoke_batch_size", 50))
        )
    else:
        values.pop("smoke_epochs", None)
        values.pop("smoke_batch_size", None)
    return values


def _training_artifacts(
    model,
    output_dir: Path,
    model_config: dict,
    train_rows: int,
    model_name: str | None = None,
) -> dict:
    history = getattr(model, "training_history", pd.DataFrame())
    if not isinstance(history, pd.DataFrame):
        history = pd.DataFrame(history)
    atomic_write_csv(output_dir / "training_history.csv", history)
    diagnostics = _training_diagnostics(history)
    diagnostics["convergence_warnings"] = list(
        getattr(model, "convergence_warnings", [])
    )
    diagnostics["convergence_warning_count"] = len(diagnostics["convergence_warnings"])
    diagnostics["training_rows"] = train_rows
    batch_size = int(model_config.get("batch_size", train_rows))
    epochs = int(model_config.get("epochs", 0))
    diagnostics["steps_per_epoch"] = max(train_rows // batch_size, 1)
    diagnostics["epochs_configured"] = epochs
    diagnostics["generator_updates"] = diagnostics["steps_per_epoch"] * epochs
    diagnostics["discriminator_updates"] = (
        diagnostics["generator_updates"] * int(model_config.get("discriminator_steps", 1))
    )
    if model_name == "dp_cgan":
        diagnostics["differential_privacy_enabled"] = False
        diagnostics["backend_mode"] = "non_private_baseline"
    atomic_write_json(output_dir / "training_diagnostics.json", diagnostics)
    atomic_write_json(
        output_dir / "convergence_warnings.json", diagnostics["convergence_warnings"]
    )
    if getattr(model, "upstream_stdout", ""):
        (output_dir / "upstream_training.log").write_text(
            model.upstream_stdout, encoding="utf-8"
        )
    return diagnostics


def _make_adapter(
    model_name: str,
    model_config: dict,
    device,
    seed: int,
    config: dict,
    progress: str,
    run_dir: Path,
):
    from .generator_adapters import create_generator

    return create_generator(
        model_name,
        model_config,
        device=device,
        seed=seed,
        deterministic=config.get("deterministic", True),
        allow_tf32=config.get("allow_tf32", False),
        progress=progress,
        progress_label=f"{model_name} seed {seed}",
        work_dir=run_dir / "backend_work",
    )


def run_model_comparison(
    config: dict[str, Any],
    project_root: Path,
    stage: str,
    device_spec: str,
    models: Iterable[str],
    seeds: Iterable[int],
    output_override: Path | None = None,
    resume: bool = False,
    smoke: bool = False,
    progress: str = "auto",
    adapter_factory=None,
) -> Path:
    models = [model.strip().lower() for model in models]
    seeds = [int(seed) for seed in seeds]
    if not models or not seeds:
        raise ValueError("At least one model and seed are required")
    invalid = sorted(set(models) - set(VALID_MODELS))
    if invalid:
        raise ValueError(f"Unknown models: {invalid}")
    missing = sorted(set(models) - set(config.get("models", {})))
    if missing:
        raise ValueError(f"Selected models are missing configurations: {missing}")
    if (
        "dp_cgan" in models
        and config["models"]["dp_cgan"].get("private", False) is not False
    ):
        raise ValueError("dp_cgan is a non-private baseline; set private=false")
    if len(set(models)) != len(models) or len(set(seeds)) != len(seeds):
        raise ValueError("Models and seeds must not contain duplicates")
    if stage not in {"val", "test"}:
        raise ValueError("stage must be val or test")
    if progress not in {"auto", "on", "off"}:
        raise ValueError("progress must be auto, on, or off")
    if stage == "test" and smoke:
        raise ValueError("--smoke cannot be used with --stage test")
    if stage == "test" and not config.get("frozen", False):
        raise ValueError("Test evaluation requires config field frozen=true")

    config = json.loads(json.dumps(config))
    config.update(
        {"stage": stage, "device": device_spec, "selected_models": models, "seeds": seeds, "smoke": smoke}
    )
    split_seed = int(config.get("split_seed", 42))
    data_path = (project_root / config["data_path"]).resolve()
    data_hash = file_sha256(data_path)
    code_hash = _code_hash(project_root)
    fingerprint = _fingerprint(config, data_hash, code_hash)
    dataset = _safe_dataset_name(str(config.get("dataset_name", data_path.stem)))
    run_name = f"rq1_{dataset}_{stage}_{fingerprint[:10]}"
    output_dir = (output_override or project_root / config.get("results_dir", "results") / run_name).resolve()
    manifest_path = output_dir / "manifest.json"
    if output_dir.exists() and not resume:
        raise FileExistsError(f"Output directory exists; pass --resume to continue: {output_dir}")
    if resume:
        if not manifest_path.exists():
            raise FileNotFoundError("Cannot resume without manifest.json")
        previous = json.loads(manifest_path.read_text(encoding="utf-8"))
        if previous.get("fingerprint") != fingerprint:
            raise ValueError("Resume fingerprint mismatch: config, data, code, models, or seeds changed")
    output_dir.mkdir(parents=True, exist_ok=True)

    if adapter_factory is None:
        from .device import device_manifest, resolve_device, seed_everything

        device = resolve_device(device_spec)
        seed_everything(split_seed, config.get("deterministic", True), config.get("allow_tf32", False))
        environment = device_manifest(
            device, config.get("deterministic", True), config.get("allow_tf32", False)
        )
    else:
        device = device_spec
        environment = {"device": str(device), "precision": "test-double"}

    previous_started = None
    if resume and manifest_path.exists():
        previous_started = json.loads(manifest_path.read_text(encoding="utf-8")).get("started_at")
    started = time.time()
    manifest = {
        "status": "running",
        "started_at": previous_started or datetime.now(timezone.utc).isoformat(),
        "fingerprint": fingerprint,
        "data_sha256": data_hash,
        "code_sha256": code_hash,
        "split_seed": split_seed,
        "models": models,
        "seeds": seeds,
        "smoke": smoke,
        "non_thesis_result": bool(smoke),
        "progress": progress,
        **environment,
    }
    atomic_write_json(manifest_path, manifest)
    atomic_write_json(output_dir / "config.json", config)

    data = pd.read_csv(data_path)
    if smoke:
        data = _stratified_smoke_sample(
            data, config["target_col"], int(config.get("smoke_rows", 500)), split_seed
        )
    splits = create_data_splits(
        data, config["target_col"], seed=split_seed, **config.get("split", {})
    )
    split_path = output_dir / "split_indices.json"
    if resume and split_path.exists():
        prior_indices = json.loads(split_path.read_text(encoding="utf-8"))
        if prior_indices != splits.indices:
            raise ValueError("Persisted split indices do not match the reproducible split")
    atomic_write_json(split_path, splits.indices)
    real_eval = splits.val if stage == "val" else splits.test
    continuous = config.get(
        "continuous_cols", [column for column in data if column not in config["categorical_cols"]]
    )

    reference_metrics, reference_details = real_real_reference(
        splits.audit,
        real_eval,
        config["target_col"],
        config["categorical_cols"],
        continuous,
        split_seed,
    )
    atomic_write_json(output_dir / "real_real_reference.json", reference_metrics)
    atomic_write_csv(output_dir / "real_real_reference_details.csv", reference_details)

    evaluation_cfg = config.get("evaluation", {})
    utility_tasks = config.get("utility_tasks") or [{
        "name": "mortality", "balance": "imbalanced",
        "task_type": "classification", "target_col": config["target_col"],
        "positive_label": "1",
    }]
    atomic_write_json(
        output_dir / "utility_tasks.json",
        {
            "decision_rule": "random_forest_argmax",
            "threshold_tuning": False,
            "tasks": [{
                **task,
                "train_class_frequencies": _cat(splits.train[task["target_col"]]).value_counts(normalize=True).to_dict(),
                "evaluation_class_frequencies": _cat(real_eval[task["target_col"]]).value_counts(normalize=True).to_dict(),
            } for task in utility_tasks],
        },
    )
    mixed_cfg = config.get("mixed_utility", {})
    mixed_enabled = bool(mixed_cfg.get("enabled", True))
    if smoke:
        evaluation_cfg["n_estimators"] = int(config.get("smoke_n_estimators", 20))
        mixed_cfg.update(
            {
                "repeats": 1,
                "n_estimators": int(config.get("smoke_n_estimators", 20)),
                "additive_fractions": [0.0, 1.0],
                "replacement_fractions": [0.0, 1.0],
            }
        )

    real_only_path = output_dir / "utility_real_only_baseline.csv"
    real_only = pd.DataFrame()
    if mixed_enabled:
        if resume and real_only_path.exists():
            real_only = pd.read_csv(real_only_path)
        else:
            real_only_frames = []
            for task in utility_tasks:
                task_frame = evaluate_real_only_baseline(
                    splits.train, real_eval, task["target_col"], config["categorical_cols"],
                    positive_label=str(task["positive_label"]),
                    balance=str(task.get("balance", "imbalanced")),
                    repeats=int(mixed_cfg.get("repeats", 3)), seed=split_seed,
                    n_estimators=int(mixed_cfg.get("n_estimators", 300)),
                    n_jobs=int(config.get("n_jobs", -1)), progress=progress,
                )
                task_frame.insert(0, "utility_task", task["name"])
                task_frame.insert(1, "target_balance", task.get("balance", "unspecified"))
                real_only_frames.append(task_frame)
            real_only = pd.concat(real_only_frames, ignore_index=True)
            atomic_write_csv(real_only_path, real_only)

    rows: list[dict[str, Any]] = []
    mixed_frames = []
    for model_name in models:
        for seed in seeds:
            run_dir = output_dir / "models" / model_name / f"seed_{seed}"
            run_dir.mkdir(parents=True, exist_ok=True)
            complete = run_dir / ".complete.json"
            metrics_path = run_dir / "metrics.json"
            synthetic_path = run_dir / "synthetic.csv"
            mixed_path = run_dir / "utility_mixture.csv"
            if resume and complete.exists() and metrics_path.exists():
                metrics = json.loads(metrics_path.read_text(encoding="utf-8"))
                rows.append(metrics)
                if mixed_enabled and mixed_path.exists():
                    mixed_frames.append(pd.read_csv(mixed_path))
                continue

            run_started = time.time()
            model_config = _model_config(config, model_name, smoke)
            atomic_write_json(run_dir / "model_config.json", model_config)
            if resume and synthetic_path.exists():
                synthetic = pd.read_csv(synthetic_path)
                training_diagnostics = json.loads(
                    (run_dir / "training_diagnostics.json").read_text(encoding="utf-8")
                ) if (run_dir / "training_diagnostics.json").exists() else {}
            else:
                if adapter_factory is None:
                    model = _make_adapter(
                        model_name, model_config, device, seed, config, progress, run_dir
                    )
                else:
                    model = adapter_factory(model_name, model_config, device, seed, run_dir)
                try:
                    model.fit(splits.train.copy(deep=True))
                except Exception:
                    _training_artifacts(
                        model, run_dir, model_config, len(splits.train), model_name
                    )
                    raise
                training_diagnostics = _training_artifacts(
                    model, run_dir, model_config, len(splits.train), model_name
                )
                checkpoint_suffix = ".pkl" if model_name in {"ctgan", "dp_cgan"} else ".pt"
                _atomic_checkpoint(model, run_dir / f"model_checkpoint{checkpoint_suffix}")
                synthetic = model.sample(len(splits.train))
                if len(synthetic) != len(splits.train) or list(synthetic.columns) != list(splits.train.columns):
                    raise RuntimeError("Generator output did not preserve row count and schema")
                raw_synthetic = getattr(model, "last_raw_sample", None)
                if raw_synthetic is not None:
                    atomic_write_csv(run_dir / "synthetic_raw.csv", raw_synthetic)
                numeric_constraints = getattr(model, "numeric_constraints", None)
                if numeric_constraints:
                    atomic_write_json(run_dir / "numeric_postprocessing.json", numeric_constraints)
                postprocessing = getattr(model, "last_postprocessing_diagnostics", None)
                if postprocessing:
                    atomic_write_json(run_dir / "postprocessing_diagnostics.json", postprocessing)
                atomic_write_csv(synthetic_path, synthetic)

            metrics, feature_details = evaluate_variant(
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
            rq1_metrics, rq1_details = evaluate_rq1_specific(
                splits.audit,
                real_eval,
                synthetic,
                config["target_col"],
                config["categorical_cols"],
                continuous,
            )
            metrics.update(rq1_metrics)
            metrics.update(
                {
                    "dataset": str(config.get("dataset_name", dataset)),
                    "model": model_name,
                    "seed": seed,
                    "stage": stage,
                    "training_rows": len(splits.train),
                    "synthetic_rows": len(synthetic),
                    "training_seconds": time.time() - run_started,
                    "training_stability_status": training_diagnostics.get("status", "unavailable"),
                    "convergence_warning_count": training_diagnostics.get("convergence_warning_count", 0),
                }
            )
            atomic_write_json(metrics_path, metrics)
            atomic_write_csv(run_dir / "feature_metrics.csv", feature_details)
            atomic_write_csv(run_dir / "rq1_region_metrics.csv", rq1_details)

            if mixed_enabled:
                if resume and mixed_path.exists():
                    mixed = pd.read_csv(mixed_path)
                else:
                    task_mixed = []
                    for task in utility_tasks:
                        baseline = real_only[real_only["utility_task"] == task["name"]].drop(
                            columns=["utility_task", "target_balance"]
                        )
                        task_result = evaluate_mixed_utility_curve(
                            model_name, splits.train, synthetic, real_eval,
                            task["target_col"], config["categorical_cols"], baseline,
                            positive_label=str(task["positive_label"]),
                            balance=str(task.get("balance", "imbalanced")),
                            additive_fractions=mixed_cfg.get("additive_fractions", [0.0, 0.25, 0.5, 1.0]),
                            replacement_fractions=mixed_cfg.get("replacement_fractions", [0.0, 0.25, 0.5, 0.75, 1.0]),
                            repeats=int(mixed_cfg.get("repeats", 3)), seed=seed,
                            n_estimators=int(mixed_cfg.get("n_estimators", 300)),
                            n_jobs=int(config.get("n_jobs", -1)), progress=progress,
                        )
                        task_result.insert(1, "utility_task", task["name"])
                        task_result.insert(2, "target_balance", task.get("balance", "unspecified"))
                        task_mixed.append(task_result)
                    mixed = pd.concat(task_mixed, ignore_index=True)
                    mixed.insert(1, "generator_seed", seed)
                    atomic_write_csv(mixed_path, mixed)
                mixed_frames.append(mixed)
            atomic_write_json(complete, {"status": "complete", "completed_at": datetime.now(timezone.utc).isoformat()})
            rows.append(metrics)

            atomic_write_csv(output_dir / "rq1_results.csv", pd.DataFrame(rows))
            partial_summary = aggregate_results(pd.DataFrame(rows), reference_metrics)
            atomic_write_csv(output_dir / "rq1_summary.csv", partial_summary)

    results = pd.DataFrame(rows)
    summary = aggregate_results(results, reference_metrics)
    atomic_write_csv(output_dir / "rq1_results.csv", results)
    atomic_write_csv(output_dir / "rq1_summary.csv", summary)
    if mixed_frames:
        mixed = pd.concat(mixed_frames, ignore_index=True)
        atomic_write_csv(output_dir / "utility_mixture_results.csv", mixed)
        atomic_write_csv(output_dir / "utility_mixture_summary.csv", summarize_mixed_utility(mixed))
    plot_paths = save_comparison_plots(summary, output_dir / "plots")
    manifest.update(
        {
            "status": "complete",
            "completed_at": datetime.now(timezone.utc).isoformat(),
            "elapsed_seconds_this_invocation": time.time() - started,
            "completed_model_seed_runs": len(results),
            "plots": [str(path.relative_to(output_dir)) for path in plot_paths],
        }
    )
    atomic_write_json(manifest_path, manifest)
    return output_dir


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True, type=Path)
    parser.add_argument("--stage", choices=("val", "test"), default="val")
    parser.add_argument("--device", default="auto")
    parser.add_argument("--models", default=",".join(VALID_MODELS))
    parser.add_argument("--seeds", default="42,43,44")
    parser.add_argument("--output-dir", type=Path)
    parser.add_argument("--smoke", action="store_true")
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--progress", choices=("auto", "on", "off"), default="auto")
    return parser


def main(argv=None) -> int:
    args = build_parser().parse_args(argv)
    project_root = Path(__file__).resolve().parents[1]
    config = _load_config(args.config.resolve())
    output = run_model_comparison(
        config,
        project_root,
        args.stage,
        args.device,
        [value.strip() for value in args.models.split(",") if value.strip()],
        [int(value.strip()) for value in args.seeds.split(",") if value.strip()],
        output_override=args.output_dir,
        resume=args.resume,
        smoke=args.smoke,
        progress=args.progress,
    )
    print(output)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
