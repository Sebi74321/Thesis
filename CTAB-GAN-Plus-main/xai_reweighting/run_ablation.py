"""CLI runner for the A0/A1/A2/A4/A5 weighted-retraining experiment."""

from __future__ import annotations

import argparse
import hashlib
import json
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Iterable, List

import pandas as pd

from .augmentation import create_uniform_augmentation, create_weighted_augmentation
from .data_split import create_data_splits
from .detector import train_detector
from .evaluation import evaluate_variant
from .io_utils import atomic_write_csv, atomic_write_json, combined_sha256, file_sha256
from .scoring import (
    build_region_definitions,
    compute_feature_components,
    compute_feature_priority,
    compute_row_weights,
    weight_diagnostics,
)

VALID_VARIANTS = ("A0", "A1", "A2", "A4", "A5")
PREVIOUS = {"A1": "A0", "A2": "A1", "A4": "A2", "A5": "A4"}


def _load_config(path: Path) -> Dict[str, Any]:
    with path.open(encoding="utf-8") as handle:
        config = json.load(handle)
    required = {"data_path", "target_col", "categorical_cols", "generator"}
    missing = required - set(config)
    if missing:
        raise ValueError(f"Config is missing required fields: {sorted(missing)}")
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


def _adapter(config, device, seed):
    from .generator_adapters import CTABGANPlusAdapter

    generator = dict(config["generator"])
    return CTABGANPlusAdapter(
        **generator,
        device=device,
        seed=seed,
        deterministic=config.get("deterministic", True),
        allow_tf32=config.get("allow_tf32", False),
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
    diagnostics["warning_fraction_capped"] = diagnostics["fraction_capped"] > 0.10
    diagnostics["warning_mean_outside_recommended_range"] = not (1.10 <= diagnostics["mean"] <= 1.30)
    diagnostics["warning_median_far_from_one"] = diagnostics["50%"] > 1.25
    diagnostics["warning_nearly_uniform"] = diagnostics["std"] < 1e-6
    atomic_write_json(output_dir / f"row_weight_summary_{variant}.json", diagnostics)
    atomic_write_csv(
        output_dir / f"row_weights_{variant}.csv",
        pd.concat([weights, contributions], axis=1),
    )
    augmented, counts = create_weighted_augmentation(train, weights, gamma, seed)
    return augmented, counts, diagnostics


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
) -> Path:
    variants = [v.upper() for v in variants]
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
    if smoke:
        config["generator"]["epochs"] = int(config.get("smoke_epochs", 1))
        config["generator"]["batch_size"] = int(config.get("smoke_batch_size", 64))
        config.setdefault("detector", {})["n_estimators"] = 20
        config["detector"]["shap_max_rows"] = 100
        config.setdefault("evaluation", {})["n_estimators"] = 20

    data_path = (project_root / config["data_path"]).resolve()
    data_hash, code_hash = file_sha256(data_path), _code_hash(project_root)
    fingerprint = _fingerprint(config, data_hash, code_hash)
    run_name = f"mimic_ctabgan_{stage}_seed{seed}_{fingerprint[:10]}"
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
    make_adapter = adapter_factory or (lambda: _adapter(config, device, seed))

    baseline_audit_path = output_dir / "baseline_synthetic_audit.csv"
    baseline_eval_path = output_dir / "synthetic_A0.csv"
    baseline_model = None
    if resume and baseline_audit_path.exists() and baseline_eval_path.exists():
        baseline_audit = pd.read_csv(baseline_audit_path)
        baseline_eval = pd.read_csv(baseline_eval_path)
    else:
        baseline_model = make_adapter()
        baseline_model.fit(splits.train)
        baseline_audit = baseline_model.sample(len(splits.audit))
        baseline_eval = baseline_model.sample(len(splits.train))
        atomic_write_csv(baseline_audit_path, baseline_audit)
        atomic_write_csv(baseline_eval_path, baseline_eval)

    detector_cfg = config.get("detector", {})
    audit_result = train_detector(
        splits.audit,
        baseline_audit,
        config["categorical_cols"],
        seed=seed,
        n_estimators=int(detector_cfg.get("n_estimators", 300)),
        shap_max_rows=int(detector_cfg.get("shap_max_rows", 2000)),
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
    definitions = build_region_definitions(
        splits.train, splits.audit, baseline_audit, config["categorical_cols"], continuous
    )
    atomic_write_json(
        output_dir / "underrepresented_regions.json",
        {feature: definition.to_dict() for feature, definition in definitions.items()},
    )

    weighting = {"alpha": 1.0, "gamma": 0.25, "top_k": 5, "w_max": 2.0}
    weighting.update(config.get("weighting", {}))
    priorities = {
        variant: compute_feature_priority(components, variant, int(weighting["top_k"]))
        for variant in ("A2", "A4", "A5")
    }
    for variant, priority in priorities.items():
        atomic_write_csv(output_dir / f"feature_scores_{variant}.csv", priority)

    rows: List[Dict[str, Any]] = []
    metrics_by_variant: Dict[str, Dict[str, Any]] = {}
    for variant in variants:
        complete_marker = output_dir / f".{variant}.complete"
        metrics_path = output_dir / f"metrics_{variant}.json"
        if resume and complete_marker.exists() and metrics_path.exists():
            metrics = json.loads(metrics_path.read_text(encoding="utf-8"))
            metrics_by_variant[variant] = metrics
            rows.append({"variant": variant, **metrics})
            continue

        if variant == "A0":
            synthetic = baseline_eval
            selection_counts = {}
            diagnostics = None
        else:
            retrain, selection_counts, diagnostics = _prepare_variant_data(
                variant, splits.train, priorities, definitions, weighting, seed, output_dir
            )
            model = make_adapter()
            model.fit(retrain)
            synthetic = model.sample(len(splits.train))
            atomic_write_csv(output_dir / f"synthetic_{variant}.csv", synthetic)
        atomic_write_json(output_dir / f"augmentation_counts_{variant}.json", selection_counts)

        metrics, details = evaluate_variant(
            splits.train,
            real_eval,
            synthetic,
            config["target_col"],
            config["categorical_cols"],
            continuous,
            seed,
            int(config.get("n_jobs", -1)),
            int(config.get("evaluation", {}).get("n_estimators", 300)),
        )
        metrics["variant"] = variant
        metrics["training_rows"] = len(splits.train) if variant == "A0" else len(retrain)
        metrics["synthetic_rows"] = len(synthetic)
        if diagnostics:
            metrics["weight_mean"] = diagnostics["mean"]
            metrics["weight_fraction_capped"] = diagnostics["fraction_capped"]
        atomic_write_json(metrics_path, metrics)
        atomic_write_csv(output_dir / f"feature_metrics_{variant}.csv", details)
        atomic_write_json(complete_marker, {"status": "complete"})
        metrics_by_variant[variant] = metrics
        rows.append(metrics)

    summary = pd.DataFrame(rows)
    delta_rows = []
    for variant, previous in PREVIOUS.items():
        if variant not in metrics_by_variant or previous not in metrics_by_variant:
            continue
        current, control = metrics_by_variant[variant], metrics_by_variant[previous]
        delta = {"comparison": f"{variant}-{previous}", "variant": variant, "control": previous}
        for key, value in current.items():
            if isinstance(value, (int, float)) and isinstance(control.get(key), (int, float)):
                delta[f"delta_{key}"] = value - control[key]
        delta_rows.append(delta)
        row_mask = summary["variant"] == variant
        summary.loc[row_mask, "comparison"] = f"{variant}-{previous}"
        for key, value in delta.items():
            if key.startswith("delta_"):
                summary.loc[row_mask, f"vs_previous_{key}"] = value
    atomic_write_csv(output_dir / "ablation_summary.csv", summary)
    atomic_write_csv(output_dir / "ablation_deltas.csv", pd.DataFrame(delta_rows))
    manifest.update(
        {
            "status": "complete",
            "completed_at": datetime.now(timezone.utc).isoformat(),
            "elapsed_seconds": time.time() - started,
            "variants_completed": variants,
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
        args.output_dir,
        args.resume,
        args.smoke,
    )
    print(output)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
