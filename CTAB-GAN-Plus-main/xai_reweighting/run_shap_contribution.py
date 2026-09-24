"""Focused SHAP controls: four GAN fits per seed, with the full ablation evaluation."""

import argparse
import copy
import json
import time
from pathlib import Path

import numpy as np

from .io_utils import atomic_write_json, file_sha256
from .run_ablation import _code_hash, _load_config, run_experiment
from .shap_contribution_reporting import STUDY_VARIANTS, summarize_contribution, validate_child


def run_study(config, project_root, output_dir, *, seeds=(42, 43, 44), split_seed=42,
              device="auto", stage="val", smoke=False, resume=False, dry_run=False,
              summarize_only=False, progress="auto", primary_metric="mean_wasserstein_scaled",
              primary_direction="lower", permutation_seed=2026, baseline_run_minutes=60.,
              budget_hours=20., runtime_margin=1.25, experiment_runner=run_experiment):
    project_root, output_dir = Path(project_root).resolve(), Path(output_dir).resolve()
    seeds = list(dict.fromkeys(int(s) for s in seeds))
    if stage not in {"val", "test"} or (stage == "test" and (smoke or not config.get("frozen"))):
        raise ValueError("Test studies require frozen=true and cannot use smoke mode")
    if not seeds or (not smoke and len(seeds) < 3):
        raise ValueError("Provide at least three distinct generator seeds (except --smoke)")
    if min(seeds) < 0 or split_seed < 0 or permutation_seed < 0:
        raise ValueError("Seeds must be non-negative")
    if primary_direction not in {"lower", "higher"} or not primary_metric:
        raise ValueError("Declare a primary metric and its higher/lower improvement direction")
    for name, value in (("baseline_run_minutes", baseline_run_minutes), ("budget_hours", budget_hours)):
        if not np.isfinite(value) or value <= 0:
            raise ValueError(f"{name} must be positive")
    if not np.isfinite(runtime_margin) or runtime_margin < 1:
        raise ValueError("runtime_margin must be >= 1")
    if smoke:
        seeds = [42]
    plan = {
        "design": "paired_shap_contribution", "config": copy.deepcopy(config),
        "seeds": seeds, "split_seed": int(split_seed), "variants": list(STUDY_VARIANTS),
        "device": device, "stage": stage, "smoke": smoke, "thesis_results": not smoke,
        "primary_metric": primary_metric, "primary_direction": primary_direction,
        "permutation_seed": int(permutation_seed),
        "shuffle_seeds": {str(seed): int(np.random.SeedSequence([permutation_seed, seed]).generate_state(1)[0])
                          for seed in seeds},
        "data_sha256": file_sha256(project_root / config["data_path"]),
        "code_sha256": _code_hash(project_root),
        "priority_coefficients": {"A5": [.5, .3, .2], "A5_NO_SHAP": [0., .6, .4],
                                  "A5_SHUFFLED_SHAP": [.5, .3, .2]},
        "coefficient_order": ["shap", "mismatch", "tail"],
    }
    estimate = len(seeds) * len(STUDY_VARIANTS) / 6 * baseline_run_minutes / 60
    print(f"{len(seeds) * len(STUDY_VARIANTS)} GAN fits; approximately {estimate:.1f} h "
          f"({estimate * runtime_margin:.1f} h with margin). Full configured evaluation retained.", flush=True)
    if dry_run:
        print(json.dumps(plan, indent=2))
        return output_dir
    if not (smoke or summarize_only) and estimate * runtime_margin > budget_hours:
        raise ValueError("Estimated study exceeds budget; inspect --dry-run or increase --budget-hours explicitly")
    plan_path = output_dir / "study_plan.json"
    if output_dir.exists():
        if not (resume or summarize_only):
            raise FileExistsError("Study exists; use --resume or a new output directory")
        if json.loads(plan_path.read_text(encoding="utf-8")) != plan:
            raise ValueError("Study fingerprint mismatch: config, seeds, endpoint, data, device or code changed")
    else:
        if resume or summarize_only:
            raise FileNotFoundError("No existing SHAP contribution study to resume/summarize")
        atomic_write_json(plan_path, plan)
    if summarize_only:
        summarize_contribution(output_dir, plan)
        return output_dir
    atomic_write_json(output_dir / "runtime_budget.json", {
        "gan_fits": len(seeds) * len(STUDY_VARIANTS), "estimated_hours": estimate,
        "buffered_hours": estimate * runtime_margin, "budget_hours": budget_hours,
        "baseline_six_variant_minutes": baseline_run_minutes,
        "note": "Estimate only. Launch budget per invocation; in-flight seed runs are not killed."})
    completed, split_hashes = [], set()
    started = time.monotonic()
    predicted_seconds = baseline_run_minutes * 60 * len(STUDY_VARIANTS) / 6
    atomic_write_json(output_dir / "manifest.json", {"status": "running", "thesis_results": not smoke})
    try:
        for seed in seeds:
            child = output_dir / f"seed{seed}"
            child_manifest = child / "manifest.json"
            is_complete = child_manifest.exists() and json.loads(child_manifest.read_text()).get("status") == "complete"
            if not is_complete:
                if time.monotonic() - started + predicted_seconds * runtime_margin > budget_hours * 3600:
                    atomic_write_json(output_dir / "manifest.json", {
                        "status": "budget_paused", "completed_seeds": completed,
                        "next_seed": seed, "thesis_results": not smoke})
                    summarize_contribution(output_dir, plan)
                    return output_dir
                cfg = copy.deepcopy(config)
                cfg.update(seed=seed, split_seed=split_seed,
                           shap_contribution={"shuffle_seed": plan["shuffle_seeds"][str(seed)]})
                child_start = time.monotonic()
                experiment_runner(cfg, project_root, stage, device, STUDY_VARIANTS,
                                  output_override=child, resume=child_manifest.exists(),
                                  smoke=smoke, progress=progress)
                predicted_seconds = max(predicted_seconds, time.monotonic() - child_start)
            split_hashes.add(validate_child(child, plan, seed))
            if len(split_hashes) > 1:
                raise ValueError("Generator seeds have different split indices")
            completed.append(seed)
            summarize_contribution(output_dir, plan)
    except BaseException as exc:
        atomic_write_json(output_dir / "manifest.json", {
            "status": "interrupted", "completed_seeds": completed, "error": str(exc),
            "thesis_results": not smoke})
        raise
    atomic_write_json(output_dir / "manifest.json", {
        "status": "complete", "completed_seeds": completed,
        "elapsed_seconds": time.monotonic() - started, "thesis_results": not smoke})
    return output_dir


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--seeds", default="42,43,44")
    parser.add_argument("--split-seed", type=int, default=42)
    parser.add_argument("--permutation-seed", type=int, default=2026)
    parser.add_argument("--stage", choices=("val", "test"), default="val")
    parser.add_argument("--device", default="auto")
    parser.add_argument("--progress", choices=("auto", "on", "off"), default="auto")
    parser.add_argument("--primary-metric", default="mean_wasserstein_scaled")
    parser.add_argument("--primary-direction", choices=("lower", "higher"), default="lower")
    parser.add_argument("--baseline-run-minutes", type=float, default=60.)
    parser.add_argument("--budget-hours", type=float, default=20.)
    parser.add_argument("--runtime-margin", type=float, default=1.25)
    for flag in ("smoke", "resume", "dry-run", "summarize-only"):
        parser.add_argument("--" + flag, action="store_true")
    args = vars(parser.parse_args(argv))
    config = _load_config(args.pop("config"))
    args["seeds"] = [int(s.strip()) for s in args["seeds"].split(",")]
    print(run_study(config, Path(__file__).resolve().parents[1], **args))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
