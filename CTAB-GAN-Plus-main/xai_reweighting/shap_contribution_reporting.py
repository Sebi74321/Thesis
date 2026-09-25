"""Paired generator-seed contrasts for the SHAP contribution experiment."""

import copy
import json
import os
import tempfile
from pathlib import Path

import numpy as np
import pandas as pd

from .io_utils import atomic_write_csv, atomic_write_json, file_sha256
from .run_ablation import _fingerprint, apply_smoke_overrides
from .sensitivity_reporting import collect_run, spread


STUDY_VARIANTS = ("A0", "A5_NO_SHAP", "A5_SHUFFLED_SHAP", "A5")
CONTRASTS = (("A5", "A5_NO_SHAP"), ("A5", "A5_SHUFFLED_SHAP"),
             ("A5", "A0"), ("A5_NO_SHAP", "A0"), ("A5_SHUFFLED_SHAP", "A0"))


def validate_child(child, plan, seed):
    """Check a completed child before accepting it as a paired observation."""
    manifest = json.loads((child / "manifest.json").read_text(encoding="utf-8"))
    config = json.loads((child / "config.json").read_text(encoding="utf-8"))
    expected = copy.deepcopy(plan["config"])
    expected.update(seed=seed, split_seed=plan["split_seed"], stage=plan["stage"],
                    device=plan["device"], variants=list(STUDY_VARIANTS), smoke=plan["smoke"],
                    shap_contribution={"shuffle_seed": plan["shuffle_seeds"][str(seed)]})
    if plan["smoke"]:
        apply_smoke_overrides(expected)
    if (manifest.get("status") != "complete"
            or config != expected
            or manifest.get("fingerprint") != _fingerprint(
                config, plan["data_sha256"], plan["code_sha256"])
            or set(manifest.get("variants_completed", [])) != set(STUDY_VARIANTS)):
        raise ValueError(f"Completed study child has incompatible provenance: {child}")
    for variant in STUDY_VARIANTS:
        for name in (f".{variant}.complete", f"synthetic_{variant}.csv", f"metrics_{variant}.json"):
            if not (child / name).is_file():
                raise FileNotFoundError(f"Completed child is missing {name}: {child}")
    summary = pd.read_csv(child / "ablation_summary.csv")
    if summary.variant.tolist() != list(STUDY_VARIANTS):
        raise ValueError(f"Incomplete or duplicate study summary rows: {child}")
    return file_sha256(child / "split_indices.json")


def variant_records(collected):
    """Remove variant identity from matching keys, not from evaluation settings."""
    records = []
    for row in collected.to_dict("records"):
        context = json.loads(row["context"])
        variant = context.pop("variant", None)
        artifact = row["artifact"]
        # Longest first: A5 is a prefix of the control identifiers.
        for candidate in sorted(STUDY_VARIANTS, key=len, reverse=True):
            stem, extension = os.path.splitext(artifact)
            suffix = "_" + candidate
            if stem.endswith(suffix):
                if variant is not None and variant != candidate:
                    raise ValueError(f"Conflicting variant identifiers in {artifact}")
                variant = candidate
                artifact = stem[:-len(suffix)] + "_{variant}" + extension
                break
        if variant not in STUDY_VARIANTS:
            continue  # Shared A0 audit / real-only tables remain in the raw inventory.
        # These are already derived comparisons, not independent measurements.
        if row["artifact"] == "ablation_deltas.csv" or row["metric"].startswith(
                ("delta_", "real_baseline_", "diagnostic_delta_")):
            continue
        context.pop("comparison", None)
        records.append({**row, "variant": variant, "artifact": artifact,
                        "context": json.dumps(context, sort_keys=True)})
    return pd.DataFrame(records)


def paired_contrasts(measurements):
    keys = ["seed", "artifact", "context", "metric"]
    if measurements.duplicated(["variant", *keys]).any():
        raise ValueError("Ambiguous paired measurements; evaluation contexts must be unique")
    pairs = []
    for variant, control in CONTRASTS:
        left = measurements[measurements.variant == variant][keys + ["value"]]
        right = measurements[measurements.variant == control][keys + ["value"]]
        # Outer join retains missing/undefined measurements instead of silently
        # claiming a complete comparison from only the successful subset.
        pair = left.merge(right, on=keys, how="outer", suffixes=("", "_control"), validate="one_to_one")
        pair["delta"] = pair["value"] - pair["value_control"]
        pair["comparison"] = f"{variant}-{control}"
        pairs.append(pair)
    return pd.concat(pairs, ignore_index=True)


def mechanism_diagnostics(child, seed):
    """Check whether the feature and sampling interventions actually differ."""
    baseline = pd.read_csv(child / "feature_scores_A5.csv").set_index("feature")
    weights = pd.read_csv(child / "row_weights_A5.csv").weight.to_numpy(dtype=float)
    probabilities = weights / weights.sum()
    selected = set(baseline.index[baseline.selected])
    records = []
    for variant in STUDY_VARIANTS[1:]:
        scores = pd.read_csv(child / f"feature_scores_{variant}.csv").set_index("feature")
        scores = scores.reindex(baseline.index)
        control_weights = pd.read_csv(child / f"row_weights_{variant}.csv").weight.to_numpy(dtype=float)
        if len(control_weights) != len(weights):
            raise ValueError("Paired weighting controls must have identical base training rows")
        control_probabilities = control_weights / control_weights.sum()
        control_selected = set(scores.index[scores.selected])
        records.append({
            "seed": seed, "variant": variant,
            "nonzero_original_shap_features": int((baseline.shap != 0).sum()),
            "changed_shap_values": int((~np.isclose(scores.shap, baseline.shap)).sum()),
            "selected_overlap_with_A5": len(selected & control_selected),
            "selected_jaccard_with_A5": len(selected & control_selected) / max(1, len(selected | control_selected)),
            "priority_l1_distance_to_A5": float(np.abs(scores.priority - baseline.priority).sum()),
            "augmentation_probability_tv_to_A5": float(.5 * np.abs(control_probabilities - probabilities).sum()),
            "sampling_identical_to_A5": bool(np.array_equal(control_probabilities, probabilities)),
        })
    return records


def summarize_contribution(output_dir, plan):
    rows, coverage, statuses, hashes, mechanisms = [], [], [], set(), []
    for seed in plan["seeds"]:
        child = output_dir / f"seed{seed}"
        path = child / "manifest.json"
        status = json.loads(path.read_text()).get("status") if path.exists() else "not_started"
        statuses.append({"seed": seed, "status": status})
        if status != "complete":
            continue
        hashes.add(validate_child(child, plan, seed))
        values, inventory = collect_run(child, "shap_contribution", seed)
        rows.append(values)
        coverage.append(inventory)
        mechanisms.extend(mechanism_diagnostics(child, seed))
    if len(hashes) > 1:
        raise ValueError("Generator seeds must use identical split indices")
    atomic_write_csv(output_dir / "study_run_coverage.csv", pd.DataFrame(statuses))
    if not rows:
        return
    raw = pd.concat(rows, ignore_index=True)
    measurements = variant_records(raw)
    atomic_write_csv(output_dir / "study_run_metrics.csv", raw)
    from .shap_contribution_evaluation import write_evaluation_reports
    write_evaluation_reports(output_dir, raw, plan)
    atomic_write_csv(output_dir / "study_mechanism_diagnostics.csv", pd.DataFrame(mechanisms))
    atomic_write_csv(output_dir / "study_artifact_coverage.csv", pd.concat(coverage, ignore_index=True))
    absolute = spread(measurements, ["variant", "artifact", "context", "metric"])
    absolute["expected_seeds"] = len(plan["seeds"])
    absolute["unavailable_seeds"] = absolute.expected_seeds - absolute.n
    atomic_write_csv(output_dir / "study_seed_spread.csv", absolute)
    paired = paired_contrasts(measurements)
    atomic_write_csv(output_dir / "study_paired_deltas.csv", paired)
    deltas = spread(paired, ["comparison", "artifact", "context", "metric"], "delta")
    deltas["expected_seeds"] = len(plan["seeds"])
    deltas["unavailable_pairs"] = deltas.expected_seeds - deltas.n
    atomic_write_csv(output_dir / "study_delta_spread.csv", deltas)
    primary = paired[(paired.artifact == "ablation_summary.csv") &
                     (paired.metric == plan["primary_metric"])].copy()
    sign = 1 if plan["primary_direction"] == "higher" else -1
    primary["improvement"] = sign * primary.delta
    atomic_write_csv(output_dir / "study_primary_pairs.csv", primary)
    primary_spread = spread(primary, ["comparison", "metric"], "improvement")
    wins = primary.groupby("comparison").improvement.apply(lambda x: int((x > 0).sum()))
    primary_spread["improved_seeds"] = primary_spread.comparison.map(wins)
    primary_spread["expected_seeds"] = len(plan["seeds"])
    atomic_write_csv(output_dir / "study_primary_summary.csv", primary_spread)
    weights = absolute[absolute.artifact == "row_weight_summary_{variant}.json"]
    atomic_write_csv(output_dir / "study_weight_diagnostics.csv", weights)
    atomic_write_json(output_dir / "study_interpretation.json", {
        "primary_metric": plan["primary_metric"], "primary_direction": plan["primary_direction"],
        "primary_available_pairs": int(primary.delta.notna().sum()),
        "delta_definition": "variant minus control; original metric units",
        "primary_improvement_definition": "positive means better; direction frozen before training",
        "uncertainty_unit": "generator seed; utility repeats averaged within seed",
        "limits": ["One fixed data split: spread is training variability, not population uncertainty.",
                   "Three seeds provide descriptive evidence, not a reliable significance claim.",
                   "One SHAP permutation per generator seed is a negative control, not a permutation test.",
                   "Shuffling preserves SHAP magnitudes, not final row-weight concentration.",
                   "No-SHAP is a renormalized replacement policy, not an additive causal coefficient.",
                   "Inspect secondary utility, tail and privacy-proxy results, including regressions."]})
    plot_primary_pairs(primary, output_dir)


def plot_primary_pairs(primary, output_dir):
    if primary.empty:
        return
    import matplotlib.pyplot as plt

    comparisons = [f"{a}-{b}" for a, b in CONTRASTS[:2]]
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.5), squeeze=False)
    for ax, comparison in zip(axes.flat, comparisons):
        data = primary[primary.comparison == comparison].sort_values("seed")
        finite = data.dropna(subset=["improvement"])
        if not finite.empty:
            ax.scatter(finite.seed.astype(str), finite.improvement, label="Paired generator seed")
            ax.axhline(finite.improvement.mean(), color="tab:orange", label="Mean improvement")
        ax.axhline(0, color="black", linestyle="--", linewidth=1)
        ax.set(title=comparison, xlabel="Generator seed",
               ylabel=f"{primary.metric.iloc[0]} improvement\n(positive = better)")
        ax.legend()
    fig.tight_layout()
    fd, temporary = tempfile.mkstemp(suffix=".png", dir=output_dir)
    os.close(fd)
    try:
        fig.savefig(temporary, dpi=160)
        os.replace(temporary, output_dir / "study_primary_pairs.png")
    finally:
        plt.close(fig)
        if os.path.exists(temporary):
            os.unlink(temporary)
