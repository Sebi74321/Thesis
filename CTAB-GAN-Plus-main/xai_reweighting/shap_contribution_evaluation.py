"""Notebook-facing evaluation panels with generator-seed (not RF-repeat) spread.

This reporting-only module consumes the validated, per-seed inventory saved by
the SHAP study. It never loads a generator or changes the training fingerprint.
"""

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

from .io_utils import atomic_write_csv, atomic_write_json


VARIANTS = ("A0", "A5_NO_SHAP", "A5_SHUFFLED_SHAP", "A5")
UTILITY_METRICS = (
    "roc_auc", "pr_auc", "accuracy", "balanced_accuracy", "precision_macro",
    "recall_macro", "f1_macro", "positive_precision", "positive_recall", "positive_f1",
)
METRIC_GROUPS = {
    "Global fidelity": ("mean_wasserstein", "mean_wasserstein_scaled", "mean_ks",
                        "mean_jensen_shannon", "correlation_distance"),
    "Tails and rare events": (
        "mean_cdf_tail_divergence", "mean_q05_lower_mass_error",
        "mean_q95_upper_mass_error", "mean_q99_upper_mass_error",
        "mean_q05_quantile_error_scaled", "mean_q95_quantile_error_scaled",
        "mean_q99_quantile_error_scaled", "mean_rare_category_frequency_error",
        "rare_outcome_frequency_error"),
    "Detector": ("detector_auc", "detector_average_precision", "detector_accuracy"),
    "Privacy proxies": ("privacy_exact_match_rate", "privacy_median_distance_ratio",
                       "privacy_synthetic_nn_p5", "privacy_synthetic_nn_p50",
                       "privacy_synthetic_nn_p95", "privacy_heldout_nn_p5",
                       "privacy_heldout_nn_p50", "privacy_heldout_nn_p95"),
    "Training": ("training_rows", "synthetic_rows", "weight_mean", "weight_fraction_capped"),
}
COMPOSITION_METRICS = (
    "training_positive_fraction", "target_positive_fraction",
    "source_synthetic_positive_fraction", "unadjusted_training_positive_fraction",
    "synthetic_share_of_training", "training_rows",
)
DIMENSIONS = ["domain", "utility_task", "protocol", "synthetic_fraction", "feature", "metric"]
RECORD_COLUMNS = ["seed", "variant", *DIMENSIONS, "value", "within_run_std", "observations"]
LABELS = {
    "pr_auc": "Average precision (PR)", "roc_auc": "ROC-AUC",
    "mean_jensen_shannon": "Jensen–Shannon distance",
    "mean_cdf_tail_divergence": "Upper-tail ECDF gap",
    "privacy_median_distance_ratio": "Median NN distance ratio (not a privacy guarantee)",
}


def evaluation_records(raw):
    """Select original measurements, retaining tasks, fractions and features.

    The input inventory has already averaged utility repetitions within a seed.
    Reject ambiguous keys instead of treating duplicates as independent seeds.
    """
    records = []
    headline = raw[raw.artifact == "ablation_summary.csv"]
    tasks = sorted({m[len("utility_"):-len("_roc_auc")]
                    for m in headline.metric if m.startswith("utility_") and m.endswith("_roc_auc")})
    groups = {m: domain for domain, metrics in METRIC_GROUPS.items() for m in metrics}

    def add(row, context, domain, metric=None, **overrides):
        entry = dict(seed=int(row.seed), variant=context.get("variant", "REAL"),
                     domain=domain, utility_task=context.get("utility_task", ""),
                     protocol=context.get("protocol", ""),
                     synthetic_fraction=context.get("synthetic_fraction", -1.),
                     feature=context.get("feature", ""), metric=metric or row.metric,
                     value=row.value, within_run_std=row.within_run_std,
                     observations=row.observations)
        entry.update(overrides)
        records.append(entry)

    for row in raw.itertuples(index=False):
        context = json.loads(row.context)
        variant = context.get("variant")
        if row.artifact == "ablation_summary.csv" and variant in VARIANTS:
            if row.metric in groups:
                add(row, context, groups[row.metric])
            for task in tasks:
                for metric in UTILITY_METRICS:
                    # Legacy flat keys collide: mortality/ balanced_accuracy
                    # and mortality_balanced/ accuracy. Binary macro recall is
                    # exactly balanced accuracy and has an unambiguous key.
                    source = "recall_macro" if metric == "balanced_accuracy" else metric
                    if row.metric == f"utility_{task}_{source}":
                        add(row, context, "Utility", metric, utility_task=task,
                            protocol="synthetic_only", synthetic_fraction=1.)
        elif row.artifact in {"utility_mixture_results.csv", "utility_real_only_baseline.csv"}:
            if variant not in VARIANTS and row.artifact != "utility_real_only_baseline.csv":
                continue
            domain = "Utility" if row.metric in UTILITY_METRICS else "Training composition"
            if row.metric in (*UTILITY_METRICS, *COMPOSITION_METRICS):
                add(row, context, domain, variant=variant or "REAL")
        elif row.artifact == "top_shap_feature_variant_metrics.csv" and variant in VARIANTS:
            if row.metric == "distribution_discrepancy":
                add(row, context, "Prioritized feature fidelity",
                    metric={"continuous": "wasserstein_scaled", "categorical": "jensen_shannon"}.get(
                        context.get("kind"), "distribution_discrepancy"))
        elif row.artifact.startswith("feature_metrics_") and row.artifact.endswith(".csv"):
            variant = row.artifact[len("feature_metrics_"):-4]
            if variant in VARIANTS:
                add(row, context, "Per-feature fidelity", variant=variant)

    result = pd.DataFrame(records, columns=RECORD_COLUMNS)
    result["value"] = pd.to_numeric(result.value, errors="coerce").replace([np.inf, -np.inf], np.nan)
    keys = ["seed", "variant", *DIMENSIONS]
    if result.duplicated(keys).any():
        raise ValueError("Ambiguous evaluation contexts: more than one value per generator seed")
    return result


def seed_summary(records, expected_seeds, *, paired=False):
    keys = (["variant", "reference"] if paired else ["variant"]) + DIMENSIONS
    value = "delta" if paired else "value"
    result = records.groupby(keys, dropna=False)[value].agg(
        mean="mean", std="std", n="count", minimum="min", maximum="max",
    ).reset_index()
    result["expected_seeds"] = len(expected_seeds)
    result["unavailable_seeds"] = len(expected_seeds) - result.n
    return result  # ddof=1; undefined for a single available seed, never zero-filled.


def evaluation_deltas(records):
    """Subtract matched seed values BEFORE estimating the SD of differences."""
    keys = ["seed", *DIMENSIONS]
    pairs = []
    for reference in ("A0", "A5_NO_SHAP", "A5_SHUFFLED_SHAP"):
        left = records[records.variant.isin(VARIANTS)].copy()
        if reference != "A0":
            left = left[left.variant == "A5"]
        right = records[records.variant == reference][keys + ["value"]]
        pair = left.merge(right.rename(columns={"value": "reference_value"}),
                          on=keys, how="left", validate="many_to_one")
        pair["reference"] = reference
        pairs.append(pair)
    # Only the fixed-protocol mixtures share the real-only classifier settings.
    # Do not pair the separate 300-tree main utility with a 100-tree RF baseline.
    left = records[(records.domain == "Utility") & records.protocol.isin(["additive", "replacement"])]
    real_keys = ["seed", "domain", "utility_task", "feature", "metric"]
    right = records[(records.variant == "REAL") & (records.domain == "Utility")]
    pair = left.merge(right[real_keys + ["value"]].rename(columns={"value": "reference_value"}),
                      on=real_keys, how="left", validate="many_to_one")
    pair["reference"] = "REAL"
    pairs.append(pair)
    result = pd.concat(pairs, ignore_index=True)
    result["delta"] = result.value - result.reference_value
    return result


def write_evaluation_reports(output_dir, raw=None, plan=None):
    """Refresh display reports without training, evaluation, or fingerprint bypass."""
    output_dir = Path(output_dir)
    if plan is None:
        plan = json.loads((output_dir / "study_plan.json").read_text(encoding="utf-8"))
    if raw is None:
        raw = pd.read_csv(output_dir / "study_run_metrics.csv")
    if not set(raw.seed.unique()).issubset(set(plan["seeds"])):
        raise ValueError("Inventory contains generator seeds outside the saved study plan")
    records = evaluation_records(raw)
    pairs = evaluation_deltas(records)
    tables = {
        "study_evaluation_seed_metrics.csv": records,
        "study_evaluation_summary.csv": seed_summary(records, plan["seeds"]),
        "study_evaluation_paired_seed_deltas.csv": pairs,
        "study_evaluation_delta_summary.csv": seed_summary(pairs, plan["seeds"], paired=True),
    }
    for name, frame in tables.items():
        atomic_write_csv(output_dir / name, frame)
    atomic_write_json(output_dir / "study_evaluation_reporting.json", {
        "source": "study_run_metrics.csv: saved inventory of validated completed children",
        "available_seeds": sorted(int(s) for s in raw.seed.unique()),
        "expected_seeds": plan["seeds"], "stage": plan["stage"], "smoke": plan["smoke"],
        "spread": "sample SD (ddof=1) across generator-seed means; n=1 SD is undefined",
        "delta": "paired within generator seed, then summarized; variant minus reference",
        "utility": "RF repetitions averaged inside each seed; tasks and fractions kept separate",
        "legacy_balanced_accuracy": "Main binary utility uses unambiguous macro recall, which equals balanced accuracy",
        "limits": "Reporting-only refresh does not reread child results or rerun evaluations. It does not authorize resuming changed training code.",
    })
    return tables


def plot_metric_panels(summary, title, *, order=VARIANTS, delta=False):
    """One labelled axis per metric; error bars represent seed SD, never SE/CI."""
    import matplotlib.pyplot as plt

    if summary.empty:
        return None
    contexts = [c for c in DIMENSIONS if c != "metric"]
    if len(summary[contexts].drop_duplicates()) != 1:
        raise ValueError("Select one task/protocol/fraction/feature context before plotting")
    if summary.duplicated(["variant", "metric"]).any():
        raise ValueError("Select one reference before plotting deltas")
    metrics = summary.metric.drop_duplicates().tolist()
    fig, axes = plt.subplots(int(np.ceil(len(metrics) / 3)), 3,
                             figsize=(17, 4.2 * int(np.ceil(len(metrics) / 3))), squeeze=False)
    labels = [v for v in order if v in set(summary.variant)]
    for ax, metric in zip(axes.flat, metrics):
        g = summary[summary.metric == metric].set_index("variant").reindex(labels)
        x = np.arange(len(labels))
        ax.scatter(x, g["mean"], color="#4C78A8")
        finite = g["mean"].notna() & g["std"].notna()
        ax.errorbar(x[finite], g.loc[finite, "mean"], yerr=g.loc[finite, "std"],
                    fmt="none", capsize=4, color="#4C78A8")
        if delta:
            ax.axhline(0, color="black", linestyle="--", linewidth=1)
        for i, row in enumerate(g.itertuples()):
            if pd.notna(row.mean):
                ax.annotate(f"n={int(row.n)}", (i, row.mean), xytext=(4, 5), textcoords="offset points", fontsize=8)
        ax.set_xticks(x, labels, rotation=20, ha="right")
        ax.set(xlabel="Variant", ylabel=("Paired delta: " if delta else "Absolute: ") + metric,
               title=LABELS.get(metric, metric.replace("_", " ")))
        ax.grid(axis="y", alpha=.2)
    for ax in axes.flat[len(metrics):]:
        ax.remove()
    fig.suptitle(title + " — mean ± sample SD across generator seeds", y=1.01)
    fig.tight_layout()
    return fig


def plot_utility_heatmap(summary, title):
    """Absolute utility mean colours with mean, seed SD and count annotations."""
    import matplotlib.pyplot as plt

    if summary.empty:
        return None
    contexts = [c for c in DIMENSIONS if c != "metric"]
    if len(summary[contexts].drop_duplicates()) != 1 or set(summary.domain) != {"Utility"}:
        raise ValueError("A utility heatmap needs one task/protocol/fraction context")
    order = [v for v in ("REAL", *VARIANTS) if v in set(summary.variant)]
    metrics = summary.metric.drop_duplicates().tolist()
    means = summary.pivot(index="variant", columns="metric", values="mean").reindex(index=order, columns=metrics)
    stds = summary.pivot(index="variant", columns="metric", values="std").reindex_like(means)
    counts = summary.pivot(index="variant", columns="metric", values="n").reindex_like(means)
    fig, ax = plt.subplots(figsize=(max(9, 1.9 * len(metrics)), 1.1 * len(order) + 2))
    cmap = plt.get_cmap("Blues").with_extremes(bad="#eeeeee")
    im = ax.imshow(np.ma.masked_invalid(means.to_numpy()), vmin=0, vmax=1, cmap=cmap, aspect="auto")
    for i in range(len(order)):
        for j in range(len(metrics)):
            mean, std, n = means.iloc[i, j], stds.iloc[i, j], counts.iloc[i, j]
            if pd.isna(mean):
                label = "NA"
            elif pd.notna(std):
                label = f"{mean:.3f} ± {std:.3f}"
            else:
                label = f"{mean:.3f} ± NA"
            label += f"\nn={int(n) if pd.notna(n) else 0}"
            ax.text(j, i, label, ha="center", va="center", fontsize=9,
                    color="white" if pd.notna(mean) and mean > .6 else "black")
    ax.set_xticks(range(len(metrics)), [LABELS.get(m, m.replace("_", " ")) for m in metrics], rotation=25, ha="right")
    ax.set_yticks(range(len(order)), order)
    ax.set(xlabel="Utility metric", ylabel="Variant", title=title + "\nMean ± sample SD across generator seeds")
    fig.colorbar(im, ax=ax, label="Absolute mean (0–1)")
    fig.tight_layout()
    return fig


def plot_mixture_curves(summary, task, protocol, *, reference="REAL"):
    """Task-specific delta curves; use the absolute table for original scores."""
    import matplotlib.pyplot as plt

    selected = summary[(summary.domain == "Utility") & (summary.utility_task == task)
                       & (summary.protocol == protocol) & (summary.reference == reference)]
    if selected.empty:
        return None
    metrics = [m for m in UTILITY_METRICS if m in set(selected.metric)]
    colors = dict(zip(VARIANTS, plt.get_cmap("tab10").colors))
    fig, axes = plt.subplots(int(np.ceil(len(metrics) / 3)), 3,
                             figsize=(17, 4.2 * int(np.ceil(len(metrics) / 3))), squeeze=False)
    for ax, metric in zip(axes.flat, metrics):
        for variant in VARIANTS:
            g = selected[(selected.metric == metric) & (selected.variant == variant)].sort_values("synthetic_fraction")
            if g.empty:
                continue
            line, = ax.plot(g.synthetic_fraction, g["mean"], marker="o", label=variant, color=colors[variant])
            finite = g["std"].notna() & g["mean"].notna()
            ax.errorbar(g.loc[finite, "synthetic_fraction"], g.loc[finite, "mean"],
                        yerr=g.loc[finite, "std"], fmt="none", capsize=3, color=line.get_color())
        ax.axhline(0, color="black", linestyle="--", linewidth=1)
        ax.set(xlabel="Synthetic / real-base row ratio" if protocol == "additive" else "Synthetic training share",
               ylabel=f"{metric}: delta versus {reference}", title=LABELS.get(metric, metric.replace("_", " ")))
        ax.legend(fontsize=8)
        ax.grid(alpha=.2)
    for ax in axes.flat[len(metrics):]:
        ax.remove()
    fig.suptitle(f"{task} — {protocol}: mean paired delta ± sample SD across generator seeds", y=1.01)
    fig.tight_layout()
    return fig


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--study-dir", required=True, type=Path)
    args = parser.parse_args(argv)
    write_evaluation_reports(args.study_dir)
    print(f"Refreshed evaluation displays from saved inventory: {args.study_dir}")


if __name__ == "__main__":
    main()
