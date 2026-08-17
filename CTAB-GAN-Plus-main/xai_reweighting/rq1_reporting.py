"""Tables and plots for the RQ1 multi-model comparison."""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


PLOT_GROUPS = {
    "utility": ["utility_roc_auc", "utility_pr_auc", "utility_recall", "rare_event_recall"],
    "fidelity": ["mean_wasserstein_scaled", "correlation_distance", "detector_auc"],
    "tails": [
        "mean_cdf_tail_divergence",
        "rq1_mean_fixed_tail_mass_error",
        "rq1_joint_rare_region_mass_error",
        "rare_outcome_frequency_error",
    ],
    "privacy_proxy": [
        "privacy_exact_match_rate",
        "privacy_median_distance_ratio",
        "privacy_synthetic_nn_p50",
    ],
}


def aggregate_results(results: pd.DataFrame, reference: dict | None = None) -> pd.DataFrame:
    numeric = [
        column for column in results.select_dtypes(include=[np.number]).columns
        if column not in {"seed"}
    ]
    rows = []
    for model, frame in results.groupby("model", sort=False):
        row = {"model": model, "seeds_completed": int(frame["seed"].nunique())}
        for column in numeric:
            row[f"{column}_mean"] = float(frame[column].mean())
            row[f"{column}_std"] = float(frame[column].std(ddof=1)) if len(frame) > 1 else 0.0
            if reference is not None and isinstance(reference.get(column), (int, float)):
                row[f"{column}_vs_real_reference"] = row[f"{column}_mean"] - float(reference[column])
        rows.append(row)
    return pd.DataFrame(rows)


def save_comparison_plots(summary: pd.DataFrame, output_dir: Path) -> list[Path]:
    output_dir.mkdir(parents=True, exist_ok=True)
    paths = []
    for group, metrics in PLOT_GROUPS.items():
        available = [metric for metric in metrics if f"{metric}_mean" in summary]
        if not available:
            continue
        figure, axes = plt.subplots(
            1, len(available), figsize=(4.2 * len(available), 4.2), squeeze=False
        )
        for axis, metric in zip(axes[0], available):
            means = summary[f"{metric}_mean"].to_numpy(dtype=float)
            errors = summary.get(f"{metric}_std", pd.Series(np.zeros(len(summary)))).to_numpy(dtype=float)
            axis.bar(summary["model"], means, yerr=errors, capsize=3)
            axis.set_title(metric.replace("_", " "))
            axis.tick_params(axis="x", rotation=25)
            axis.grid(axis="y", alpha=0.25)
        figure.suptitle(f"RQ1 {group.replace('_', ' ').title()} Comparison")
        figure.tight_layout()
        path = output_dir / f"rq1_{group}.png"
        figure.savefig(path, dpi=180, bbox_inches="tight")
        plt.close(figure)
        paths.append(path)
    return paths
