"""Direction-aware utility, fidelity, and trade-off heatmaps for ablations."""

from __future__ import annotations

import os
from pathlib import Path
import tempfile
from typing import Any, Iterable, Mapping

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

from .io_utils import atomic_write_csv


UTILITY_SCORE_LABELS: dict[str, str] = {
    "roc_auc": "ROC-AUC",
    "pr_auc": "PR-AUC",
    "accuracy": "Accuracy",
    "balanced_accuracy": "Balanced accuracy",
    "precision_macro": "Macro precision",
    "recall_macro": "Macro recall",
    "f1_macro": "Macro F1",
    "positive_precision": "Positive precision",
    "positive_recall": "Positive recall",
    "positive_f1": "Positive F1",
}

FIDELITY_SCORE_SPECS: dict[str, dict[str, Any]] = {
    "mean_wasserstein_scaled": {
        "label": "Scaled Wasserstein",
        "ideal": 0.0,
        "direction": "lower",
    },
    "mean_ks": {"label": "Mean KS", "ideal": 0.0, "direction": "lower"},
    "mean_jensen_shannon": {
        "label": "Jensen-Shannon",
        "ideal": 0.0,
        "direction": "lower",
    },
    "correlation_distance": {
        "label": "Correlation distance",
        "ideal": 0.0,
        "direction": "lower",
    },
    "mean_cdf_tail_divergence": {
        "label": "Tail-CDF divergence",
        "ideal": 0.0,
        "direction": "lower",
    },
    "mean_q05_lower_mass_error": {
        "label": "Lower 5% mass error",
        "ideal": 0.0,
        "direction": "lower",
    },
    "mean_q95_upper_mass_error": {
        "label": "Upper 5% mass error",
        "ideal": 0.0,
        "direction": "lower",
    },
    "mean_q99_upper_mass_error": {
        "label": "Upper 1% mass error",
        "ideal": 0.0,
        "direction": "lower",
    },
    "mean_q95_quantile_error_scaled": {
        "label": "Scaled q95 error",
        "ideal": 0.0,
        "direction": "lower",
    },
    "mean_q99_quantile_error_scaled": {
        "label": "Scaled q99 error",
        "ideal": 0.0,
        "direction": "lower",
    },
    "mean_rare_category_frequency_error": {
        "label": "Rare-category error",
        "ideal": 0.0,
        "direction": "lower",
    },
    "rare_outcome_frequency_error": {
        "label": "Rare-outcome error",
        "ideal": 0.0,
        "direction": "lower",
    },
    "detector_auc": {
        "label": "Detector AUC",
        "ideal": 0.5,
        "direction": "target",
    },
    "detector_average_precision": {
        "label": "Detector average precision",
        "ideal": 0.5,
        "direction": "target",
    },
}

PRIVACY_SCORE_SPECS: dict[str, dict[str, Any]] = {
    "privacy_exact_match_rate": {
        "label": "Exact-match rate",
        "ideal": 0.0,
        "direction": "lower",
        "interpretation": "Lower means fewer synthetic rows exactly match real training rows.",
    },
    "privacy_nn_p5_distance_ratio": {
        "label": "NN p5 distance ratio",
        "ideal": 1.0,
        "direction": "higher",
        "numerator": "privacy_synthetic_nn_p5",
        "denominator": "privacy_heldout_nn_p5",
        "interpretation": "1.0 equals the held-out-real p5 distance from real training data.",
    },
    "privacy_median_distance_ratio": {
        "label": "NN median distance ratio",
        "ideal": 1.0,
        "direction": "higher",
        "interpretation": "1.0 equals the held-out-real median distance from training data.",
    },
    "privacy_nn_p95_distance_ratio": {
        "label": "NN p95 distance ratio",
        "ideal": 1.0,
        "direction": "higher",
        "numerator": "privacy_synthetic_nn_p95",
        "denominator": "privacy_heldout_nn_p95",
        "interpretation": "1.0 equals the held-out-real p95 distance from real training data.",
    },
}


def _atomic_save_figure(figure, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary = tempfile.mkstemp(
        prefix=path.name, suffix=".tmp", dir=path.parent
    )
    os.close(descriptor)
    try:
        figure.savefig(temporary, format="png", dpi=180, bbox_inches="tight")
        os.replace(temporary, path)
    finally:
        plt.close(figure)
        if os.path.exists(temporary):
            os.unlink(temporary)


def _variant_order(variants: Iterable[str]) -> list[str]:
    available = list(dict.fromkeys(str(variant) for variant in variants))
    canonical = ["A0", "A1", "A2", "A3", "A4", "A5"]
    return [variant for variant in canonical if variant in available] + sorted(
        set(available) - set(canonical)
    )


def build_utility_heatmap_scores(
    summary: pd.DataFrame,
    real_only: pd.DataFrame,
    utility_tasks: Iterable[Mapping[str, Any]],
    mixture_results: pd.DataFrame | None = None,
) -> pd.DataFrame:
    """Combine real-only, synthetic-only, and additive-1.0 utility."""
    if "variant" not in summary:
        raise ValueError("Ablation summary must contain a variant column")
    if "utility_task" not in real_only:
        raise ValueError("Real-only utility must contain a utility_task column")

    rows: list[dict[str, Any]] = []
    variants = _variant_order(summary["variant"].astype(str))
    indexed = summary.assign(variant=summary["variant"].astype(str)).set_index("variant")
    for task in utility_tasks:
        task_name = str(task["name"])
        balance = str(task.get("balance", "unspecified"))
        reference = real_only[real_only["utility_task"].astype(str) == task_name]
        if reference.empty:
            raise ValueError(f"Real-only utility is missing task {task_name!r}")
        numeric_reference = reference.select_dtypes(include=[np.number]).mean()
        rows.append(
            {
                "utility_task": task_name,
                "target_balance": balance,
                "training_source": "real",
                "training_scenario": "real_only",
                "protocol": "real_only",
                "synthetic_fraction": 0.0,
                "synthetic_share_of_training": 0.0,
                "variant": "REAL",
                "display_label": "Real baseline",
                **{
                    metric: float(numeric_reference[metric])
                    if metric in numeric_reference
                    else np.nan
                    for metric in UTILITY_SCORE_LABELS
                },
            }
        )

        mixture_task = pd.DataFrame()
        if mixture_results is not None and not mixture_results.empty:
            required = {"variant", "utility_task", "protocol", "synthetic_fraction"}
            missing = sorted(required - set(mixture_results.columns))
            if missing:
                raise ValueError(
                    "Mixed utility results are missing heatmap columns: "
                    + ", ".join(missing)
                )
            fractions = pd.to_numeric(
                mixture_results["synthetic_fraction"], errors="coerce"
            )
            mixture_task = mixture_results[
                (mixture_results["utility_task"].astype(str) == task_name)
                & (mixture_results["protocol"].astype(str) == "additive")
                & np.isclose(fractions, 1.0)
            ]

        for variant in variants:
            result = indexed.loc[variant]
            if isinstance(result, pd.DataFrame):
                raise ValueError(f"Ablation summary contains duplicate rows for {variant}")
            rows.append(
                {
                    "utility_task": task_name,
                    "target_balance": balance,
                    "training_source": "synthetic",
                    "training_scenario": "synthetic_only",
                    "protocol": "synthetic_only",
                    "synthetic_fraction": 1.0,
                    "synthetic_share_of_training": 1.0,
                    "variant": variant,
                    "display_label": f"{variant} (100% synthetic)",
                    **{
                        metric: pd.to_numeric(
                            result.get(f"utility_{task_name}_{metric}", np.nan),
                            errors="coerce",
                        )
                        for metric in UTILITY_SCORE_LABELS
                    },
                }
            )
            mixture_variant = mixture_task[
                mixture_task["variant"].astype(str) == variant
            ]
            if not mixture_variant.empty:
                numeric_mixture = mixture_variant.select_dtypes(
                    include=[np.number]
                ).mean()
                rows.append(
                    {
                        "utility_task": task_name,
                        "target_balance": balance,
                        "training_source": "mixed",
                        "training_scenario": "additive_1_0",
                        "protocol": "additive",
                        "synthetic_fraction": 1.0,
                        "synthetic_share_of_training": float(
                            numeric_mixture.get(
                                "synthetic_share_of_training", 0.5
                            )
                        ),
                        "variant": variant,
                        "display_label": f"{variant} (50% real / 50% synthetic)",
                        **{
                            metric: float(numeric_mixture[metric])
                            if metric in numeric_mixture
                            else np.nan
                            for metric in UTILITY_SCORE_LABELS
                        },
                    }
                )

    columns = [
        "utility_task",
        "target_balance",
        "training_source",
        "training_scenario",
        "protocol",
        "synthetic_fraction",
        "synthetic_share_of_training",
        "variant",
        "display_label",
        *UTILITY_SCORE_LABELS,
    ]
    return pd.DataFrame(rows, columns=columns)


def save_utility_heatmap_artifacts(
    summary: pd.DataFrame,
    real_only: pd.DataFrame,
    utility_tasks: Iterable[Mapping[str, Any]],
    output_dir: Path,
    mixture_results: pd.DataFrame | None = None,
) -> tuple[Path, Path]:
    """Atomically save the heatmap's exact values and rendered PNG."""
    scores = build_utility_heatmap_scores(
        summary, real_only, utility_tasks, mixture_results
    )
    if scores.empty:
        raise ValueError("No utility scores are available for the heatmap")
    output_dir.mkdir(parents=True, exist_ok=True)
    table_path = output_dir / "utility_heatmap_scores.csv"
    image_path = output_dir / "utility_heatmap.png"
    atomic_write_csv(table_path, scores)

    task_names = scores["utility_task"].drop_duplicates().tolist()
    figure, axes = plt.subplots(
        len(task_names),
        1,
        figsize=(16, max(4.5, 1.0 + 1.05 * len(scores))),
        squeeze=False,
    )
    for axis, task_name in zip(axes[:, 0], task_names):
        task_scores = scores[scores["utility_task"] == task_name]
        matrix = task_scores.set_index("display_label")[list(UTILITY_SCORE_LABELS)]
        matrix = matrix.rename(columns=UTILITY_SCORE_LABELS)
        sns.heatmap(
            matrix,
            annot=True,
            fmt=".3f",
            cmap="YlGnBu",
            vmin=0.0,
            vmax=1.0,
            linewidths=0.5,
            linecolor="white",
            mask=matrix.isna(),
            cbar_kws={"label": "Absolute score (0-1)"},
            ax=axis,
        )
        for row, column in zip(*np.where(matrix.isna().to_numpy())):
            axis.text(column + 0.5, row + 0.5, "NA", ha="center", va="center")
        balance = str(task_scores["target_balance"].iloc[0])
        axis.set_title(f"{task_name.replace('_', ' ').title()} ({balance})")
        axis.set_xlabel("Utility measurement")
        axis.set_ylabel("Training data")
        axis.tick_params(axis="x", rotation=35)
        axis.tick_params(axis="y", rotation=0)
    figure.suptitle(
        "Real-only, synthetic-only, and additive-1.0 (50/50) utility scores",
        fontsize=16,
        y=1.01,
    )
    figure.tight_layout()

    _atomic_save_figure(figure, image_path)
    return table_path, image_path


def build_fidelity_heatmap_scores(summary: pd.DataFrame) -> pd.DataFrame:
    """Return absolute fidelity values plus direction-aware relative coloring."""
    if "variant" not in summary:
        raise ValueError("Ablation summary must contain a variant column")
    variants = _variant_order(summary["variant"].astype(str))
    indexed = summary.assign(variant=summary["variant"].astype(str)).set_index("variant")
    rows: list[dict[str, Any]] = []
    for metric, specification in FIDELITY_SCORE_SPECS.items():
        if metric not in indexed:
            continue
        values = pd.to_numeric(indexed.loc[variants, metric], errors="coerce")
        ideal = float(specification["ideal"])
        discrepancy = (values - ideal).abs()
        finite = discrepancy[np.isfinite(discrepancy)]
        if finite.empty:
            normalized = pd.Series(np.nan, index=values.index)
        elif np.isclose(float(finite.max()), float(finite.min())):
            normalized = pd.Series(0.5, index=values.index)
            normalized[values.isna()] = np.nan
        else:
            normalized = (discrepancy - float(finite.min())) / (
                float(finite.max()) - float(finite.min())
            )
        for variant in variants:
            rows.append(
                {
                    "variant": variant,
                    "metric": metric,
                    "display_metric": str(specification["label"]),
                    "direction": str(specification["direction"]),
                    "ideal_value": ideal,
                    "absolute_value": values.loc[variant],
                    "distance_from_ideal": discrepancy.loc[variant],
                    "normalized_discrepancy": normalized.loc[variant],
                }
            )
    return pd.DataFrame(rows)


def build_privacy_heatmap_scores(summary: pd.DataFrame) -> pd.DataFrame:
    """Return comparable privacy-proxy values and relative risk coloring."""
    if "variant" not in summary:
        raise ValueError("Ablation summary must contain a variant column")
    variants = _variant_order(summary["variant"].astype(str))
    indexed = summary.assign(variant=summary["variant"].astype(str)).set_index("variant")
    rows: list[dict[str, Any]] = []
    for metric, specification in PRIVACY_SCORE_SPECS.items():
        numerator = specification.get("numerator")
        denominator = specification.get("denominator")
        if numerator and denominator:
            if numerator not in indexed or denominator not in indexed:
                continue
            numerator_values = pd.to_numeric(indexed.loc[variants, numerator], errors="coerce")
            denominator_values = pd.to_numeric(
                indexed.loc[variants, denominator], errors="coerce"
            ).replace(0.0, np.nan)
            values = numerator_values / denominator_values
        else:
            if metric not in indexed:
                continue
            values = pd.to_numeric(indexed.loc[variants, metric], errors="coerce")

        direction = str(specification["direction"])
        # A lower risk score always means the more favorable privacy proxy.
        risk = values if direction == "lower" else -values
        finite = risk[np.isfinite(risk)]
        if finite.empty:
            normalized = pd.Series(np.nan, index=values.index)
        elif np.isclose(float(finite.max()), float(finite.min())):
            normalized = pd.Series(0.5, index=values.index)
            normalized[values.isna()] = np.nan
        else:
            normalized = (risk - float(finite.min())) / (
                float(finite.max()) - float(finite.min())
            )
        for variant in variants:
            rows.append(
                {
                    "variant": variant,
                    "metric": metric,
                    "display_metric": str(specification["label"]),
                    "direction": direction,
                    "heldout_equivalence_value": float(specification["ideal"]),
                    "interpretation": str(specification["interpretation"]),
                    "absolute_value": values.loc[variant],
                    "normalized_privacy_risk": normalized.loc[variant],
                }
            )
    return pd.DataFrame(rows)


def build_utility_fidelity_privacy_tradeoff_scores(
    summary: pd.DataFrame,
    utility_scores: pd.DataFrame,
    utility_tasks: Iterable[Mapping[str, Any]],
) -> pd.DataFrame:
    """Compare every available domain with A0 and align positive with improvement."""
    variants = _variant_order(summary["variant"].astype(str))
    indexed = summary.assign(variant=summary["variant"].astype(str)).set_index("variant")
    if "A0" not in indexed.index:
        raise ValueError("Combined trade-off normalization requires the A0 reference")
    rows: list[dict[str, Any]] = []

    for task in utility_tasks:
        task_name = str(task["name"])
        task_label = task_name.replace("_", " ").title()
        task_scores = utility_scores[utility_scores["utility_task"] == task_name]
        if task_scores.empty:
            continue
        if "training_scenario" not in task_scores:
            task_scores = task_scores.assign(training_scenario="synthetic_only")
        scenarios = [
            scenario
            for scenario in ["synthetic_only", "additive_1_0"]
            if scenario in set(task_scores["training_scenario"].astype(str))
        ]
        for scenario in scenarios:
            scenario_scores = task_scores[
                task_scores["training_scenario"].astype(str) == scenario
            ]
            task_indexed = scenario_scores.set_index("variant")
            if "A0" not in task_indexed.index:
                raise ValueError(
                    f"Utility trade-off reference is missing for {task_name!r} "
                    f"under {scenario!r}"
                )
            scenario_label = (
                "100% synthetic"
                if scenario == "synthetic_only"
                else "Additive 1.0 (50% real / 50% synthetic)"
            )
            metric_suffix = "" if scenario == "synthetic_only" else "_additive_1_0"
            for metric, label in UTILITY_SCORE_LABELS.items():
                reference_value = pd.to_numeric(
                    task_indexed.loc["A0", metric], errors="coerce"
                )
                variant_values = pd.to_numeric(
                    task_indexed.reindex(variants)[metric], errors="coerce"
                )
                if not np.isfinite(reference_value) or not variant_values.notna().any():
                    continue
                metric_key = f"utility_{task_name}_{metric}{metric_suffix}"
                for variant in variants:
                    value = variant_values.loc[variant]
                    rows.append(
                        {
                            "domain": "utility",
                            "utility_task": task_name,
                            "utility_training_scenario": scenario,
                            "metric": metric,
                            "metric_key": metric_key,
                            "display_metric": (
                                f"Utility | {task_label} | {scenario_label} | {label}"
                            ),
                            "variant": variant,
                            "reference": "A0",
                            "direction_rule": (
                                "variant_minus_a0"
                                if scenario == "synthetic_only"
                                else "variant_minus_a0_same_training_scenario"
                            ),
                            "ideal_value": 1.0,
                            "absolute_value": value,
                            "reference_value": reference_value,
                            "improvement_delta": value - reference_value,
                        }
                    )

    for metric, specification in FIDELITY_SCORE_SPECS.items():
        if metric not in indexed:
            continue
        reference_value = pd.to_numeric(indexed.loc["A0", metric], errors="coerce")
        if not np.isfinite(reference_value):
            continue
        direction = str(specification["direction"])
        ideal = float(specification["ideal"])
        for variant in variants:
            value = pd.to_numeric(indexed.loc[variant, metric], errors="coerce")
            if direction == "target":
                improvement = abs(reference_value - ideal) - abs(value - ideal)
                rule = "a0_distance_to_ideal_minus_variant_distance_to_ideal"
            else:
                improvement = reference_value - value
                rule = "a0_error_minus_variant_error"
            rows.append(
                {
                    "domain": "fidelity",
                    "utility_task": None,
                    "metric": metric,
                    "metric_key": f"fidelity_{metric}",
                    "display_metric": f"Fidelity | {specification['label']}",
                    "variant": variant,
                    "reference": "A0",
                    "direction_rule": rule,
                    "ideal_value": ideal,
                    "absolute_value": value,
                    "reference_value": reference_value,
                    "improvement_delta": improvement,
                }
            )

    privacy = build_privacy_heatmap_scores(summary)
    if not privacy.empty:
        privacy_indexed = privacy.set_index(["metric", "variant"])
        for metric, specification in PRIVACY_SCORE_SPECS.items():
            if metric not in set(privacy["metric"]):
                continue
            reference_value = pd.to_numeric(
                privacy_indexed.loc[(metric, "A0"), "absolute_value"], errors="coerce"
            )
            if not np.isfinite(reference_value):
                continue
            direction = str(specification["direction"])
            for variant in variants:
                value = pd.to_numeric(
                    privacy_indexed.loc[(metric, variant), "absolute_value"],
                    errors="coerce",
                )
                if direction == "lower":
                    improvement = reference_value - value
                    rule = "a0_privacy_risk_minus_variant_privacy_risk"
                else:
                    improvement = value - reference_value
                    rule = "variant_distance_ratio_minus_a0_distance_ratio"
                rows.append(
                    {
                        "domain": "privacy_proxy",
                        "utility_task": None,
                        "metric": metric,
                        "metric_key": f"privacy_proxy_{metric}",
                        "display_metric": f"Privacy proxy | {specification['label']}",
                        "variant": variant,
                        "reference": "A0",
                        "direction_rule": rule,
                        "ideal_value": float(specification["ideal"]),
                        "absolute_value": value,
                        "reference_value": reference_value,
                        "improvement_delta": improvement,
                    }
                )

    result = pd.DataFrame(rows)
    if result.empty:
        return result
    result["normalized_improvement"] = np.nan
    for _, indices in result.groupby("metric_key", sort=False).groups.items():
        values = pd.to_numeric(result.loc[indices, "improvement_delta"], errors="coerce")
        maximum = float(values.abs().max(skipna=True))
        if np.isfinite(maximum) and maximum > 0:
            result.loc[indices, "normalized_improvement"] = values / maximum
        else:
            result.loc[indices, "normalized_improvement"] = 0.0
            result.loc[values.index[values.isna()], "normalized_improvement"] = np.nan
    return result


def save_fidelity_privacy_tradeoff_heatmap_artifacts(
    summary: pd.DataFrame,
    utility_scores: pd.DataFrame,
    utility_tasks: Iterable[Mapping[str, Any]],
    output_dir: Path,
) -> tuple[Path, Path, Path, Path, Path, Path]:
    """Save absolute fidelity/privacy and combined normalized trade-off artifacts."""
    variants = _variant_order(summary["variant"].astype(str))
    fidelity = build_fidelity_heatmap_scores(summary)
    if fidelity.empty:
        raise ValueError("No fidelity scores are available for the heatmap")
    fidelity_table = output_dir / "fidelity_heatmap_scores.csv"
    fidelity_image = output_dir / "fidelity_heatmap.png"
    atomic_write_csv(fidelity_table, fidelity)

    metric_order = fidelity["display_metric"].drop_duplicates().tolist()
    raw_fidelity = fidelity.pivot(
        index="display_metric", columns="variant", values="absolute_value"
    ).reindex(index=metric_order, columns=variants)
    fidelity_colors = fidelity.pivot(
        index="display_metric", columns="variant", values="normalized_discrepancy"
    ).reindex(index=metric_order, columns=variants)
    figure, axis = plt.subplots(
        1, 1, figsize=(10, max(7.0, 0.58 * len(metric_order) + 2.0))
    )
    sns.heatmap(
        fidelity_colors,
        annot=raw_fidelity,
        fmt=".3f",
        cmap="YlOrRd",
        vmin=0.0,
        vmax=1.0,
        linewidths=0.5,
        linecolor="white",
        mask=raw_fidelity.isna(),
        cbar_kws={"label": "Relative discrepancy within each metric"},
        ax=axis,
    )
    axis.set_title("Absolute fidelity values (color: best to worst within metric)")
    axis.set_xlabel("Ablation variant")
    axis.set_ylabel("Fidelity measurement")
    axis.tick_params(axis="x", rotation=0)
    axis.tick_params(axis="y", rotation=0)
    figure.tight_layout()
    _atomic_save_figure(figure, fidelity_image)

    privacy = build_privacy_heatmap_scores(summary)
    if privacy.empty:
        raise ValueError("No privacy-proxy scores are available for the heatmap")
    privacy_table = output_dir / "privacy_proxy_heatmap_scores.csv"
    privacy_image = output_dir / "privacy_proxy_heatmap.png"
    atomic_write_csv(privacy_table, privacy)
    privacy_order = privacy["display_metric"].drop_duplicates().tolist()
    raw_privacy = privacy.pivot(
        index="display_metric", columns="variant", values="absolute_value"
    ).reindex(index=privacy_order, columns=variants)
    privacy_colors = privacy.pivot(
        index="display_metric", columns="variant", values="normalized_privacy_risk"
    ).reindex(index=privacy_order, columns=variants)
    figure, axis = plt.subplots(
        1, 1, figsize=(10, max(4.5, 0.75 * len(privacy_order) + 2.0))
    )
    sns.heatmap(
        privacy_colors,
        annot=raw_privacy,
        fmt=".3f",
        cmap="YlOrRd",
        vmin=0.0,
        vmax=1.0,
        linewidths=0.5,
        linecolor="white",
        mask=raw_privacy.isna(),
        cbar_kws={"label": "Relative privacy risk within each proxy"},
        ax=axis,
    )
    axis.set_title(
        "Absolute privacy proxies (distance ratio 1.0 = held-out-real baseline)"
    )
    axis.set_xlabel("Ablation variant")
    axis.set_ylabel("Privacy proxy (not a formal guarantee)")
    axis.tick_params(axis="x", rotation=0)
    axis.tick_params(axis="y", rotation=0)
    figure.tight_layout()
    _atomic_save_figure(figure, privacy_image)

    tradeoff = build_utility_fidelity_privacy_tradeoff_scores(
        summary, utility_scores, utility_tasks
    )
    if tradeoff.empty:
        raise ValueError("No utility/fidelity trade-off scores are available")
    tradeoff_table = output_dir / "utility_fidelity_privacy_tradeoff_scores.csv"
    tradeoff_image = output_dir / "utility_fidelity_privacy_tradeoff_heatmap.png"
    atomic_write_csv(tradeoff_table, tradeoff)
    tradeoff_order = tradeoff["display_metric"].drop_duplicates().tolist()
    raw_tradeoff = tradeoff.pivot(
        index="display_metric", columns="variant", values="improvement_delta"
    ).reindex(index=tradeoff_order, columns=variants)
    normalized_tradeoff = tradeoff.pivot(
        index="display_metric", columns="variant", values="normalized_improvement"
    ).reindex(index=tradeoff_order, columns=variants)
    figure, axis = plt.subplots(
        1, 1, figsize=(11, max(10.0, 0.48 * len(tradeoff_order) + 2.0))
    )
    sns.heatmap(
        normalized_tradeoff,
        annot=raw_tradeoff,
        fmt="+.3f",
        cmap="RdYlGn",
        center=0.0,
        vmin=-1.0,
        vmax=1.0,
        linewidths=0.5,
        linecolor="white",
        mask=raw_tradeoff.isna(),
        cbar_kws={"label": "Normalized improvement (-1 to +1)"},
        ax=axis,
    )
    axis.set_title(
        "Utility, fidelity, and privacy proxy versus A0 (positive = improvement)"
    )
    axis.set_xlabel("Ablation variant")
    axis.set_ylabel("Measurement and reference")
    axis.tick_params(axis="x", rotation=0)
    axis.tick_params(axis="y", rotation=0)
    figure.tight_layout()
    _atomic_save_figure(figure, tradeoff_image)
    return (
        fidelity_table,
        fidelity_image,
        privacy_table,
        privacy_image,
        tradeoff_table,
        tradeoff_image,
    )


def save_ablation_heatmap_artifacts(
    summary: pd.DataFrame,
    real_only: pd.DataFrame,
    utility_tasks: Iterable[Mapping[str, Any]],
    output_dir: Path,
    mixture_results: pd.DataFrame | None = None,
) -> list[Path]:
    """Save all absolute and direction-aware ablation heatmaps."""
    utility_tasks = list(utility_tasks)
    utility_table, utility_image = save_utility_heatmap_artifacts(
        summary, real_only, utility_tasks, output_dir, mixture_results
    )
    utility_scores = pd.read_csv(utility_table)
    fidelity_paths = save_fidelity_privacy_tradeoff_heatmap_artifacts(
        summary, utility_scores, utility_tasks, output_dir
    )
    return [utility_table, utility_image, *fidelity_paths]
