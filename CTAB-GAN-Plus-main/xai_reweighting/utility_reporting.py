"""Utility-evaluation tables and heatmaps for ablation runs."""

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
) -> pd.DataFrame:
    """Combine real-only and synthetic-training utility on one absolute scale."""
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

        for variant in variants:
            result = indexed.loc[variant]
            if isinstance(result, pd.DataFrame):
                raise ValueError(f"Ablation summary contains duplicate rows for {variant}")
            rows.append(
                {
                    "utility_task": task_name,
                    "target_balance": balance,
                    "training_source": "synthetic",
                    "variant": variant,
                    "display_label": variant,
                    **{
                        metric: pd.to_numeric(
                            result.get(f"utility_{task_name}_{metric}", np.nan),
                            errors="coerce",
                        )
                        for metric in UTILITY_SCORE_LABELS
                    },
                }
            )

    columns = [
        "utility_task",
        "target_balance",
        "training_source",
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
) -> tuple[Path, Path]:
    """Atomically save the heatmap's exact values and rendered PNG."""
    scores = build_utility_heatmap_scores(summary, real_only, utility_tasks)
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
        "Real-only and ablation-variant utility scores",
        fontsize=16,
        y=1.01,
    )
    figure.tight_layout()

    descriptor, temporary = tempfile.mkstemp(
        prefix=image_path.name, suffix=".tmp", dir=output_dir
    )
    os.close(descriptor)
    try:
        figure.savefig(temporary, format="png", dpi=180, bbox_inches="tight")
        os.replace(temporary, image_path)
    finally:
        plt.close(figure)
        if os.path.exists(temporary):
            os.unlink(temporary)
    return table_path, image_path
