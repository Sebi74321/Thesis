import numpy as np
import pandas as pd

from xai_reweighting.utility_reporting import (
    UTILITY_SCORE_LABELS,
    build_utility_heatmap_scores,
    save_utility_heatmap_artifacts,
)


def _metric_values(value):
    return {metric: value for metric in UTILITY_SCORE_LABELS}


def test_heatmap_scores_include_real_and_every_available_ablation_variant(tmp_path):
    variants = ["A0", "A1", "A2", "A3", "A4", "A5"]
    tasks = [
        {"name": "mortality", "balance": "imbalanced"},
        {"name": "mortality_balanced", "balance": "balanced"},
    ]
    summary_rows = []
    for position, variant in enumerate(variants):
        row = {"variant": variant}
        for task in tasks:
            for metric in UTILITY_SCORE_LABELS:
                row[f"utility_{task['name']}_{metric}"] = 0.50 + position / 100
        summary_rows.append(row)
    summary = pd.DataFrame(summary_rows)

    real_rows = []
    for task in tasks:
        for repeat, value in enumerate([0.70, 0.72]):
            real_rows.append(
                {
                    "utility_task": task["name"],
                    "target_balance": task["balance"],
                    "repeat": repeat,
                    **_metric_values(value),
                }
            )
    real_only = pd.DataFrame(real_rows)

    scores = build_utility_heatmap_scores(summary, real_only, tasks)

    for task in tasks:
        task_scores = scores[scores["utility_task"] == task["name"]]
        assert task_scores["display_label"].tolist() == ["Real baseline", *variants]
        assert np.isclose(task_scores.iloc[0]["positive_precision"], 0.71)
        assert np.isclose(task_scores.iloc[0]["positive_recall"], 0.71)
        assert np.isclose(task_scores.iloc[0]["positive_f1"], 0.71)
        assert np.isclose(task_scores.iloc[-1]["f1_macro"], 0.55)

    table_path, image_path = save_utility_heatmap_artifacts(
        summary, real_only, tasks, tmp_path
    )
    assert table_path.is_file()
    assert image_path.is_file()
    assert image_path.stat().st_size > 0


def test_missing_utility_measurement_remains_missing_instead_of_zero():
    summary = pd.DataFrame([{"variant": "A0", "utility_mortality_roc_auc": 0.8}])
    real_only = pd.DataFrame(
        [{"utility_task": "mortality", "roc_auc": 0.9, "positive_recall": 0.6}]
    )

    scores = build_utility_heatmap_scores(
        summary,
        real_only,
        [{"name": "mortality", "balance": "imbalanced"}],
    )

    assert np.isnan(scores.loc[scores["variant"] == "A0", "positive_recall"].iloc[0])
    assert scores.loc[scores["variant"] == "REAL", "positive_recall"].iloc[0] == 0.6
