import numpy as np
import pandas as pd

from xai_reweighting.utility_reporting import (
    FIDELITY_SCORE_SPECS,
    PRIVACY_SCORE_SPECS,
    UTILITY_SCORE_LABELS,
    build_fidelity_heatmap_scores,
    build_privacy_heatmap_scores,
    build_utility_heatmap_scores,
    build_utility_fidelity_privacy_tradeoff_scores,
    save_ablation_heatmap_artifacts,
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
    mixture_rows = []
    for task in tasks:
        for position, variant in enumerate(variants):
            for repeat in range(2):
                mixture_rows.append(
                    {
                        "variant": variant,
                        "utility_task": task["name"],
                        "protocol": "additive",
                        "synthetic_fraction": 1.0,
                        "repeat": repeat,
                        **_metric_values(0.80 + position / 100),
                    }
                )
                # A replacement 50/50 row must not be selected for this view.
                mixture_rows.append(
                    {
                        "variant": variant,
                        "utility_task": task["name"],
                        "protocol": "replacement",
                        "synthetic_fraction": 0.5,
                        "repeat": repeat,
                        **_metric_values(0.20 + position / 100),
                    }
                )
    mixture_results = pd.DataFrame(mixture_rows)

    scores = build_utility_heatmap_scores(
        summary, real_only, tasks, mixture_results
    )

    for task in tasks:
        task_scores = scores[scores["utility_task"] == task["name"]]
        expected = ["Real baseline"]
        for variant in variants:
            expected.extend(
                [
                    f"{variant} (100% synthetic)",
                    f"{variant} (50% real / 50% synthetic)",
                ]
            )
        assert task_scores["display_label"].tolist() == expected
        assert np.isclose(task_scores.iloc[0]["positive_precision"], 0.71)
        assert np.isclose(task_scores.iloc[0]["positive_recall"], 0.71)
        assert np.isclose(task_scores.iloc[0]["positive_f1"], 0.71)
        a5_synthetic = task_scores[
            (task_scores["variant"] == "A5")
            & (task_scores["training_scenario"] == "synthetic_only")
        ]
        a5_mixed = task_scores[
            (task_scores["variant"] == "A5")
            & (task_scores["training_scenario"] == "additive_1_0")
        ]
        assert np.isclose(a5_synthetic["f1_macro"].iloc[0], 0.55)
        assert np.isclose(a5_mixed["f1_macro"].iloc[0], 0.85)
        assert a5_mixed["protocol"].iloc[0] == "additive"
        assert np.isclose(a5_mixed["synthetic_fraction"].iloc[0], 1.0)
        assert np.isclose(
            a5_mixed["synthetic_share_of_training"].iloc[0], 0.5
        )

    table_path, image_path = save_utility_heatmap_artifacts(
        summary, real_only, tasks, tmp_path, mixture_results
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


def test_tradeoff_directions_use_a0_for_every_domain(tmp_path):
    summary = pd.DataFrame(
        [
            {
                "variant": "A0",
                **{f"utility_mortality_{metric}": 0.60 for metric in UTILITY_SCORE_LABELS},
                "mean_wasserstein_scaled": 0.40,
                "detector_auc": 0.90,
                "privacy_exact_match_rate": 0.02,
                "privacy_synthetic_nn_p5": 0.80,
                "privacy_heldout_nn_p5": 1.00,
                "privacy_median_distance_ratio": 0.85,
                "privacy_synthetic_nn_p95": 1.20,
                "privacy_heldout_nn_p95": 1.50,
            },
            {
                "variant": "A1",
                **{f"utility_mortality_{metric}": 0.65 for metric in UTILITY_SCORE_LABELS},
                "mean_wasserstein_scaled": 0.30,
                "detector_auc": 0.80,
                "privacy_exact_match_rate": 0.01,
                "privacy_synthetic_nn_p5": 0.90,
                "privacy_heldout_nn_p5": 1.00,
                "privacy_median_distance_ratio": 0.95,
                "privacy_synthetic_nn_p95": 1.35,
                "privacy_heldout_nn_p95": 1.50,
            },
        ]
    )
    real_only = pd.DataFrame(
        [{"utility_task": "mortality", **_metric_values(0.70)}]
    )
    tasks = [{"name": "mortality", "balance": "imbalanced"}]
    mixture = pd.DataFrame(
        [
            {
                "variant": "A0",
                "utility_task": "mortality",
                "protocol": "additive",
                "synthetic_fraction": 1.0,
                **_metric_values(0.70),
            },
            {
                "variant": "A1",
                "utility_task": "mortality",
                "protocol": "additive",
                "synthetic_fraction": 1.0,
                **_metric_values(0.78),
            },
        ]
    )
    utility = build_utility_heatmap_scores(summary, real_only, tasks, mixture)

    tradeoff = build_utility_fidelity_privacy_tradeoff_scores(summary, utility, tasks)
    utility_a1 = tradeoff[
        (tradeoff["metric_key"] == "utility_mortality_positive_recall")
        & (tradeoff["variant"] == "A1")
    ].iloc[0]
    mixed_utility_a1 = tradeoff[
        (
            tradeoff["metric_key"]
            == "utility_mortality_positive_recall_additive_1_0"
        )
        & (tradeoff["variant"] == "A1")
    ].iloc[0]
    wasserstein_a1 = tradeoff[
        (tradeoff["metric_key"] == "fidelity_mean_wasserstein_scaled")
        & (tradeoff["variant"] == "A1")
    ].iloc[0]
    detector_a1 = tradeoff[
        (tradeoff["metric_key"] == "fidelity_detector_auc")
        & (tradeoff["variant"] == "A1")
    ].iloc[0]
    exact_match_a1 = tradeoff[
        (tradeoff["metric_key"] == "privacy_proxy_privacy_exact_match_rate")
        & (tradeoff["variant"] == "A1")
    ].iloc[0]
    p5_ratio_a1 = tradeoff[
        (tradeoff["metric_key"] == "privacy_proxy_privacy_nn_p5_distance_ratio")
        & (tradeoff["variant"] == "A1")
    ].iloc[0]

    assert np.isclose(utility_a1["improvement_delta"], 0.05)
    assert utility_a1["reference"] == "A0"
    assert utility_a1["direction_rule"] == "variant_minus_a0"
    assert np.isclose(mixed_utility_a1["improvement_delta"], 0.08)
    assert mixed_utility_a1["reference"] == "A0"
    assert mixed_utility_a1["utility_training_scenario"] == "additive_1_0"
    assert (
        mixed_utility_a1["direction_rule"]
        == "variant_minus_a0_same_training_scenario"
    )
    assert np.isclose(wasserstein_a1["improvement_delta"], 0.10)
    assert np.isclose(detector_a1["improvement_delta"], 0.10)
    assert wasserstein_a1["reference"] == "A0"
    assert detector_a1["direction_rule"].startswith("a0_distance_to_ideal")
    assert np.isclose(exact_match_a1["improvement_delta"], 0.01)
    assert np.isclose(p5_ratio_a1["improvement_delta"], 0.10)
    assert exact_match_a1["reference"] == "A0"

    fidelity = build_fidelity_heatmap_scores(summary)
    assert set(fidelity["metric"]) == {"mean_wasserstein_scaled", "detector_auc"}
    assert set(FIDELITY_SCORE_SPECS).issuperset(fidelity["metric"])
    privacy = build_privacy_heatmap_scores(summary)
    assert set(privacy["metric"]) == set(PRIVACY_SCORE_SPECS)
    assert np.isclose(
        privacy[
            (privacy["metric"] == "privacy_nn_p95_distance_ratio")
            & (privacy["variant"] == "A1")
        ]["absolute_value"].iloc[0],
        0.90,
    )

    paths = save_ablation_heatmap_artifacts(summary, real_only, tasks, tmp_path)
    assert {path.name for path in paths} == {
        "utility_heatmap_scores.csv",
        "utility_heatmap.png",
        "fidelity_heatmap_scores.csv",
        "fidelity_heatmap.png",
        "privacy_proxy_heatmap_scores.csv",
        "privacy_proxy_heatmap.png",
        "utility_fidelity_privacy_tradeoff_scores.csv",
        "utility_fidelity_privacy_tradeoff_heatmap.png",
    }
    assert all(path.is_file() and path.stat().st_size > 0 for path in paths)
