from types import SimpleNamespace

import numpy as np
import pandas as pd

from xai_reweighting.diagnostics import baseline_detector_diagnostics


def test_baseline_diagnostics_rank_and_condition_on_outcome():
    real = pd.DataFrame(
        {
            "x": np.r_[np.arange(10), np.arange(10, 20)],
            "category": ["a", "b"] * 10,
            "target": [0] * 10 + [1] * 10,
        }
    )
    synthetic = real.copy()
    synthetic.loc[synthetic["target"] == 1, "x"] += 20
    synthetic.loc[synthetic["target"] == 1, "category"] = "b"
    components = pd.DataFrame(
        {
            "feature": ["target", "x", "category"],
            "shap_raw": [1.0, 0.8, 0.4],
            "mismatch_raw": [0.0, 0.0, 0.0],
            "tail_raw": [0.0, 0.0, 0.0],
            "shap": [1.0, 0.8, 0.4],
            "mismatch": [0.0, 0.0, 0.0],
            "tail": [0.0, 0.0, 0.0],
        }
    )
    calls = []

    def fake_detector(real_group, synthetic_group, categorical_cols, **kwargs):
        calls.append((real_group, synthetic_group, categorical_cols, kwargs))
        return SimpleNamespace(
            metrics={
                "detector_auc": 0.75,
                "detector_average_precision": 0.76,
                "detector_accuracy": 0.7,
                "n_real": len(real_group),
                "n_synthetic": len(synthetic_group),
            }
        )

    ranking, feature_gaps, category_gaps, detector_metrics = baseline_detector_diagnostics(
        real,
        synthetic,
        components,
        "target",
        ["category", "target"],
        ["x"],
        top_n=2,
        n_estimators=20,
        n_jobs=1,
        detector_fn=fake_detector,
    )

    assert ranking.iloc[0]["feature"] == "target"
    assert ranking.loc[ranking["selected_for_conditional_diagnostic"], "feature"].tolist() == [
        "x",
        "category",
    ]
    assert len(calls) == 2
    assert all("target" not in call[0].columns for call in calls)
    assert detector_metrics["detector_auc"].tolist() == [0.75, 0.75]
    death_x = feature_gaps.loc[
        (feature_gaps["outcome"] == "1") & (feature_gaps["feature"] == "x")
    ].iloc[0]
    assert death_x["wasserstein_scaled"] > 0
    assert not category_gaps.empty


def test_baseline_diagnostics_marks_tiny_outcome_groups():
    real = pd.DataFrame({"x": range(5), "target": [0, 0, 0, 0, 1]})
    synthetic = real.copy()
    components = pd.DataFrame({"feature": ["x"], "shap_raw": [1.0]})

    _, _, _, detector_metrics = baseline_detector_diagnostics(
        real, synthetic, components, "target", ["target"], ["x"], top_n=1
    )

    rare = detector_metrics.loc[detector_metrics["outcome"] == "1"].iloc[0]
    assert rare["status"] == "insufficient_rows"
    assert np.isnan(rare["detector_auc"])
