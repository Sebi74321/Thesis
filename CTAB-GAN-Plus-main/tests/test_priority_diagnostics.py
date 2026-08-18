from types import SimpleNamespace

import numpy as np
import pandas as pd

from xai_reweighting.priority_diagnostics import prioritized_feature_diagnostics


def test_priority_diagnostics_flag_artifact_patterns_and_detector_dependence():
    real_train = pd.DataFrame(
        {"spo2_max": np.arange(100, dtype=float), "other": np.arange(100), "target": [0, 1] * 50}
    )
    real_audit = real_train.iloc[:40].reset_index(drop=True)
    synthetic = real_audit.copy()
    synthetic["spo2_max"] = 120.0
    priority = pd.DataFrame(
        {
            "feature": ["spo2_max", "other", "target"],
            "shap": [0.8, 0.1, 0.1],
            "mismatch": [0.8, 0.1, 0.1],
            "tail": [0.8, 0.1, 0.1],
            "combined_raw": [0.8, 0.1, 0.1],
            "selected": [True, False, False],
            "priority": [1.0, 0.0, 0.0],
        }
    )

    def fake_detector(real, synthetic, categorical_cols, **kwargs):
        if list(real.columns) == ["spo2_max"]:
            auc = 0.95
        elif "spo2_max" not in real.columns:
            auc = 0.60
        else:
            auc = 0.90
        return SimpleNamespace(metrics={"detector_auc": auc})

    summary, spikes, detector, correlations = prioritized_feature_diagnostics(
        real_train,
        real_audit,
        synthetic,
        {"A5": priority},
        "target",
        ["target"],
        ["spo2_max", "other"],
        baseline_detector_auc=0.99,
        n_estimators=5,
        detector_fn=fake_detector,
    )

    row = summary.iloc[0]
    assert row["feature"] == "spo2_max"
    assert row["synthetic_out_of_train_range_rate"] == 1.0
    assert row["univariate_detector_auc"] == 0.95
    assert np.isclose(row["detector_auc_drop_when_removed"], 0.30)
    assert bool(row["requires_artifact_review"])
    assert row["artifact_flag_count"] >= 2
    assert not spikes.empty
    assert not detector.empty
    assert correlations.iloc[0]["other_feature"] == "other"
