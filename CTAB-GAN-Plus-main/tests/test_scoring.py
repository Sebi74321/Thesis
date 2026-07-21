import numpy as np
import pandas as pd

from xai_reweighting.scoring import (
    build_region_definitions,
    compute_feature_components,
    compute_feature_priority,
    compute_row_weights,
    normalize_signal,
)


def _components(tail_value=0.0):
    return pd.DataFrame(
        {
            "feature": ["x", "y", "z"],
            "shap_raw": [1.0, 0.0, 0.0],
            "mismatch_raw": [0.0, 1.0, 0.0],
            "tail_raw": [0.0, 0.0, tail_value],
            "shap": [1.0, 0.0, 0.0],
            "mismatch": [0.0, 1.0, 0.0],
            "tail": [0.0, 0.0, tail_value],
        }
    )


def test_normalize_signal_handles_zero_and_scale():
    assert normalize_signal(pd.Series([0.0, 0.0])).tolist() == [0.0, 0.0]
    assert normalize_signal(pd.Series([0.0, 2.0])).tolist() == [0.0, 1.0]


def test_variant_formulas_and_tail_isolation():
    a2 = compute_feature_priority(_components(1.0), "A2", top_k=3).set_index("feature")
    a4 = compute_feature_priority(_components(1.0), "A4", top_k=3).set_index("feature")
    a5_without = compute_feature_priority(_components(0.0), "A5", top_k=3).set_index("feature")
    a5_with = compute_feature_priority(_components(1.0), "A5", top_k=3).set_index("feature")
    assert a2.loc["x", "combined_raw"] == 0.0
    assert a2.loc["z", "combined_raw"] == 0.0
    assert np.isclose(a4.loc["x", "combined_raw"], 0.625)
    assert np.isclose(a4.loc["y", "combined_raw"], 0.375)
    assert a4.loc["z", "combined_raw"] == 0.0
    assert np.isclose(a5_with.loc["x", "combined_raw"], 0.5)
    assert np.isclose(a5_with.loc["y", "combined_raw"], 0.3)
    assert np.isclose(a5_with.loc["z", "combined_raw"], 0.2)
    assert a5_without.loc["z", "combined_raw"] != a5_with.loc["z", "combined_raw"]


def test_duplicate_quantiles_missing_values_and_weight_bounds():
    train = pd.DataFrame({"x": [1.0] * 9 + [np.nan], "cat": ["a"] * 8 + ["b", None]})
    audit = pd.DataFrame({"x": [1.0, np.nan] * 5, "cat": ["a", "b"] * 5})
    synthetic = pd.DataFrame({"x": [1.0] * 10, "cat": ["a"] * 10})
    definitions = build_region_definitions(train, audit, synthetic, ["cat"], ["x"])
    priority = pd.DataFrame(
        {
            "feature": ["cat", "x"],
            "shap": [0.0, 0.0],
            "mismatch": [1.0, 0.5],
            "tail": [0.0, 0.0],
            "selected": [True, True],
            "priority": [2 / 3, 1 / 3],
        }
    )
    weights, _ = compute_row_weights(train, priority, definitions, alpha=1.0, w_max=1.5)
    assert np.isfinite(weights).all()
    assert weights.between(1.0, 1.5).all()


def test_target_minority_is_included_in_tail_score_above_five_percent():
    real = pd.DataFrame({"target": [0] * 8 + [1] * 2})
    synthetic = pd.DataFrame({"target": [0] * 10})
    result = compute_feature_components(
        real, synthetic, {"target": 0.0}, ["target"], [], target_col="target"
    ).set_index("feature")
    assert result.loc["target", "tail_raw"] > 0
