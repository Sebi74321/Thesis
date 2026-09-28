import numpy as np
import pandas as pd

from xai_reweighting.rq1_evaluation import evaluate_rq1_specific, real_real_reference


def _frames():
    reference = pd.DataFrame(
        {"x": np.arange(100), "category": ["rare"] * 5 + ["common"] * 95, "target": [1] * 10 + [0] * 90}
    )
    real = reference.sample(frac=1.0, random_state=1).reset_index(drop=True)
    synthetic = real.copy()
    return reference, real, synthetic


def test_identical_data_has_zero_fixed_region_errors():
    reference, real, synthetic = _frames()
    metrics, details = evaluate_rq1_specific(
        reference, real, synthetic, "target", ["category", "target"], ["x"]
    )
    assert metrics["rq1_mean_fixed_tail_mass_error"] == 0.0
    assert metrics["rq1_joint_rare_region_mass_error"] == 0.0
    assert details["mass_error"].max() == 0.0


def test_real_reference_is_reproducible():
    reference, real, _ = _frames()
    first, _ = real_real_reference(reference, real, "target", ["category", "target"], ["x"], 42)
    second, _ = real_real_reference(reference, real, "target", ["category", "target"], ["x"], 42)
    pd.testing.assert_series_equal(pd.Series(first), pd.Series(second))
