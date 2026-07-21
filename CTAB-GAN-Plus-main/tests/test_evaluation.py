import numpy as np
import pandas as pd

from xai_reweighting.detector import train_detector
from xai_reweighting.evaluation import evaluate_fidelity_and_tails, evaluate_utility


def test_identical_continuous_data_has_zero_tail_cdf_divergence():
    frame = pd.DataFrame({"x": np.linspace(-2.0, 3.0, 101)})
    metrics, details = evaluate_fidelity_and_tails(frame, frame.copy(), [], ["x"])

    assert metrics["mean_cdf_tail_divergence"] == 0.0
    assert details.loc[0, "wasserstein"] == 0.0
    assert details.loc[0, "ks"] == 0.0


def test_utility_accepts_equivalent_real_and_generated_category_dtypes():
    x = np.arange(80)
    real = pd.DataFrame(
        {"x": x, "category": x % 2, "target": (x >= 40).astype(int)}
    )
    synthetic = real.copy()
    synthetic["category"] = synthetic["category"].astype(str)
    synthetic["target"] = synthetic["target"].astype(str)

    metrics = evaluate_utility(
        synthetic,
        real,
        target_col="target",
        categorical_cols=["category", "target"],
        seed=42,
        n_estimators=20,
        n_jobs=1,
    )

    assert metrics["utility_accuracy"] >= 0.95
    assert metrics["utility_roc_auc"] >= 0.95


def test_detector_accepts_equivalent_mixed_categorical_dtypes():
    x = np.arange(40)
    real = pd.DataFrame({"x": x, "category": x % 2})
    synthetic = real.copy()
    synthetic["category"] = synthetic["category"].astype(str)

    result = train_detector(
        real,
        synthetic,
        categorical_cols=["category"],
        seed=42,
        n_estimators=10,
        compute_shap=False,
        n_jobs=1,
    )

    assert 0.0 <= result.metrics["detector_auc"] <= 1.0
