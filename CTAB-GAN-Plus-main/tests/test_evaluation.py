import numpy as np
import pandas as pd

from xai_reweighting.detector import _aggregate_encoded_shap, train_detector
from xai_reweighting.evaluation import (
    evaluate_fidelity_and_tails,
    evaluate_privacy,
    evaluate_utility,
    evaluate_variant,
)


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
        positive_label="1",
        seed=42,
        n_estimators=20,
        n_jobs=1,
    )

    assert metrics["utility_accuracy"] >= 0.95
    assert metrics["utility_roc_auc"] >= 0.95
    assert metrics["utility_decision_rule"] == "random_forest_argmax"
    assert metrics["utility_f1_macro"] >= 0.95


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


def test_categorical_shap_is_grouped_before_taking_absolute_mean():
    # Encoded order: continuous x, category=a, category=b. The categorical
    # dummy contributions cancel within each row and must therefore have zero
    # grouped importance; summing their separate absolute means would be 1.3.
    encoded_shap = np.array([
        [1.0, 0.8, -0.8],
        [-1.0, -0.5, 0.5],
    ])

    importance = _aggregate_encoded_shap(
        encoded_shap, continuous=["x"], categorical=["category"], category_sizes=[2]
    )

    assert importance["x"] == 1.0
    assert importance["category"] == 0.0


def test_variant_reports_balanced_and_imbalanced_utility_separately():
    x = np.arange(80)
    real = pd.DataFrame(
        {
            "x": x,
            "gender": np.where(x % 2, "F", "M"),
            "mortality": (x % 5 == 0).astype(int),
        }
    )
    metrics, _ = evaluate_variant(
        real.iloc[:60].reset_index(drop=True),
        real.iloc[60:].reset_index(drop=True),
        real.iloc[:60].reset_index(drop=True),
        "mortality",
        ["gender", "mortality"],
        ["x"],
        utility_tasks=[
            {"name": "mortality", "target_col": "mortality", "positive_label": "1"},
            {"name": "gender", "target_col": "gender", "positive_label": "F"},
        ],
        seed=42,
        n_jobs=1,
        n_estimators=10,
    )

    assert "utility_mortality_roc_auc" in metrics
    assert "utility_mortality_positive_recall" in metrics
    assert "utility_gender_roc_auc" in metrics
    assert "utility_gender_f1_macro" in metrics


def test_privacy_sampling_limits_only_nearest_neighbor_workload():
    train = pd.DataFrame({"x": np.arange(40), "category": np.arange(40) % 2})
    heldout = pd.DataFrame({"x": np.arange(40, 60), "category": np.arange(20) % 2})
    synthetic = train.iloc[:30].copy()

    metrics = evaluate_privacy(
        train,
        heldout,
        synthetic,
        categorical_cols=["category"],
        n_jobs=1,
        seed=42,
        max_reference_rows=10,
        max_query_rows=5,
    )

    assert metrics["privacy_reference_rows"] == 10
    assert metrics["privacy_synthetic_query_rows"] == 5
    assert metrics["privacy_heldout_query_rows"] == 5
    assert metrics["privacy_exact_match_rate"] == 1.0
