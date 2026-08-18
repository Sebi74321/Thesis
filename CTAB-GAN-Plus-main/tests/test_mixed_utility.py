import numpy as np
import pandas as pd

from xai_reweighting.mixed_utility import (
    _stratified_sample,
    evaluate_mixed_utility_curve,
    evaluate_real_only_baseline,
    summarize_mixed_utility,
)


def _frame(start, size):
    x = np.arange(start, start + size)
    return pd.DataFrame(
        {
            "x": x,
            "category": np.where(x % 2, "a", "b"),
            "target": (x % 5 == 0).astype(int),
        }
    )


def test_stratified_sample_is_exact_and_deterministic():
    frame = _frame(0, 100)
    first = _stratified_sample(frame, 25, "target", seed=42)
    second = _stratified_sample(frame, 25, "target", seed=42)

    assert len(first) == 25
    pd.testing.assert_frame_equal(first, second)
    assert first["target"].sum() == 5


def test_mixed_curves_preserve_sizes_and_include_real_baselines():
    real_train = _frame(0, 100)
    synthetic = real_train.sample(frac=1.0, random_state=7).reset_index(drop=True)
    real_eval = _frame(150, 50)
    real_only = evaluate_real_only_baseline(
        real_train,
        real_eval,
        "target",
        ["category", "target"],
        positive_label="1",
        repeats=1,
        n_estimators=10,
        n_jobs=1,
    )

    result = evaluate_mixed_utility_curve(
        "A5",
        real_train,
        synthetic,
        real_eval,
        "target",
        ["category", "target"],
        real_only,
        positive_label="1",
        additive_fractions=[0.0, 0.5, 1.0],
        replacement_fractions=[0.0, 0.5, 1.0],
        repeats=1,
        n_estimators=10,
        n_jobs=1,
    )

    assert len(result) == 6
    additive = result[result["protocol"] == "additive"].set_index("synthetic_fraction")
    replacement = result[result["protocol"] == "replacement"].set_index(
        "synthetic_fraction"
    )
    assert additive.loc[1.0, "training_rows"] == 200
    assert additive.loc[1.0, "real_training_rows"] == 100
    assert additive.loc[1.0, "synthetic_share_of_training"] == 0.5
    assert replacement.loc[0.5, "training_rows"] == 100
    assert replacement.loc[0.5, "real_training_rows"] == 50
    assert replacement.loc[0.5, "synthetic_training_rows"] == 50
    assert replacement.loc[1.0, "real_training_rows"] == 0
    assert replacement.loc[1.0, "synthetic_share_of_training"] == 1.0
    assert np.isfinite(result.select_dtypes(include=[np.number]).to_numpy()).all()

    summary = summarize_mixed_utility(result)
    assert len(summary) == 6
    assert {"pr_auc_mean", "balanced_accuracy_mean", "f1_macro_mean"}.issubset(summary)
    assert not any(column.startswith("tuned_") for column in result)
