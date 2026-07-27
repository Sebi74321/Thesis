import warnings

import numpy as np
import pandas as pd
import pytest
from sklearn.exceptions import ConvergenceWarning

from xai_reweighting.run_ablation import _fit_and_save_training, _training_diagnostics


def test_mixture_nonconvergence_is_retried_and_recorded():
    pytest.importorskip("torch")
    from model.synthesizer.transformer import DataTransformer

    rng = np.random.default_rng(42)
    data = pd.DataFrame({"x": np.r_[rng.normal(-2, 0.4, 100), rng.normal(2, 0.6, 100)]})
    transformer = DataTransformer(
        train_data=data,
        mixture_max_iter=1,
        mixture_n_init=1,
        mixture_tol=1e-3,
    )

    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        transformer.fit()

    assert not any(issubclass(item.category, ConvergenceWarning) for item in caught)
    diagnostic = transformer.mixture_diagnostics[0]
    assert diagnostic["converged"] is True
    assert len(diagnostic["attempts"]) == 2
    assert diagnostic["attempts"][0]["converged"] is False
    assert diagnostic["attempts"][1]["converged"] is True


def test_training_diagnostics_summarize_epoch_history():
    history = pd.DataFrame(
        {
            "epoch": np.arange(1, 11),
            "elapsed_seconds": np.arange(10, 110, 10),
            "generator": np.linspace(2.0, 1.0, 10),
            "gradient_penalty": np.linspace(0.5, 0.2, 10),
        }
    )

    diagnostics = _training_diagnostics(history)

    assert diagnostics["status"] == "no_instability_detected"
    assert diagnostics["epochs_completed"] == 10
    assert diagnostics["tail_epochs"] == 2
    assert diagnostics["losses"]["generator"]["final"] == 1.0


def test_failed_fit_preserves_partial_training_diagnostics(tmp_path):
    class FailingGenerator:
        training_history = pd.DataFrame(
            {"epoch": [1], "elapsed_seconds": [1.0], "generator": [2.0]}
        )
        mixture_diagnostics = []
        columns = ["x"]

        def fit(self, data):
            raise FloatingPointError("non-finite test loss")

    with pytest.raises(FloatingPointError):
        _fit_and_save_training(FailingGenerator(), pd.DataFrame({"x": [1]}), "A5", tmp_path)

    assert (tmp_path / "training_history_A5.csv").exists()
    assert (tmp_path / "training_diagnostics_A5.json").exists()
    assert (tmp_path / "mixture_diagnostics_A5.json").exists()
