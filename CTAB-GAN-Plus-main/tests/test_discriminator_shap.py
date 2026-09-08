from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest
import torch

from model.synthesizer.ctabgan_synthesizer import Discriminator, determine_layers_disc
from model.synthesizer.transformer import ImageTransformer
from xai_reweighting.discriminator_shap import (
    DiscriminatorTabularWrapper,
    _encode_probe,
    _snapshot_metric_row,
    aggregate_encoded_shap,
    encoded_feature_slices,
    evaluate_discriminator_snapshots,
)


def test_snapshot_metrics_distinguish_one_class_collapse_from_convergence():
    truth = np.array([1, 1, 0, 0])
    scores = np.zeros(4)
    predictions = np.zeros(4, dtype=int)

    row, _ = _snapshot_metric_row(
        epoch=25,
        truth=truth,
        scores=scores,
        predictions=predictions,
        threshold=1.0,
        calibration_rows=4,
        bootstrap_repeats=10,
        seed=42,
    )

    assert row["accuracy"] == 0.5
    assert row["balanced_accuracy"] == 0.5
    assert row["orientation_free_separability"] == 0.0
    assert row["real_recall"] == 0.0
    assert row["synthetic_recall"] == 1.0
    assert row["predicted_real_fraction"] == 0.0


class IdentityTabularTransformer:
    output_info = [[(1, "tanh")], [(1, "tanh")]]
    output_dim = 2

    def __init__(self):
        self.ordering = []

    def transform(self, values):
        np.random.random()
        self.ordering.extend([None, None])
        return np.asarray(values, dtype=np.float32)


def test_discriminator_wrapper_marginalizes_valid_conditions():
    class SumDiscriminator(torch.nn.Module):
        def forward(self, values):
            scores = values.reshape(values.shape[0], -1).sum(dim=1, keepdim=True)
            return scores, values

    class IdentityImageTransformer:
        @staticmethod
        def transform(values):
            return values

    wrapper = DiscriminatorTabularWrapper(
        SumDiscriminator(),
        IdentityImageTransformer(),
        conditions=torch.tensor([[1.0, 0.0], [0.0, 1.0]]),
    )

    result = wrapper(torch.tensor([[2.0, 3.0], [4.0, 5.0]]))

    assert result.shape == (2, 1)
    assert torch.allclose(result[:, 0], torch.tensor([6.0, 10.0]))


def test_probe_encoding_rebuilds_and_restores_mixed_column_masks():
    class MixedMaskTransformer:
        def __init__(self):
            self.meta = [{"type": "mixed", "modal": [100.0]}]
            self.filter_arr = [np.array([True, False, True])]
            self.ordering = []

        def transform(self, values):
            assert len(self.filter_arr[0]) == len(values)
            assert self.filter_arr[0].tolist() == [True, False, True, False]
            self.ordering.append(np.array([0]))
            return np.asarray(values, dtype=np.float32)

    transformer = MixedMaskTransformer()
    original_filter = transformer.filter_arr
    adapter = SimpleNamespace(
        synthesizer=SimpleNamespace(transformer=transformer),
        data_prep=SimpleNamespace(
            df=pd.DataFrame({"spo2_max": [98.0, 100.0, 99.0]}),
            categorical_columns=[],
            log_columns=[],
            lower_bounds={},
            label_encoder_list=[],
        ),
    )

    encoded = _encode_probe(
        adapter,
        pd.DataFrame({"spo2_max": [98.0, 100.0, 97.0, 100.0]}),
        seed=42,
    )

    assert encoded.shape == (4, 1)
    assert transformer.filter_arr is original_filter
    assert transformer.ordering == []


def test_probe_encoding_reuses_fitted_component_ordering():
    class ReorderingTransformer:
        output_info = [(1, "tanh"), (3, "softmax")]
        output_dim = 4
        meta = [{"type": "continuous"}]
        general_columns = []
        filter_arr = []

        def __init__(self):
            # The discriminator saw original component coordinates [2, 0, 1]
            # during fitting.  The probe's own frequency order differs.
            self.ordering = [np.array([2, 0, 1])]

        def transform(self, values):
            probe_order = np.array([1, 2, 0])
            self.ordering.append(probe_order)
            original_onehot = np.eye(3, dtype=np.float32)[[0, 1]]
            probe_onehot = original_onehot[:, probe_order]
            return np.column_stack(
                [np.asarray(values[:, 0], dtype=np.float32), probe_onehot]
            )

    transformer = ReorderingTransformer()
    adapter = SimpleNamespace(
        synthesizer=SimpleNamespace(transformer=transformer),
        data_prep=SimpleNamespace(
            df=pd.DataFrame({"measurement": [1.0, 2.0]}),
            categorical_columns=[],
            log_columns=[],
            lower_bounds={},
            label_encoder_list=[],
        ),
    )

    encoded = _encode_probe(
        adapter,
        pd.DataFrame({"measurement": [1.0, 2.0]}),
        seed=42,
    )

    expected_original = np.eye(3, dtype=np.float32)[[0, 1]]
    assert np.array_equal(encoded[:, 1:], expected_original[:, [2, 0, 1]])
    assert len(transformer.ordering) == 1
    assert np.array_equal(transformer.ordering[0], np.array([2, 0, 1]))


def test_encoded_shap_is_aggregated_to_original_features():
    transformer = SimpleNamespace(
        output_info=[[(2, "tanh")], [(1, "softmax")]], output_dim=3
    )
    slices = encoded_feature_slices(transformer, ["continuous", "category"])
    values = np.array([[1.0, -2.0, 3.0], [-1.0, 2.0, -3.0]])

    result = aggregate_encoded_shap(values, slices).set_index("feature")

    assert result.loc["continuous", "mean_abs_shap"] == pytest.approx(1.0)
    assert result.loc["continuous", "mean_signed_shap"] == pytest.approx(0.0)
    assert result.loc["continuous", "encoded_dimensions"] == 2
    assert result.loc["category", "mean_abs_shap"] == pytest.approx(3.0)


def test_native_ctab_output_spans_are_grouped_by_original_feature():
    """A mixed CTAB+ column emits two output_info entries, not one."""
    transformer = SimpleNamespace(
        meta=[
            {"type": "continuous"},
            {"type": "mixed"},
            {"type": "categorical"},
        ],
        general_columns=[0],
        output_info=[
            (1, "tanh", "yes_g"),
            (1, "tanh", "no_g"),
            (3, "softmax"),
            (2, "softmax"),
        ],
        output_dim=7,
    )

    slices = encoded_feature_slices(
        transformer, ["general", "spo2_max", "outcome"]
    )

    assert slices == {
        "general": slice(0, 1),
        "spo2_max": slice(1, 5),
        "outcome": slice(5, 7),
    }


def test_native_ctab_31_features_can_have_32_output_spans():
    feature_names = [f"feature_{index}" for index in range(30)] + ["spo2_max"]
    transformer = SimpleNamespace(
        meta=[{"type": "categorical"} for _ in range(30)] + [{"type": "mixed"}],
        general_columns=[],
        output_info=[(1, "softmax") for _ in range(30)]
        + [(1, "tanh", "no_g"), (4, "softmax")],
        output_dim=35,
    )

    slices = encoded_feature_slices(transformer, feature_names)

    assert len(slices) == 31
    assert slices["spo2_max"] == slice(30, 35)


def test_snapshot_evaluation_reuses_rows_and_returns_tidy_trajectory(monkeypatch):
    class FakeGradientExplainer:
        backgrounds = []

        def __init__(self, model, background):
            self.model = model
            self.__class__.backgrounds.append(background.detach().cpu().numpy())

        def shap_values(self, rows):
            np.random.random()
            return rows.detach().cpu().numpy()[:, :, None]

    import shap
    import xai_reweighting.discriminator_shap as discriminator_shap_module

    monkeypatch.setattr(shap, "GradientExplainer", FakeGradientExplainer)
    monkeypatch.setattr(
        discriminator_shap_module,
        "_critic_scores",
        lambda wrapper, encoded, device: encoded[:, 0].astype(float),
    )
    monkeypatch.setattr(
        discriminator_shap_module,
        "_generate_snapshot_encoded",
        lambda snapshot, synthesizer, noise, conditions, device, seed: np.column_stack(
            [-np.full(len(noise), 10.0), np.resize([0.0, 1.0], len(noise))]
        ).astype(np.float32),
    )
    discriminator = Discriminator(4, determine_layers_disc(4, 2))
    state = {
        key: value.detach().cpu().clone()
        for key, value in discriminator.state_dict().items()
    }
    transformer = IdentityTabularTransformer()
    synthesizer = SimpleNamespace(
        dside=4,
        num_channels=2,
        transformer=transformer,
        Dtransformer=ImageTransformer(4),
        cond_generator=SimpleNamespace(n_opt=0),
    )
    adapter = SimpleNamespace(
        synthesizer=synthesizer,
        data_prep=SimpleNamespace(
            df=pd.DataFrame({"feature": [1.0, 2.0, 3.0], "target": [0.0, 1.0, 0.0]}),
            categorical_columns=[],
            log_columns=[],
            lower_bounds={},
            label_encoder_list=[],
        ),
        discriminator_snapshots=[
            {"epoch": 25, "state_dict": state, "generator_state_dict": {}},
            {"epoch": 50, "state_dict": state, "generator_state_dict": {}},
        ],
        device="cpu",
    )

    np.random.seed(99)
    expected_next_random = np.random.random()
    np.random.seed(99)
    evaluation = evaluate_discriminator_snapshots(
        adapter,
        pd.DataFrame(
            {"feature": [10.0] * 4, "target": [0.0, 1.0, 0.0, 1.0]}
        ),
        pd.DataFrame(
            {"feature": [-10.0] * 4, "target": [0.0, 1.0, 0.0, 1.0]}
        ),
        background_size=2,
        explain_size=2,
        test_size=0.5,
        seed=7,
        exclude_features=["target"],
    )
    result = evaluation.trajectory

    assert result["epoch"].tolist() == [25, 25, 50, 50]
    assert result["outcome_group"].tolist() == [
        "correct_real",
        "correct_synthetic",
        "correct_real",
        "correct_synthetic",
    ]
    assert result["feature"].tolist() == ["feature"] * 4
    assert result["rank"].tolist() == [1] * 4
    assert result["background_rows"].tolist() == [2] * 4
    assert result["primary_scope"].tolist() == [False, True, False, True]
    assert set(evaluation.predictions["outcome_group"]) == {
        "correct_real",
        "correct_synthetic",
    }
    assert (evaluation.metrics["balanced_accuracy"] == 1.0).all()
    assert (evaluation.metrics["orientation_free_separability"] == 1.0).all()
    assert (evaluation.metrics["real_recall"] == 1.0).all()
    assert (evaluation.metrics["synthetic_recall"] == 1.0).all()
    assert (evaluation.epoch_matched_metrics["balanced_accuracy"] == 1.0).all()
    assert set(evaluation.late_window_summary["probe_mode"]) == {
        "fixed_final_generator",
        "epoch_matched_generator",
    }
    assert evaluation.shap_stability.iloc[0]["top_k_jaccard"] == 1.0
    assert np.array_equal(
        FakeGradientExplainer.backgrounds[0], FakeGradientExplainer.backgrounds[1]
    )
    assert transformer.ordering == []
    assert np.random.random() == expected_next_random


def test_snapshot_evaluation_requires_snapshot_capture():
    adapter = SimpleNamespace(
        synthesizer=object(), data_prep=object(), discriminator_snapshots=[]
    )

    with pytest.raises(ValueError, match="snapshot_frq"):
        evaluate_discriminator_snapshots(
            adapter,
            pd.DataFrame({"x": range(4)}),
            pd.DataFrame({"x": range(4)}),
        )
