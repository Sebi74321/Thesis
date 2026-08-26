from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest
import torch

from model.synthesizer.ctabgan_synthesizer import Discriminator, determine_layers_disc
from model.synthesizer.transformer import ImageTransformer
from xai_reweighting.discriminator_shap import (
    aggregate_encoded_shap,
    encoded_feature_slices,
    evaluate_discriminator_snapshots,
)


class IdentityTabularTransformer:
    output_info = [[(1, "tanh")], [(1, "tanh")]]
    output_dim = 2

    def __init__(self):
        self.ordering = []

    def transform(self, values):
        np.random.random()
        self.ordering.extend([None, None])
        return np.asarray(values, dtype=np.float32)


def test_encoded_shap_is_aggregated_to_original_features():
    transformer = SimpleNamespace(
        output_info=[[(2, "tanh")], [(1, "softmax")]], output_dim=3
    )
    slices = encoded_feature_slices(transformer, ["continuous", "category"])
    values = np.array([[1.0, -2.0, 3.0], [-1.0, 2.0, -3.0]])

    result = aggregate_encoded_shap(values, slices).set_index("feature")

    assert result.loc["continuous", "mean_abs_shap"] == pytest.approx(3.0)
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

    monkeypatch.setattr(shap, "GradientExplainer", FakeGradientExplainer)
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
            df=pd.DataFrame({"feature": [1.0, 2.0, 3.0], "target": [0.0, 1.0, 0.0]})
        ),
        discriminator_snapshots=[
            {"epoch": 25, "state_dict": state},
            {"epoch": 50, "state_dict": state},
        ],
        device="cpu",
    )

    np.random.seed(99)
    expected_next_random = np.random.random()
    np.random.seed(99)
    result = evaluate_discriminator_snapshots(
        adapter,
        background_size=2,
        explain_size=3,
        seed=7,
        exclude_features=["target"],
    )

    assert result["epoch"].tolist() == [25, 50]
    assert result["feature"].tolist() == ["feature", "feature"]
    assert result["rank"].tolist() == [1, 1]
    assert result["background_rows"].tolist() == [2, 2]
    assert result["explained_rows"].tolist() == [3, 3]
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
        evaluate_discriminator_snapshots(adapter)
