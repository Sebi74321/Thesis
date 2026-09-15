import sys
import types

import numpy as np
import pandas as pd
import pytest
import torch

from model.pipeline.data_preparation import DataPrep
from xai_reweighting.generator_adapters import (
    CTABGANPlusAdapter,
    CTGANAdapter,
    DPCGANAdapter,
    _apply_numeric_constraints,
    _dequantize_integer_features,
    _decimal_places,
    _infer_numeric_constraints,
    create_generator,
)


class FakeCTGAN:
    last_kwargs = None
    fitted = None

    def __init__(self, **kwargs):
        FakeCTGAN.last_kwargs = kwargs
        self.loss_values = pd.DataFrame({"Epoch": [1], "Generator Loss": [0.1]})

    def fit(self, frame, discrete_columns):
        FakeCTGAN.fitted = frame.copy(deep=True)
        self.frame = frame.copy(deep=True)
        self.discrete_columns = list(discrete_columns)

    def set_device(self, device):
        self.device = device

    def sample(self, n):
        return self.frame.sample(n=n, replace=True, random_state=1).reset_index(drop=True)

    def save(self, path):
        with open(path, "wb") as handle:
            handle.write(b"ctgan")


class FakeDPCGAN:
    last_kwargs = None
    fitted = None

    def __init__(self, **kwargs):
        FakeDPCGAN.last_kwargs = kwargs

    def fit(self, frame):
        FakeDPCGAN.fitted = frame.copy(deep=True)
        self.frame = frame.copy(deep=True)
        print("differential privacy with eps = 7.25 and delta = 2e-06.")

    def sample(self, n):
        return self.frame.sample(n=n, replace=True, random_state=2).reset_index(drop=True)

    def save(self, path):
        with open(path, "wb") as handle:
            handle.write(b"dp")


@pytest.fixture
def fake_backends(monkeypatch):
    ctgan = types.ModuleType("ctgan")
    ctgan.CTGAN = FakeCTGAN
    dp = types.ModuleType("dp_cgans")
    dp.DP_CGAN = FakeDPCGAN
    monkeypatch.setitem(sys.modules, "ctgan", ctgan)
    monkeypatch.setitem(sys.modules, "dp_cgans", dp)


def test_ctgan_adapter_preserves_input_schema_and_device(fake_backends):
    frame = pd.DataFrame({"value": [1.25, 2.5], "category": [1, 2], "target": [0, 1]})
    original = frame.copy(deep=True)
    adapter = CTGANAdapter(
        categorical_columns=["category", "target"], batch_size=10, pac=10, epochs=1,
        device="cpu", progress="off",
    )
    adapter.fit(frame)
    generated = adapter.sample(5)

    pd.testing.assert_frame_equal(frame, original)
    assert list(generated.columns) == list(frame.columns)
    assert generated.dtypes.to_dict() == frame.dtypes.to_dict()
    assert len(generated) == 5
    assert FakeCTGAN.last_kwargs["enable_gpu"] is False


def test_ctab_adapter_dequantizes_before_synthesizer_fit(monkeypatch):
    class FakeSynthesizer:
        def __init__(self, **kwargs):
            self.training_history = []
            self.mixture_diagnostics = []

        def fit(self, **kwargs):
            self.fitted = kwargs["train_data"].copy(deep=True)
            return []

    import xai_reweighting.generator_adapters as adapters_module

    monkeypatch.setattr(adapters_module, "CTABGANSynthesizer", FakeSynthesizer)
    frame = pd.DataFrame(
        {"spo2_max": [90.0, 95.0, 100.0], "target": [0, 1, 0]}
    )
    original = frame.copy(deep=True)
    adapter = CTABGANPlusAdapter(
        categorical_columns=["target"],
        mixed_columns={"spo2_max": [100.0]},
        general_columns=["spo2_max"],
        batch_size=2,
        epochs=1,
        device="cpu",
        progress="off",
        dequantize_integer_features=True,
    )

    adapter.fit(frame)

    pd.testing.assert_frame_equal(frame, original)
    assert adapter.data_prep.df.loc[0, "spo2_max"] != 90.0
    assert adapter.data_prep.df.loc[1, "spo2_max"] != 95.0
    assert adapter.data_prep.df.loc[2, "spo2_max"] == 100.0


def test_ctgan_adapter_dequantizes_continuous_integer_grid_before_fit(fake_backends):
    frame = pd.DataFrame(
        {"spo2_max": [90.0, 95.0, 100.0], "target": [0, 1, 0]}
    )
    original = frame.copy(deep=True)
    adapter = CTGANAdapter(
        categorical_columns=["target"],
        batch_size=10,
        pac=10,
        epochs=1,
        device="cpu",
        progress="off",
        dequantize_integer_features=True,
        dequantization_modal_values={"spo2_max": [100.0]},
    )

    adapter.fit(frame)

    pd.testing.assert_frame_equal(frame, original)
    assert FakeCTGAN.fitted.loc[0, "spo2_max"] != 90.0
    assert FakeCTGAN.fitted.loc[1, "spo2_max"] != 95.0
    assert FakeCTGAN.fitted.loc[2, "spo2_max"] == 100.0
    assert adapter.dequantization_diagnostics["columns"]["spo2_max"][
        "preserved_modal_rows"
    ] == 1


def test_dp_adapter_forces_non_private_baseline_and_restores_schema(
    fake_backends, tmp_path
):
    frame = pd.DataFrame({"value": [1.25, 2.5], "category": [1, 2], "target": [0, 1]})
    adapter = DPCGANAdapter(
        categorical_columns=["category", "target"], batch_size=10, pac=10, epochs=1,
        private=False, device="cpu", progress="off", work_dir=tmp_path / "backend",
    )
    adapter.fit(frame)
    generated = adapter.sample(4)

    assert FakeDPCGAN.last_kwargs["private"] is False
    assert FakeDPCGAN.fitted["category"].dtype == object
    assert generated.dtypes.to_dict() == frame.dtypes.to_dict()
    assert adapter.differential_privacy_enabled is False
    assert adapter.backend_mode == "non_private_baseline"
    assert len(generated) == 4


def test_dp_adapter_dequantizes_before_transformer_fit(fake_backends, tmp_path):
    frame = pd.DataFrame(
        {"spo2_max": [90.0, 95.0, 100.0], "target": [0, 1, 0]}
    )
    adapter = DPCGANAdapter(
        categorical_columns=["target"],
        batch_size=10,
        pac=10,
        epochs=1,
        private=False,
        device="cpu",
        progress="off",
        work_dir=tmp_path / "dequantized_backend",
        dequantize_integer_features=True,
        dequantization_modal_values={"spo2_max": [100.0]},
    )

    adapter.fit(frame)

    assert FakeDPCGAN.fitted.loc[0, "spo2_max"] != 90.0
    assert FakeDPCGAN.fitted.loc[1, "spo2_max"] != 95.0
    assert FakeDPCGAN.fitted.loc[2, "spo2_max"] == 100.0


def test_dp_adapter_rejects_private_mode(tmp_path):
    with pytest.raises(ValueError, match="private=false"):
        DPCGANAdapter(
            categorical_columns=["target"], private=True, batch_size=10, pac=10,
            work_dir=tmp_path,
        )


def test_dp_adapter_requires_and_forwards_existing_saved_transformer(
    fake_backends, tmp_path
):
    transformer = tmp_path / "shared" / "fitted_transformer.pkl"
    transformer.parent.mkdir(parents=True)
    transformer.write_bytes(b"fitted transformer")

    adapter = DPCGANAdapter(
        categorical_columns=["target"],
        private=False,
        batch_size=10,
        pac=10,
        epochs=1,
        saved_transformer=transformer,
        work_dir=tmp_path / "variant",
        progress="off",
    )

    assert adapter.transformer_reused is True
    assert adapter.saved_transformer_path == transformer.resolve()
    adapter.fit(pd.DataFrame({"x": range(10), "target": [0, 1] * 5}))
    assert FakeDPCGAN.last_kwargs["saved_transformer"] == str(transformer.resolve())

    with pytest.raises(FileNotFoundError, match="transformer does not exist"):
        DPCGANAdapter(
            categorical_columns=["target"],
            private=False,
            batch_size=10,
            pac=10,
            saved_transformer=tmp_path / "missing.pkl",
        )


def test_registry_validates_model_name():
    with pytest.raises(ValueError, match="Unknown generator"):
        create_generator("unknown", {}, device=torch.device("cpu"), seed=42)


def test_measurement_precision_inference_distinguishes_grids_from_continuous_values():
    one_decimal = pd.Series([0.1, 0.2, 1.5, 7.9] * 30, dtype=float)
    integer_float = pd.Series([97.0, 98.0, 99.0, 100.0] * 30, dtype=float)
    continuous = pd.Series(np.linspace(0.1234567, 0.9876543, 120), dtype=float)

    assert _decimal_places(one_decimal) == 1
    assert _decimal_places(integer_float) == 0
    assert _decimal_places(continuous) > 1


def test_training_dequantization_is_bounded_deterministic_and_preserves_modal_values():
    fitted = pd.DataFrame(
        {
            "spo2_max": [90.0, 91.0, 95.0, 100.0, np.nan],
            "continuous": [0.12, 0.34, 0.56, 0.78, 0.91],
            "category": [0, 1, 0, 1, 0],
        }
    )
    constraints = _infer_numeric_constraints(fitted)
    kwargs = {
        "categorical_columns": ["category"],
        "mixed_columns": {"spo2_max": [100.0]},
        "seed": 42,
        "half_width": 0.5,
    }

    first, diagnostics = _dequantize_integer_features(
        fitted, constraints, **kwargs
    )
    second, _ = _dequantize_integer_features(fitted, constraints, **kwargs)

    pd.testing.assert_frame_equal(first, second)
    pd.testing.assert_frame_equal(
        fitted,
        pd.DataFrame(
            {
                "spo2_max": [90.0, 91.0, 95.0, 100.0, np.nan],
                "continuous": [0.12, 0.34, 0.56, 0.78, 0.91],
                "category": [0, 1, 0, 1, 0],
            }
        ),
    )
    assert 90.0 <= first.loc[0, "spo2_max"] <= 90.5
    assert 90.5 <= first.loc[1, "spo2_max"] <= 91.5
    assert 94.5 <= first.loc[2, "spo2_max"] <= 95.5
    assert first.loc[3, "spo2_max"] == 100.0
    assert first["continuous"].equals(fitted["continuous"])
    assert first["category"].equals(fitted["category"])
    feature = diagnostics["columns"]["spo2_max"]
    assert feature["dequantized_rows"] == 3
    assert feature["preserved_modal_rows"] == 1
    assert feature["lower_boundary_rows"] == 1


def test_probe_dequantization_preserves_already_fractional_generator_values():
    frame = pd.DataFrame({"measurement": [90.0, 90.25, 91.0]})
    constraints = {
        "measurement": {
            "integer_valued": True,
            "minimum": 90.0,
            "maximum": 91.0,
        }
    }

    result, diagnostics = _dequantize_integer_features(
        frame,
        constraints,
        categorical_columns=[],
        mixed_columns={},
        seed=7,
    )

    assert result.loc[1, "measurement"] == 90.25
    assert diagnostics["columns"]["measurement"]["non_grid_rows_preserved"] == 1


def test_numeric_postprocessing_rounds_integer_valued_floats_without_clipping():
    fitted = pd.DataFrame({
        "spo2_max": [90.0, 95.0, 100.0],
        "temperature": [36.1, 36.5, 37.2],
    })
    constraints = _infer_numeric_constraints(fitted)
    raw = pd.DataFrame({
        "spo2_max": [89.6, 95.6, 100.6],
        "temperature": [35.96, 36.54, 37.26],
    })

    processed, diagnostics = _apply_numeric_constraints(raw, constraints)

    assert constraints["spo2_max"]["integer_valued"] is True
    assert processed["spo2_max"].tolist() == [90.0, 96.0, 100.6]
    assert processed["temperature"].tolist() == [36.0, 36.5, 37.26]
    assert diagnostics["spo2_max"]["generated_below_min_rows"] == 1
    assert diagnostics["spo2_max"]["generated_above_max_rows"] == 1


def test_rounding_guard_does_not_push_an_in_range_value_past_fitted_support():
    constraints = {
        "measurement": {
            "decimals": 0,
            "integer_valued": True,
            "minimum": 0.25,
            "maximum": 0.75,
            "source_dtype": "float64",
        }
    }
    processed, diagnostics = _apply_numeric_constraints(
        pd.DataFrame({"measurement": [0.3, 0.7]}), constraints
    )

    assert processed["measurement"].tolist() == [0.3, 0.7]
    assert diagnostics["measurement"]["rounding_guarded_rows"] == 2


def test_positive_log_columns_round_trip_through_data_prep():
    frame = pd.DataFrame({"wbc": [0.1, 1.0, 10.0, 100.0]})
    prep = DataPrep(frame.copy(), [], ["wbc"], {}, ["wbc"], [], [], {None: None}, 0.0)
    restored = prep.inverse_prep(prep.df.to_numpy(copy=True))

    np.testing.assert_allclose(restored["wbc"], frame["wbc"], rtol=1e-7, atol=1e-9)
