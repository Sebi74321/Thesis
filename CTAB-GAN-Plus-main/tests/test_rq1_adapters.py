import sys
import types

import pandas as pd
import pytest
import torch

from xai_reweighting.generator_adapters import CTGANAdapter, DPCGANAdapter, create_generator


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


def test_dp_adapter_forces_private_and_restores_numeric_categories(fake_backends, tmp_path):
    frame = pd.DataFrame({"value": [1.25, 2.5], "category": [1, 2], "target": [0, 1]})
    adapter = DPCGANAdapter(
        categorical_columns=["category", "target"], batch_size=10, pac=10, epochs=1,
        private=True, device="cpu", progress="off", work_dir=tmp_path / "backend",
    )
    adapter.fit(frame)
    generated = adapter.sample(4)

    assert FakeDPCGAN.last_kwargs["private"] is True
    assert FakeDPCGAN.fitted["category"].dtype == object
    assert generated.dtypes.to_dict() == frame.dtypes.to_dict()
    assert adapter.upstream_reported_epsilon == 7.25
    assert len(generated) == 4


def test_dp_adapter_rejects_non_private_mode(tmp_path):
    with pytest.raises(ValueError, match="private=True"):
        DPCGANAdapter(
            categorical_columns=["target"], private=False, batch_size=10, pac=10,
            work_dir=tmp_path,
        )


def test_registry_validates_model_name():
    with pytest.raises(ValueError, match="Unknown generator"):
        create_generator("unknown", {}, device=torch.device("cpu"), seed=42)
