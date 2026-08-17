"""Optional real-package smoke tests; enable after installing generator dependencies."""

import os

import pandas as pd
import pytest

from xai_reweighting.generator_adapters import CTGANAdapter, DPCGANAdapter


pytestmark = pytest.mark.skipif(
    os.environ.get("RUN_GENERATOR_SMOKE") != "1",
    reason="set RUN_GENERATOR_SMOKE=1 to run real third-party GAN fits",
)


def test_real_ctgan_and_dp_cgan_cpu_smoke(tmp_path):
    frame = pd.DataFrame(
        {
            "x": [float(index % 20) for index in range(100)],
            "category": ["a", "b"] * 50,
            "target": [0] * 80 + [1] * 20,
        }
    )
    ctgan = CTGANAdapter(
        categorical_columns=["category", "target"],
        generator_dim=(16,), discriminator_dim=(16,), batch_size=50, pac=10,
        epochs=1, verbose=False, progress="off", device="cpu",
    )
    ctgan.fit(frame)
    assert len(ctgan.sample(25)) == 25
    ctgan.save_checkpoint(tmp_path / "ctgan.pkl")

    dp_cgan = DPCGANAdapter(
        categorical_columns=["category", "target"],
        generator_dim=(16,), discriminator_dim=(16,), batch_size=50, pac=10,
        epochs=1, discriminator_steps=1, verbose=False, progress="off", device="cpu",
        work_dir=tmp_path / "dp_backend",
    )
    dp_cgan.fit(frame)
    generated = dp_cgan.sample(25)
    assert len(generated) == 25
    assert list(generated.columns) == list(frame.columns)
    assert dp_cgan.upstream_reported_epsilon is not None
    dp_cgan.save_checkpoint(tmp_path / "dp_cgan.pkl")
