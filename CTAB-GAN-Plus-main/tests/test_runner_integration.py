import json
from types import SimpleNamespace

import pandas as pd
import pytest

from xai_reweighting.run_ablation import (
    VALID_VARIANTS,
    _adapter,
    _build_controlled_deltas,
    _fit_and_save_training,
    build_parser,
    run_experiment,
)


class FakeGenerator:
    def fit(self, df):
        self.df = df.copy(deep=True).reset_index(drop=True)

    def sample(self, n):
        return self.df.sample(n=n, replace=True, random_state=42).reset_index(drop=True)

    def save_checkpoint(self, path):
        path.write_text("fake checkpoint", encoding="utf-8")


def test_progress_cli_modes():
    parser = build_parser()
    assert parser.parse_args(["--config", "config.json"]).progress == "auto"
    assert parser.parse_args(
        ["--config", "config.json", "--progress", "off"]
    ).progress == "off"


@pytest.mark.parametrize("generator_name", ["ctabgan_plus", "ctgan", "dp_cgan"])
def test_ablation_adapter_uses_generator_registry(generator_name, tmp_path, monkeypatch):
    captured = {}

    def fake_create(name, generator_config, **kwargs):
        captured.update(name=name, generator_config=generator_config, **kwargs)
        return object()

    monkeypatch.setattr("xai_reweighting.generator_adapters.create_generator", fake_create)
    result = _adapter(
        {"generator_name": generator_name, "generator": {"epochs": 2}},
        "cpu",
        42,
        "off",
        "test training",
        tmp_path / "backend",
    )

    assert result is not None
    assert captured["name"] == generator_name
    assert captured["generator_config"] == {"epochs": 2}
    assert captured["work_dir"] == tmp_path / "backend"


def test_dp_weighted_fit_saves_checkpoint_and_variant_privacy(tmp_path):
    model = FakeGenerator()
    training = pd.DataFrame({"x": range(20), "target": [0, 1] * 10})
    diagnostics = _fit_and_save_training(
        model,
        training,
        "A5",
        tmp_path,
        generator_name="dp_cgan",
        generator_config={
            "batch_size": 10,
            "epochs": 2,
            "discriminator_steps": 10,
        },
    )

    assert diagnostics["checkpoint_saved"] is True
    assert (tmp_path / "model_checkpoint_A5.pkl").exists()
    privacy = json.loads((tmp_path / "privacy_accounting_A5.json").read_text())
    assert privacy["private"] is True
    assert privacy["variant"] == "A5"
    assert "not covered" in privacy["pipeline_privacy_scope"]


def test_controlled_deltas_use_a0_and_real_utility_references():
    metrics = {
        "A0": {"utility_mortality_roc_auc": 0.60, "detector_auc": 0.90},
        "A2": {"utility_mortality_roc_auc": 0.70, "detector_auc": 0.80},
        "A5": {"utility_mortality_roc_auc": 0.75, "detector_auc": 0.70},
    }
    real_only = pd.DataFrame(
        {
            "utility_task": ["mortality", "mortality"],
            "roc_auc": [0.80, 0.82],
            "repeat": [0, 1],
            "seed": [42, 1042],
        }
    )
    summary, deltas = _build_controlled_deltas(
        pd.DataFrame([{"variant": key, **value} for key, value in metrics.items()]),
        metrics,
        real_only,
        [{"name": "mortality"}],
    )

    a5 = summary.set_index("variant").loc["A5"]
    assert a5["delta_utility_mortality_roc_auc_vs_A0"] == pytest.approx(0.15)
    assert a5["real_baseline_utility_mortality_roc_auc"] == pytest.approx(0.81)
    assert a5["delta_utility_mortality_roc_auc_vs_real"] == pytest.approx(-0.06)
    assert set(deltas["reference_type"]) == {
        "synthetic_baseline",
        "real_data_utility_baseline",
    }
    assert "A5-A2" not in set(deltas["comparison"])


def test_all_five_variants_end_to_end_with_fake_generator(tmp_path, monkeypatch):
    project = tmp_path / "project"
    data_dir = project / "data"
    data_dir.mkdir(parents=True)
    data = pd.DataFrame(
        {
            "continuous": list(range(100)),
            "category": ["a", "b"] * 50,
            "target": [0] * 80 + [1] * 20,
        }
    )
    data.to_csv(data_dir / "input.csv", index=False)

    def fake_detector(real, synthetic, categorical_cols, **kwargs):
        return SimpleNamespace(
            metrics={"detector_auc": 0.5},
            shap_importance=pd.Series(
                {"continuous": 1.0, "category": 0.5, "target": 0.25}
            ),
        )

    def fake_evaluation(*args, **kwargs):
        details = pd.DataFrame({"feature": ["continuous"], "metric": [1.0]})
        return {"score": 1.0}, details

    monkeypatch.setattr("xai_reweighting.run_ablation.train_detector", fake_detector)
    monkeypatch.setattr("xai_reweighting.run_ablation.evaluate_variant", fake_evaluation)
    config = {
        "data_path": "data/input.csv",
        "target_col": "target",
        "categorical_cols": ["category", "target"],
        "continuous_cols": ["continuous"],
        "generator": {},
        "seed": 42,
        "frozen": False,
        "weighting": {"alpha": 1.0, "gamma": 0.25, "top_k": 2, "w_max": 2.0},
        "mixed_utility": {
            "enabled": True,
            "additive_fractions": [0.0, 1.0],
            "replacement_fractions": [0.0, 1.0],
            "repeats": 1,
            "n_estimators": 5,
        },
    }
    output = run_experiment(
        config,
        project,
        "val",
        "cpu",
        VALID_VARIANTS,
        output_override=project / "results" / "integration",
        adapter_factory=FakeGenerator,
    )
    summary = pd.read_csv(output / "ablation_summary.csv")
    assert summary["variant"].tolist() == list(VALID_VARIANTS)
    deltas = pd.read_csv(output / "ablation_deltas.csv")
    assert set(deltas["control"]) == {"A0", "REAL"}
    assert "comparison_vs_A0" in summary
    assert "comparison_vs_real" in summary
    assert all((output / f"metrics_{variant}.json").exists() for variant in VALID_VARIANTS)
    assert all((output / f"model_checkpoint_{variant}.pt").exists() for variant in VALID_VARIANTS)
    assert (output / "utility_real_only_baseline.csv").exists()
    assert (output / "utility_mixture_results.csv").exists()
    assert (output / "utility_mixture_summary.csv").exists()
    mixture = pd.read_csv(output / "utility_mixture_results.csv")
    assert set(mixture["variant"]) == set(VALID_VARIANTS)
    assert set(mixture["protocol"]) == {"additive", "replacement"}
