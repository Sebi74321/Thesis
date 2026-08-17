from types import SimpleNamespace

import pandas as pd

from xai_reweighting.run_model_comparison import build_parser, run_model_comparison


class FakeGenerator:
    fits = []

    def __init__(self, name, seed):
        self.name = name
        self.seed = seed
        self.training_history = pd.DataFrame({"epoch": [1], "generator": [0.1]})
        self.convergence_warnings = []

    def fit(self, frame):
        self.frame = frame.copy(deep=True).reset_index(drop=True)
        FakeGenerator.fits.append((self.name, self.seed, self.frame.copy()))

    def sample(self, n):
        return self.frame.sample(n=n, replace=True, random_state=self.seed).reset_index(drop=True)


def test_rq1_parser_defaults():
    args = build_parser().parse_args(["--config", "config.json"])
    assert args.models == "ctabgan_plus,ctgan,dp_cgan"
    assert args.seeds == "42,43,44"


def test_three_models_share_split_and_resume(tmp_path, monkeypatch):
    project = tmp_path / "project"
    (project / "data").mkdir(parents=True)
    data = pd.DataFrame(
        {"x": range(100), "category": ["a", "b"] * 50, "target": [0] * 80 + [1] * 20}
    )
    data.to_csv(project / "data" / "input.csv", index=False)
    config = {
        "dataset_name": "test",
        "data_path": "data/input.csv",
        "target_col": "target",
        "categorical_cols": ["category", "target"],
        "continuous_cols": ["x"],
        "models": {name: {"epochs": 1, "batch_size": 10} for name in ("ctabgan_plus", "ctgan", "dp_cgan")},
        "split_seed": 42,
        "frozen": False,
        "mixed_utility": {"enabled": False},
        "evaluation": {"n_estimators": 5},
        "n_jobs": 1,
    }

    def fake_adapter(name, model_config, device, seed, run_dir):
        return FakeGenerator(name, seed)

    def fake_evaluate(*args, **kwargs):
        return {"utility_roc_auc": 0.5, "detector_auc": 0.5}, pd.DataFrame({"feature": ["x"]})

    def fake_specific(*args, **kwargs):
        return {"rq1_joint_rare_region_mass_error": 0.0}, pd.DataFrame({"feature": ["x"]})

    monkeypatch.setattr("xai_reweighting.run_model_comparison.evaluate_variant", fake_evaluate)
    monkeypatch.setattr("xai_reweighting.run_model_comparison.evaluate_rq1_specific", fake_specific)
    monkeypatch.setattr(
        "xai_reweighting.run_model_comparison.real_real_reference",
        lambda *args, **kwargs: ({"detector_auc": 0.5}, pd.DataFrame()),
    )
    output = project / "results" / "rq1"
    run_model_comparison(
        config, project, "val", "cpu", ["ctabgan_plus", "ctgan", "dp_cgan"], [42],
        output_override=output, adapter_factory=fake_adapter,
    )
    results = pd.read_csv(output / "rq1_results.csv")
    assert set(results["model"]) == {"ctabgan_plus", "ctgan", "dp_cgan"}
    assert all(len(fit[2]) == 60 for fit in FakeGenerator.fits)
    assert all((output / "models" / name / "seed_42" / ".complete.json").exists() for name in results["model"])

    def fail_adapter(*args, **kwargs):
        raise AssertionError("completed models must not be fitted during resume")

    run_model_comparison(
        config, project, "val", "cpu", ["ctabgan_plus", "ctgan", "dp_cgan"], [42],
        output_override=output, resume=True, adapter_factory=fail_adapter,
    )

