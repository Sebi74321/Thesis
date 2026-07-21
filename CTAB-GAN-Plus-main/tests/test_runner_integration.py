from types import SimpleNamespace

import pandas as pd

from xai_reweighting.run_ablation import VALID_VARIANTS, run_experiment


class FakeGenerator:
    def fit(self, df):
        self.df = df.copy(deep=True).reset_index(drop=True)

    def sample(self, n):
        return self.df.sample(n=n, replace=True, random_state=42).reset_index(drop=True)


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
    assert (output / "ablation_deltas.csv").exists()
    assert all((output / f"metrics_{variant}.json").exists() for variant in VALID_VARIANTS)
