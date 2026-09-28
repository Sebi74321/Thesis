import json
import ast
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest

from xai_reweighting.run_ablation import VALID_VARIANTS, build_parser, run_experiment
from xai_reweighting.run_shap_contribution import run_study
from xai_reweighting.scoring import compute_feature_priority, weight_diagnostics
from xai_reweighting.shap_contribution_reporting import (
    STUDY_VARIANTS, paired_contrasts, variant_records, validate_child,
)
from xai_reweighting.sensitivity_reporting import collect_run
from xai_reweighting.io_utils import atomic_write_csv


def test_priority_controls_preserve_components_and_formulas():
    components = pd.DataFrame({"feature": list("abcd"), "shap": [.1, .2, .7, 1.],
                               "mismatch": [.9, .1, .4, .2], "tail": [.1, .3, .2, .9]})
    before = components.copy(deep=True)
    no_shap = compute_feature_priority(components, "A5_NO_SHAP", 4).set_index("feature").sort_index()
    np.testing.assert_allclose(no_shap.combined_raw, .6 * components.mismatch + .4 * components["tail"])
    changed = components.assign(shap=[1., .9, .2, .1])
    other = compute_feature_priority(changed, "A5_NO_SHAP", 4).set_index("feature").sort_index()
    np.testing.assert_array_equal(no_shap.priority, other.priority)
    shuffled = compute_feature_priority(components, "A5_SHUFFLED_SHAP", 4, shuffle_seed=123)
    shuffled = shuffled.set_index("feature").sort_index()
    assert sorted(shuffled.shap) == sorted(components.shap)
    np.testing.assert_allclose(shuffled.combined_raw, .5 * shuffled.shap + .3 * shuffled.mismatch + .2 * shuffled["tail"])
    np.testing.assert_allclose(shuffled.mismatch, components.mismatch)
    np.testing.assert_allclose(shuffled["tail"], components["tail"])
    assert shuffled.shap_source_feature.tolist() != list("abcd")
    permuted_input = compute_feature_priority(components.iloc[::-1], "A5_SHUFFLED_SHAP", 4, shuffle_seed=123)
    pd.testing.assert_frame_equal(shuffled, permuted_input.set_index("feature").sort_index())
    assert shuffled.priority.sum() == pytest.approx(1.)
    pd.testing.assert_frame_equal(components, before)


def test_degenerate_shap_does_not_create_artificial_signal_or_tie_changes():
    frame = pd.DataFrame({"feature": ["b", "a", "c"], "shap": [0.] * 3,
                          "mismatch": [0.] * 3, "tail": [0.] * 3})
    for variant in STUDY_VARIANTS[1:]:
        result = compute_feature_priority(frame, variant, 2)
        assert result.priority.sum() == 0
        assert result.loc[result.selected, "feature"].tolist() == ["a", "b"]
    diagnostic = weight_diagnostics(pd.Series([1., 1., 1.]), 3.)
    assert diagnostic["sampling_ess"] == pytest.approx(3)
    assert diagnostic["sampling_ess_fraction"] == pytest.approx(1)
    assert weight_diagnostics(pd.Series([1., 1., 3.]), 3.)["sampling_ess"] < 3


def test_report_pairs_repeats_within_seed_and_keeps_missing_values(tmp_path):
    atomic_write_csv(tmp_path / "utility_mixture_results.csv", pd.DataFrame({
        "variant": ["A5", "A5", "A5_NO_SHAP", "A5_NO_SHAP", "A5"],
        "utility_task": ["mortality"] * 4 + ["mortality_balanced"],
        "protocol": ["additive"] * 5, "synthetic_fraction": [1.] * 5,
        "repeat": [0, 1, 0, 1, 0], "precision": [.6, .8, .3, .5, .9]}))
    raw, _ = collect_run(tmp_path, "study", 42)
    pairs = paired_contrasts(variant_records(raw))
    selected = pairs[pairs.comparison == "A5-A5_NO_SHAP"]
    normal = selected[selected.context.str.contains('"mortality"')]
    assert len(normal) == 1
    assert normal.delta.iloc[0] == pytest.approx(.3)
    assert selected.delta.isna().sum() == 1


def test_defaults_and_design_guards(tmp_path):
    assert build_parser().parse_args(["--config", "x"]).variants == ",".join(VALID_VARIANTS)
    cfg = {"data_path": "data.csv"}
    (tmp_path / "data.csv").write_text("x\n1\n")
    with pytest.raises(ValueError, match="three distinct"):
        run_study(cfg, tmp_path, tmp_path / "study", seeds=[42, 42, 43])
    with pytest.raises(ValueError, match="Test studies"):
        run_study(cfg, tmp_path, tmp_path / "study", stage="test", smoke=True)
    with pytest.raises(ValueError, match="exceeds budget"):
        run_study(cfg, tmp_path, tmp_path / "study", baseline_run_minutes=1000)
    run_study(cfg, tmp_path, tmp_path / "study", dry_run=True)
    assert not (tmp_path / "study").exists()


def test_three_seed_study_full_evaluation_fake_gan_and_resume(tmp_path, monkeypatch):
    data = pd.DataFrame({"x": np.arange(100, dtype=float), "y": np.sin(np.arange(100)),
                         "target": [0] * 80 + [1] * 20})
    data.to_csv(tmp_path / "data.csv", index=False)
    detector_calls = []
    fits = []

    def detector(real, synthetic, categorical_cols, **kwargs):
        if kwargs.get("compute_shap", True):
            detector_calls.append(real.copy())
        scores = pd.Series({"x": 1., "y": .3, "target": .2})
        return SimpleNamespace(metrics={"detector_auc": .5}, shap_signed=scores, shap_importance=scores)

    monkeypatch.setattr("xai_reweighting.run_ablation.train_detector", detector)

    class FakeGenerator:
        def fit(self, df):
            self.df = df.copy(deep=True)
            fits.append(self.df)

        def sample(self, n):
            return self.df.sample(n, replace=True, random_state=42).reset_index(drop=True)

        def save_checkpoint(self, path):
            path.write_text("test checkpoint")

    cfg = {"data_path": "data.csv", "target_col": "target", "categorical_cols": ["target"],
           "continuous_cols": ["x", "y"], "generator": {"categorical_columns": ["target"]}, "n_jobs": 1,
           "baseline_diagnostics": {"enabled": False}, "priority_diagnostics": {"enabled": False},
           "feature_exclusion_sensitivity": {"enabled": False},
           "weighting": {"alpha": 3., "gamma": .6, "w_max": 3., "top_k": 2},
           "evaluation": {"n_estimators": 5},
           "utility_tasks": [
               {"name": "mortality", "target_col": "target", "positive_label": "1", "balance": "imbalanced"},
               {"name": "mortality_balanced", "target_col": "target", "positive_label": "1", "balance": "balanced"}],
           "mixed_utility": {"enabled": True, "repeats": 2, "n_estimators": 5,
                             "additive_fractions": [0., 1.], "replacement_fractions": [0., 1.]}}
    calls = []

    def experiment(*args, **kwargs):
        calls.append(args[0]["seed"])
        if len(calls) == 2:
            raise RuntimeError("simulated disconnection between seeds")
        return run_experiment(*args, **kwargs, adapter_factory=FakeGenerator)

    output = tmp_path / "study"
    with pytest.raises(RuntimeError, match="simulated disconnection"):
        run_study(cfg, tmp_path, output, device="cpu", progress="off", experiment_runner=experiment)
    assert len(fits) == 4
    protected = (output / "seed42" / "synthetic_A5.csv").read_bytes()
    run_study(cfg, tmp_path, output, device="cpu", progress="off", resume=True, experiment_runner=experiment)
    assert len(fits) == 12
    assert len(detector_calls) == 3  # Exactly one shared SHAP audit per seed.
    assert (output / "seed42" / "synthetic_A5.csv").read_bytes() == protected
    split = json.loads((output / "seed42" / "split_indices.json").read_text())
    train = data.iloc[split["train"]].reset_index(drop=True)
    for index, frame in enumerate(fits):
        assert len(frame) == (60 if index % 4 == 0 else 96)
        pd.testing.assert_frame_equal(frame.iloc[:60].reset_index(drop=True), train)
        assert set(frame.x) <= set(train.x)
    for seed in (42, 43, 44):
        child = output / f"seed{seed}"
        assert json.loads((child / "split_indices.json").read_text()) == split
        summary = pd.read_csv(child / "ablation_summary.csv")
        assert summary.variant.tolist() == list(STUDY_VARIANTS)
        assert np.isfinite(summary.mean_wasserstein_scaled).all()
        mixture = pd.read_csv(child / "utility_mixture_results.csv")
        assert set(mixture.variant) == set(STUDY_VARIANTS)
        assert set(mixture.protocol) == {"additive", "replacement"}
        for variant in STUDY_VARIANTS[1:]:
            weights = pd.read_csv(child / f"row_weights_{variant}.csv").weight
            assert weights.between(1, 3).all()
            counts = json.loads((child / f"augmentation_counts_{variant}.json").read_text())
            assert sum(counts.values()) == 36
        assert (child / "feature_scores_A5_SHUFFLED_SHAP.csv").is_file()
    report = pd.read_csv(output / "study_primary_summary.csv")
    assert set(report.n) == {3}
    assert (output / "study_primary_pairs.png").exists()
    assert (output / "study_weight_diagnostics.csv").exists()
    mechanisms = pd.read_csv(output / "study_mechanism_diagnostics.csv")
    assert len(mechanisms) == 9
    assert (mechanisms.loc[mechanisms.variant == "A5", "augmentation_probability_tv_to_A5"] == 0).all()
    run_study(cfg, tmp_path, output, device="cpu", progress="off", resume=True, experiment_runner=experiment)
    assert len(fits) == 12
    # A valid fingerprint for a different config is not enough to join the study.
    from xai_reweighting.run_ablation import _fingerprint
    child = output / "seed42"
    original_config = (child / "config.json").read_text()
    original_manifest = (child / "manifest.json").read_text()
    foreign_config = json.loads(original_config)
    foreign_config["generator"]["epochs"] = 999
    plan = json.loads((output / "study_plan.json").read_text())
    foreign_manifest = json.loads(original_manifest)
    foreign_manifest["fingerprint"] = _fingerprint(foreign_config, plan["data_sha256"], plan["code_sha256"])
    (child / "config.json").write_text(json.dumps(foreign_config))
    (child / "manifest.json").write_text(json.dumps(foreign_manifest))
    with pytest.raises(ValueError, match="incompatible provenance"):
        validate_child(child, plan, 42)
    (child / "config.json").write_text(original_config)
    (child / "manifest.json").write_text(original_manifest)
    with pytest.raises(ValueError, match="fingerprint mismatch"):
        run_study(cfg, tmp_path, output, device="cpu", resume=True, primary_direction="higher")
    (tmp_path / "data.csv").write_text("changed source data")
    with pytest.raises(ValueError, match="fingerprint mismatch"):
        run_study(cfg, tmp_path, output, device="cpu", resume=True)


def test_notebook_is_valid_and_does_not_launch_training_by_default():
    notebook = Path(__file__).resolve().parents[1] / "notebooks/shap_contribution_study.ipynb"
    book = json.loads(notebook.read_text(encoding="utf-8"))
    assert book["nbformat"] == 4
    code_cells = ["".join(cell["source"]) for cell in book["cells"] if cell["cell_type"] == "code"]
    for source in code_cells:
        ast.parse(source)
    assert any("LAUNCH = False" in source for source in code_cells)
    assert any("REFRESH_REPORTS = False" in source for source in code_cells)
