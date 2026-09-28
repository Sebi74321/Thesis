"""Thesis plots preserve seed pairing, protocol separation and missing coverage."""
import json
from pathlib import Path
import zipfile

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pytest

from xai_reweighting.shap_contribution_figures import (
    Study, load_study, paired_values, plot_primary, plot_utility_controls,
    plot_fixed_features, export_thesis_figures,
)


def fixture_study(label="CTGAN"):
    rows = []
    for i, seed in enumerate([40, 41, 42]):
        for variant, offset in [("A0", .3), ("A5_NO_SHAP", .2), ("A5_SHUFFLED_SHAP", .1), ("A5", 0.)]:
            def add(metric, value, domain, **context):
                rows.append(dict(seed=seed, variant=variant, value=value, domain=domain,
                                 utility_task="", protocol="", synthetic_fraction=-1.,
                                 feature="", metric=metric, within_run_std=99.))
                rows[-1].update(context)
            add("mean_wasserstein_scaled", .5 + i*.1 + offset, "Global fidelity")
            for feature in ["creatinine_min", "creatinine_max", "temperature_min"]:
                add("wasserstein_scaled", .4+i*.2+offset, "Per-feature fidelity", feature=feature)
                # A conflicting top-selected value must never enter fixed-feature plots.
                add("wasserstein_scaled", 20., "Prioritized feature fidelity", feature=feature)
            for task in ["mortality", "mortality_balanced"]:
                for protocol in ["replacement", "additive", "synthetic_only"]:
                    for fraction in [0.5, 1.]:
                        for metric in ["pr_auc", "roc_auc", "positive_recall", "positive_f1"]:
                            value = .6+i*.05-offset*(i+1)/3
                            if protocol != "replacement":
                                value += .1
                            if task != "mortality":
                                value -= .1
                            add(metric, value, "Utility", utility_task=task, protocol=protocol, synthetic_fraction=fraction)
    plan = dict(seeds=[40,41,42], primary_metric="mean_wasserstein_scaled", primary_direction="lower",
                stage="val", split_seed=42, smoke=False, data_sha256="same", config={"generator_name":"ctgan","dataset_name":"test"})
    return Study(label, Path("not-loaded"), plan, pd.DataFrame(rows))


def test_pair_before_aggregation_and_keep_missing():
    study = fixture_study()
    data = paired_values(study, "mean_wasserstein_scaled", "A5_NO_SHAP")
    np.testing.assert_allclose(data.delta, -.2)
    assert data.delta.std() < 1e-10  # Not SD of either absolute series / not RF SD.
    study.records = study.records[~((study.records.seed==41) & (study.records.variant=="A5_NO_SHAP"))]
    data = paired_values(study, "mean_wasserstein_scaled", "A5_NO_SHAP")
    assert len(data) == 3 and data.delta.notna().sum() == 2
    assert pd.isna(data.loc[data.seed==41, "delta"].iloc[0])


def test_pairing_rejects_ambiguous_context():
    with pytest.raises(ValueError, match="single evaluation context"):
        paired_values(fixture_study(), "pr_auc", "A0")


def test_utility_uses_one_task_and_protocol_and_labels_every_axis():
    study = fixture_study()
    fig, data = plot_utility_controls(study)
    assert set(data.utility_task) == {"mortality"}
    assert set(data.protocol) == {"replacement"}
    assert set(data.synthetic_fraction) == {1.}
    assert set(data.metric) == {"pr_auc","roc_auc","positive_recall","positive_f1"}
    assert all(ax.get_xlabel() and ax.get_ylabel() for ax in fig.axes)
    expected = paired_values(study,"pr_auc","A5_NO_SHAP", domain="Utility", utility_task="mortality", protocol="replacement", synthetic_fraction=1.)
    np.testing.assert_allclose(data[(data.metric=="pr_auc") & (data.reference=="A5_NO_SHAP")].delta, expected.delta)
    plt.close(fig)


def test_full_feature_export_not_selected_only():
    fig, data = plot_fixed_features(fixture_study(), ["creatinine_min","temperature_min"])
    assert len(data) == 24 and data.value.max() < 2
    assert all(ax.get_xlabel() and ax.get_ylabel() for ax in fig.axes)
    plt.close(fig)
    with pytest.raises(ValueError, match="No continuous"):
        plot_fixed_features(fixture_study(), ["absent"])
    plt.close("all")


def test_primary_negative_direction_and_seed_level_points():
    fig, data = plot_primary([fixture_study()])
    assert len(data) == 9 and data.delta.lt(0).all()
    assert "Negative" in fig.axes[0].get_ylabel()
    assert all("n=3/3" in tick.get_text() for tick in fig.axes[0].get_xticklabels())
    plt.close(fig)


def test_missing_control_and_single_seed_have_visible_coverage():
    study = fixture_study()
    study.records = study.records[(study.records.seed==40) & (study.records.variant!="A5_SHUFFLED_SHAP")]
    fig, data = plot_primary([study])
    ticks = [tick.get_text() for tick in fig.axes[0].get_xticklabels()]
    assert "n=0/3" in ticks[-1] and "n=1/3" in ticks[0]
    assert data[data.reference=="A5_SHUFFLED_SHAP"].delta.isna().all()
    plt.close(fig)


def test_reject_incompatible_datasets():
    a, b = fixture_study("a"), fixture_study("b")
    b.plan["data_sha256"] = "different"
    with pytest.raises(ValueError, match="data_sha256"):
        plot_primary([a,b])


def save_source(tmp_path, study):
    (tmp_path/"study_plan.json").write_text(json.dumps(study.plan))
    study.records.to_csv(tmp_path/"study_evaluation_seed_metrics.csv",index=False)


def test_directory_zip_parity_and_duplicate_detection(tmp_path):
    save_source(tmp_path, fixture_study())
    archive = tmp_path/"export.zip"
    with zipfile.ZipFile(archive,"w") as z:
        for name in ["study_plan.json","study_evaluation_seed_metrics.csv"]:
            z.write(tmp_path/name, "nested/"+name)
    folder, zipped = load_study(tmp_path), load_study(archive)
    pd.testing.assert_frame_equal(folder.records, zipped.records)
    assert not (tmp_path/"nested").exists()  # No ZIP extraction.
    duplicate = fixture_study()
    duplicate.records = pd.concat([duplicate.records,duplicate.records.iloc[:1]])
    save_source(tmp_path,duplicate)
    with pytest.raises(ValueError,match="Duplicate"):
        load_study(tmp_path)


def test_export_read_only_sources_and_png_pdf_provenance(tmp_path):
    save_source(tmp_path,fixture_study())
    before = {p.name:p.read_bytes() for p in tmp_path.iterdir() if p.is_file()}
    exports = export_thesis_figures([load_study(tmp_path)], tmp_path/"figures", features=["creatinine_min"], tasks=["mortality"])
    assert len(exports)==3
    assert before=={p.name:p.read_bytes() for p in tmp_path.iterdir() if p.is_file()}
    for name, png in exports.items():
        assert png.read_bytes().startswith(b"\x89PNG")
        assert png.with_suffix(".pdf").read_bytes().startswith(b"%PDF")
        assert (png.parent/(name+"_points.csv")).is_file()
    manifest=json.loads((tmp_path/"figures/thesis_figures_manifest.json").read_text())
    assert "not CI/SE" in manifest["spread"]
    assert not list((tmp_path/"figures").glob("*.tmp"))
    assert not plt.get_fignums()


def test_download_suffixes_and_nonprivate_label(tmp_path):
    study = fixture_study()
    study.plan["config"].update(generator_name="dp_cgan", generator={"private": False})
    archive = tmp_path / "download.zip"
    with zipfile.ZipFile(archive, "w") as z:
        z.writestr("study_plan (1).json", json.dumps(study.plan))
        z.writestr("study_evaluation_seed_metrics (2).csv", study.records.to_csv(index=False))
    loaded = load_study(archive)
    assert loaded.label == "DP-CGANS (non-private)"
    assert len(loaded.records) == len(study.records)
    with zipfile.ZipFile(archive, "a") as z:
        z.writestr("study_plan.json", json.dumps(study.plan))
    with pytest.raises(ValueError, match="found 2"):
        load_study(archive)
