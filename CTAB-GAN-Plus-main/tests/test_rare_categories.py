import json
from pathlib import Path

import pandas as pd
import pytest

from xai_reweighting.data_split import DataSplits
from xai_reweighting.rare_categories import (
    apply_saved_pooling, fit_pooling, pooling_sensitivity, pooling_settings,
    prepare_pooled_splits, transform_pooling,
)


def fixture_data():
    train = pd.DataFrame({"code": ["common"] * 6 + ["border"] * 5 + ["singleton"],
                          "target": [0] * 11 + [1], "value": range(12)})
    config = {"target_col": "target", "categorical_cols": ["code", "target"],
              "rare_categories": {"enabled": True, "min_count": 6}}
    return train, config


def test_boundary_exempt_target_and_rows_preserved():
    train, config = fixture_data()
    original = train.copy(deep=True)
    mapping = fit_pooling(train, config)
    transformed = transform_pooling(train, mapping)
    assert mapping["columns"]["code"]["retained"] == ["common"]
    assert transformed["code"].tolist() == ["common"] * 6 + ["__OTHER_RARE__"] * 6
    pd.testing.assert_frame_equal(train, original)
    pd.testing.assert_frame_equal(transformed[["target", "value"]], train[["target", "value"]])
    assert "target" not in mapping["columns"]
    pd.testing.assert_frame_equal(transform_pooling(transformed, mapping), transformed)


def test_audit_frequency_cannot_promote_an_unknown_category():
    train, config = fixture_data()
    mapping = fit_pooling(train, config)
    audit = pd.DataFrame({"code": ["audit_only"] * 100, "target": [1] * 100, "value": range(100)})
    transformed = transform_pooling(audit, mapping)
    assert set(transformed["code"]) == {"__OTHER_RARE__"}
    assert "audit_only" not in mapping["columns"]["code"]["counts"]
    assert mapping == fit_pooling(train, config)


def test_sensitivity_reports_overlapping_rows_once_and_preserves_thresholds():
    train, config = fixture_data()
    train["code2"] = train["code"]
    config["categorical_cols"].append("code2")
    summary, counts = pooling_sensitivity(train, config)
    total = summary[summary["scope"] == "any_feature"].set_index("threshold")
    assert total.loc[6, "pooled_rows"] == 6
    assert total.loc[6, "pooled_levels"] == 4
    assert total.loc[6, "pooled_row_fraction"] == 0.5
    assert total.loc[3, "pooled_rows"] == 1
    assert total.loc[10, "pooled_rows"] == 12
    assert set(summary["threshold"]) == {3, 6, 10, 20}
    assert "target" not in counts["feature"].values


def test_persisted_mapping_reused_for_all_splits_and_reruns(tmp_path):
    train, config = fixture_data()
    audit = train.iloc[:2].copy()
    audit["code"] = "audit_only"
    indices = {"train": list(range(12)), "audit": [12, 13], "val": [14, 15], "test": [16, 17]}
    splits = DataSplits(train, audit, audit.copy(), audit.copy(), indices)
    pooled = prepare_pooled_splits(splits, config, tmp_path)
    assert pooled.indices == indices
    for part in ("audit", "val", "test"):
        assert set(getattr(pooled, part)["code"]) == {"__OTHER_RARE__"}
    pd.testing.assert_frame_equal(apply_saved_pooling(audit, tmp_path, config), pooled.audit)
    resumed = prepare_pooled_splits(splits, config, tmp_path, resume=True)
    pd.testing.assert_frame_equal(resumed.train, pooled.train)
    manifest = json.loads((tmp_path / "rare_category_sensitivity_manifest.json").read_text())
    assert manifest["gan_performance_evaluated"] is False
    changed = train.copy()
    changed.loc[0, "code"] = "change"
    with pytest.raises(ValueError, match="mapping"):
        prepare_pooled_splits(DataSplits(changed, audit, audit, audit, indices), config, tmp_path, resume=True)


def test_missing_mapping_is_never_silently_refitted(tmp_path):
    train, config = fixture_data()
    with pytest.raises(FileNotFoundError):
        apply_saved_pooling(train, tmp_path, config)
    pd.testing.assert_frame_equal(apply_saved_pooling(train, tmp_path, {**config, "rare_categories": {}}), train)


def test_missing_numeric_categories_and_token_collision(tmp_path):
    train, config = fixture_data()
    train["code"] = [101.05] * 6 + [None] * 6
    mapping = fit_pooling(train, config)
    pd.testing.assert_frame_equal(transform_pooling(train, mapping), train)
    train["code"] = train["code"].astype(object)
    train.loc[0, "code"] = "__OTHER_RARE__"
    with pytest.raises(ValueError, match="Reserved"):
        fit_pooling(train, config)


@pytest.mark.parametrize("value", [0, 1, -1, True, 6.5])
def test_invalid_threshold_rejected(value):
    _, config = fixture_data()
    config["rare_categories"]["min_count"] = value
    with pytest.raises(ValueError, match="thresholds"):
        pooling_settings(config)


def test_all_utility_targets_exempt():
    train, config = fixture_data()
    config["utility_tasks"] = [{"target_col": "code"}]
    mapping = fit_pooling(train, config)
    assert not mapping["columns"]
    pd.testing.assert_frame_equal(transform_pooling(train, mapping), train)


@pytest.mark.parametrize("dataset", ["mimic", "wids"])
@pytest.mark.parametrize("model", ["ctabgan", "ctgan", "dpcgan", "rq1"])
def test_all_experiment_configs_inherit_pooling(dataset, model):
    from xai_reweighting.run_ablation import _load_config
    root = Path(__file__).resolve().parents[1]
    name = f"rq1_{dataset}" if model == "rq1" else f"{dataset}_{model}"
    config = _load_config(root / "configs" / f"{name}.json")
    settings = pooling_settings(config)
    assert settings["enabled"] and settings["min_count"] == 6
    assert settings["sensitivity_thresholds"] == [3, 6, 10, 20]
    assert config["target_col"] in settings["excluded_features"]


def test_report_cli_uses_only_training_counts(tmp_path):
    from xai_reweighting.run_rare_category_sensitivity import main
    from xai_reweighting.data_split import create_data_splits
    data = pd.DataFrame({"code": [f"code_{i}" for i in range(100)], "target": [0, 1] * 50})
    source = tmp_path / "source.csv"
    data.to_csv(source, index=False)
    config = {"data_path": str(source), "target_col": "target", "categorical_cols": ["code", "target"], "generator": {},
              "rare_categories": {"enabled": True, "min_count": 6}}
    path = tmp_path / "config.json"
    path.write_text(json.dumps(config))
    output = tmp_path / "report"
    assert main(["--config", str(path), "--output-dir", str(output)]) == 0
    counts = pd.read_csv(output / "rare_category_counts.csv")
    expected = create_data_splits(data, "target", seed=42)
    assert set(counts["category"]) == set(expected.train["code"])
    manifest = json.loads((output / "manifest.json").read_text())
    assert manifest["gan_performance_evaluated"] is False
    assert manifest["train_indices"] == expected.indices["train"]


def test_notebook_cells_compile():
    root = Path(__file__).resolve().parents[1]
    for name in ("xai_retraining_orchestrator", "rq1_model_comparison"):
        notebook = json.loads((root / "notebooks" / f"{name}.ipynb").read_text(encoding="utf-8"))
        for index, cell in enumerate(notebook["cells"]):
            if cell["cell_type"] == "code":
                compile("".join(cell["source"]), f"{name}:{index}", "exec")
