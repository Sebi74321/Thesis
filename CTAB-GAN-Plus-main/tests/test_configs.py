import json
from pathlib import Path

import pandas as pd
import pytest

from xai_reweighting.run_model_comparison import _load_config as load_rq1_config


PROJECT_ROOT = Path(__file__).resolve().parents[1]


@pytest.mark.parametrize("config_name", ["mimic_ctabgan.json", "wids_ctabgan.json"])
def test_dataset_config_matches_csv_schema(config_name):
    config = json.loads((PROJECT_ROOT / "configs" / config_name).read_text(encoding="utf-8"))
    columns = pd.read_csv(PROJECT_ROOT / config["data_path"], nrows=0).columns.tolist()
    categorical = config["categorical_cols"]
    continuous = config["continuous_cols"]

    assert not set(categorical).intersection(continuous)
    assert set(categorical).union(continuous) == set(columns)
    assert config["target_col"] in categorical
    assert config["generator"]["categorical_columns"] == categorical
    assert set(config["generator"]["general_columns"]) == set(continuous)
    mixed = config["mixed_utility"]
    assert mixed["enabled"] is True
    assert mixed["additive_fractions"][0] == 0.0
    assert mixed["replacement_fractions"] == [0.0, 0.25, 0.5, 0.75, 1.0]
    assert "threshold_beta" not in mixed
    tasks = {task["name"]: task for task in config["utility_tasks"]}
    assert set(tasks) == {"mortality", "gender"}
    assert tasks["mortality"]["target_col"] == config["target_col"]
    assert tasks["mortality"]["balance"] == "imbalanced"
    assert tasks["gender"]["target_col"] == "gender"
    assert tasks["gender"]["positive_label"] == "F"
    assert tasks["gender"]["balance"] == "balanced"
    weighting = config["weighting"]
    assert weighting["correlation_aware_selection"] is True
    assert weighting["correlation_threshold"] == 0.65
    assert weighting["max_per_correlation_group"] == 1
    assert config["feature_exclusion_sensitivity"]["enabled"] is True


def test_mimic_age_is_general_and_integer():
    config = json.loads(
        (PROJECT_ROOT / "configs" / "mimic_ctabgan.json").read_text(encoding="utf-8")
    )

    assert "age_at_intime" in config["generator"]["general_columns"]
    assert "age_at_intime" in config["generator"]["integer_columns"]
    assert config["generator"]["mixed_columns"]["spo2_max"] == [100.0]
    assert set(config["generator"]["log_columns"]) == {"wbc_min", "wbc_max"}


def test_wids_large_dataset_settings():
    config = json.loads(
        (PROJECT_ROOT / "configs" / "wids_ctabgan.json").read_text(encoding="utf-8")
    )

    assert config["generator"]["batch_size"] == 1024
    assert config["generator"]["epochs"] == 200
    assert config["smoke_rows"] == 5000
    assert config["evaluation"]["privacy_max_reference_rows"] == 20000
    assert config["evaluation"]["privacy_max_query_rows"] == 10000
    assert config["mixed_utility"]["repeats"] == 3
    assert config["mixed_utility"]["n_estimators"] == 100


@pytest.mark.parametrize("name", ["rq1_mimic.json", "rq1_wids.json"])
def test_rq1_configs_define_all_models(name):
    config = load_rq1_config(PROJECT_ROOT / "configs" / name)
    assert set(config["models"]) == {"ctabgan_plus", "ctgan", "dp_cgan"}
    assert config["models"]["dp_cgan"]["private"] is True
    assert config["models"]["dp_cgan"]["saved_transformer"] is None
    assert config["models"]["dp_cgan"]["discriminator_steps"] == 10
    assert config["models"]["ctabgan_plus"]["categorical_columns"] == config["categorical_cols"]
