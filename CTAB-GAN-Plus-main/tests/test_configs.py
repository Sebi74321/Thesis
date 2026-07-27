import json
from pathlib import Path

import pandas as pd
import pytest


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


def test_mimic_age_is_general_and_integer():
    config = json.loads(
        (PROJECT_ROOT / "configs" / "mimic_ctabgan.json").read_text(encoding="utf-8")
    )

    assert "age_at_intime" in config["generator"]["general_columns"]
    assert "age_at_intime" in config["generator"]["integer_columns"]


def test_wids_large_dataset_settings():
    config = json.loads(
        (PROJECT_ROOT / "configs" / "wids_ctabgan.json").read_text(encoding="utf-8")
    )

    assert config["generator"]["batch_size"] == 1024
    assert config["generator"]["epochs"] == 200
    assert config["smoke_rows"] == 5000
    assert config["evaluation"]["privacy_max_reference_rows"] == 20000
    assert config["evaluation"]["privacy_max_query_rows"] == 10000
