import json
from pathlib import Path

import pandas as pd
import pytest

from xai_reweighting.run_model_comparison import _load_config as load_rq1_config
from xai_reweighting.run_ablation import _load_config as load_ablation_config


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
    assert set(tasks) == {"mortality", "mortality_balanced"}
    assert tasks["mortality"]["target_col"] == config["target_col"]
    assert tasks["mortality"]["balance"] == "imbalanced"
    assert tasks["mortality"]["balance_strategy"] == "match_real_train"
    assert tasks["mortality_balanced"]["target_col"] == config["target_col"]
    assert tasks["mortality_balanced"]["positive_label"] == "1"
    assert tasks["mortality_balanced"]["balance"] == "balanced"
    assert tasks["mortality_balanced"]["balance_strategy"] == "fixed_50_50"
    weighting = config["weighting"]
    assert weighting["alpha"] == 4.0
    assert weighting["gamma"] == 0.6
    assert weighting["top_k"] == 5
    assert weighting["w_max"] == 3.0
    assert weighting["correlation_aware_selection"] is False
    assert weighting["correlation_threshold"] == 0.65
    assert "max_per_correlation_group" not in weighting
    assert config["feature_exclusion_sensitivity"]["enabled"] is True
    assert config["detector"]["shap_scope"] == "misclassified_holdout_only"


def test_mimic_age_is_general_and_integer():
    config = json.loads(
        (PROJECT_ROOT / "configs" / "mimic_ctabgan.json").read_text(encoding="utf-8")
    )

    assert "age_at_intime" in config["generator"]["general_columns"]
    assert "age_at_intime" in config["generator"]["integer_columns"]
    assert config["generator"]["mixed_columns"]["spo2_max"] == [100.0]
    assert set(config["generator"]["log_columns"]) == {"wbc_min", "wbc_max"}
    assert config["generator"]["snapshot_frq"] == 25
    assert config["discriminator_shap"] == {
        "enabled": True,
        "background_size": 50,
        "explain_size": 100,
        "exclude_target": True,
    }


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


@pytest.mark.parametrize(
    ("name", "generator_name", "batch_size", "epochs"),
    [
        ("mimic_ctgan.json", "ctgan", 500, 150),
        ("mimic_dpcgan.json", "dp_cgan", 500, 150),
        ("wids_ctgan.json", "ctgan", 1000, 200),
        ("wids_dpcgan.json", "dp_cgan", 1000, 200),
    ],
)
def test_weighted_multigan_configs_inherit_dataset_protocol(
    name, generator_name, batch_size, epochs
):
    config = load_ablation_config(PROJECT_ROOT / "configs" / name)

    assert config["generator_name"] == generator_name
    assert config["generator"]["categorical_columns"] == config["categorical_cols"]
    assert config["generator"]["batch_size"] == batch_size
    assert config["generator"]["epochs"] == epochs
    assert config["weighting"]["top_k"] == 5
    assert {task["name"] for task in config["utility_tasks"]} == {
        "mortality",
        "mortality_balanced",
    }
    if generator_name == "dp_cgan":
        assert config["generator"]["private"] is True
        assert config["generator"]["discriminator_steps"] == 10


def test_all_weighted_configs_use_the_same_reweighting_protocol():
    names = [
        "mimic_ctabgan.json", "mimic_ctgan.json", "mimic_dpcgan.json",
        "wids_ctabgan.json", "wids_ctgan.json", "wids_dpcgan.json",
    ]
    protocols = [load_ablation_config(PROJECT_ROOT / "configs" / name)["weighting"] for name in names]

    assert all(protocol == protocols[0] for protocol in protocols[1:])
