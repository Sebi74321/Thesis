import json

import pandas as pd
import pytest

from xai_reweighting.io_utils import file_sha256
from xai_reweighting.run_evaluation import build_parser, run_existing_evaluation


def _completed_run(tmp_path):
    data = pd.DataFrame(
        {
            "value": list(range(10)),
            "category": ["a", "b"] * 5,
            "target": [0, 1] * 5,
        }
    )
    data_path = tmp_path / "source.csv"
    data.to_csv(data_path, index=False)
    run_dir = tmp_path / "completed"
    run_dir.mkdir()
    config = {
        "data_path": str(data_path),
        "target_col": "target",
        "categorical_cols": ["category", "target"],
        "continuous_cols": ["value"],
        "seed": 42,
        "stage": "val",
        "smoke": False,
        "variants": ["A0", "A5"],
        "evaluation": {"n_estimators": 5, "real_baseline_repeats": 1},
        "mixed_utility": {"enabled": False},
        "utility_tasks": [
            {
                "name": "mortality",
                "balance": "imbalanced",
                "target_col": "target",
                "positive_label": "1",
            }
        ],
    }
    (run_dir / "config.json").write_text(json.dumps(config), encoding="utf-8")
    (run_dir / "manifest.json").write_text(
        json.dumps(
            {
                "status": "complete",
                "data_sha256": file_sha256(data_path),
                "variants_completed": ["A0", "A5"],
            }
        ),
        encoding="utf-8",
    )
    (run_dir / "split_indices.json").write_text(
        json.dumps(
            {
                "train": [0, 1, 2, 3, 4, 5],
                "audit": [6],
                "val": [7, 8],
                "test": [9],
            }
        ),
        encoding="utf-8",
    )
    train = data.iloc[:6].reset_index(drop=True)
    train.to_csv(run_dir / "synthetic_A0.csv", index=False)
    changed = train.copy()
    changed["value"] += 1
    changed.to_csv(run_dir / "synthetic_A5.csv", index=False)
    (run_dir / "metrics_A5.json").write_text(
        json.dumps({"training_rows": 9, "weight_mean": 1.7}), encoding="utf-8"
    )
    return run_dir


def test_parser_supports_full_reevaluation_options():
    args = build_parser().parse_args(
        ["--run-dir", "result", "--variants", "A0,A5", "--progress", "on"]
    )
    assert args.variants == "A0,A5"
    assert args.progress == "on"
    assert not args.skip_diagnostics
    assert not args.skip_mixed_utility


def test_completed_run_is_reevaluated_without_training(tmp_path, monkeypatch):
    run_dir = _completed_run(tmp_path)

    def fake_real_only(*args, **kwargs):
        return pd.DataFrame(
            {
                "protocol": ["real_only"],
                "repeat": [0],
                "seed": [42],
                "roc_auc": [0.9],
            }
        )

    def fake_evaluate(real_train, real_eval, synthetic, *args, **kwargs):
        score = 0.7 if synthetic["value"].iloc[0] == 0 else 0.8
        return (
            {"utility_mortality_roc_auc": score, "detector_auc": 1.0 - score},
            pd.DataFrame({"feature": ["value"], "wasserstein_scaled": [score]}),
        )

    monkeypatch.setattr(
        "xai_reweighting.run_evaluation.evaluate_real_only_baseline", fake_real_only
    )
    monkeypatch.setattr("xai_reweighting.run_evaluation.evaluate_variant", fake_evaluate)

    output = run_existing_evaluation(
        run_dir,
        ["A5"],
        progress="off",
        run_diagnostics=False,
        run_mixed_utility=False,
    )

    assert output == run_dir.resolve()
    summary = pd.read_csv(run_dir / "ablation_summary.csv").set_index("variant")
    assert set(summary.index) == {"A0", "A5"}
    assert summary.loc["A5", "delta_utility_mortality_roc_auc_vs_A0"] == pytest.approx(0.1)
    assert summary.loc["A5", "delta_utility_mortality_roc_auc_vs_real"] == pytest.approx(-0.1)
    assert summary.loc["A5", "training_rows"] == 9
    assert summary.loc["A5", "weight_mean"] == 1.7
    rerun = json.loads((run_dir / "evaluation_rerun_manifest.json").read_text())
    assert rerun["status"] == "complete"
    assert rerun["training_performed"] is False


def test_reevaluation_rejects_invalid_saved_schema(tmp_path, monkeypatch):
    run_dir = _completed_run(tmp_path)
    invalid = pd.read_csv(run_dir / "synthetic_A5.csv").drop(columns=["category"])
    invalid.to_csv(run_dir / "synthetic_A5.csv", index=False)
    monkeypatch.setattr(
        "xai_reweighting.run_evaluation.evaluate_real_only_baseline",
        lambda *args, **kwargs: pd.DataFrame(
            {"protocol": ["real_only"], "repeat": [0], "seed": [42], "roc_auc": [0.9]}
        ),
    )

    try:
        run_existing_evaluation(
            run_dir,
            ["A5"],
            progress="off",
            run_diagnostics=False,
            run_mixed_utility=False,
        )
    except ValueError as exc:
        assert "does not preserve" in str(exc)
    else:
        raise AssertionError("Invalid saved schema was accepted")
    rerun = json.loads((run_dir / "evaluation_rerun_manifest.json").read_text())
    assert rerun["status"] == "failed"


def test_full_mode_orchestrates_mixed_utility_and_diagnostics(tmp_path, monkeypatch):
    run_dir = _completed_run(tmp_path)
    config_path = run_dir / "config.json"
    config = json.loads(config_path.read_text())
    config["mixed_utility"]["enabled"] = True
    config_path.write_text(json.dumps(config), encoding="utf-8")
    calls = []

    def fake_mixed(selected_run, variants, **kwargs):
        calls.append(("mixed", tuple(variants)))
        pd.DataFrame(
            {
                "utility_task": ["mortality"],
                "target_balance": ["imbalanced"],
                "protocol": ["real_only"],
                "repeat": [0],
                "seed": [42],
                "roc_auc": [0.9],
            }
        ).to_csv(selected_run / "utility_real_only_baseline.csv", index=False)

    def fake_diagnostics(selected_config, selected_run):
        calls.append(("diagnostics", selected_run.name))

    monkeypatch.setattr("xai_reweighting.run_evaluation.run_existing_mixed_utility", fake_mixed)
    monkeypatch.setattr("xai_reweighting.run_evaluation.run_existing_diagnostics", fake_diagnostics)
    monkeypatch.setattr(
        "xai_reweighting.run_evaluation.evaluate_variant",
        lambda *args, **kwargs: (
            {"utility_mortality_roc_auc": 0.7, "detector_auc": 0.8},
            pd.DataFrame({"feature": ["value"]}),
        ),
    )

    run_existing_evaluation(run_dir, progress="off")

    assert ("mixed", ("A0", "A5")) in calls
    assert ("diagnostics", run_dir.name) in calls
