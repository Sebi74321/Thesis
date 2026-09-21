import json

import numpy as np
import pandas as pd
import pytest

from xai_reweighting.io_utils import file_sha256
from xai_reweighting.resolution_diagnostics import (
    FEATURE, interval_frequencies, resolution_summary,
)
from xai_reweighting.run_resolution_diagnostics import run_existing_resolution_diagnostics


def test_minute_grid_tolerates_csv_precision_without_accepting_fractional_minutes():
    values = pd.Series([0.0, 0.000694444, -0.001388889, 0.002777778])
    rows = pd.DataFrame(resolution_summary(values, "REAL_TRAIN", "train")).set_index("grid")
    assert rows.loc["minute", "nonzero_grid_aligned_fraction"] == 1.0
    rows = pd.DataFrame(resolution_summary(pd.Series([0.5 / 1440]), "A0", "val")).set_index("grid")
    assert rows.loc["minute", "grid_aligned_fraction"] == 0.0


def test_interval_partition_and_exact_boundaries():
    values = pd.Series([-1, 0, 1 / 1440, 0.003472222, 60 / 1440, 1, 2, np.nan, np.inf, "bad"])
    bins = interval_frequencies(values)
    assert sum(count for count, _ in bins.values()) == len(values)
    assert sum(freq for _, freq in bins.values()) == pytest.approx(1)
    assert bins["(0, 5] minutes"][0] == 2
    assert bins["(5, 60] minutes"][0] == 1
    assert bins["(1, 24] hours"][0] == 1
    assert bins["missing"][0] == 1
    assert bins["nonfinite_or_invalid"][0] == 2


def test_rounding_view_is_nonmutating_and_zero_is_separate():
    values = pd.Series([0.1 / 1440, -0.1 / 1440, 0.9 / 1440])
    original = values.copy()
    assert interval_frequencies(values)["exact_zero"][0] == 0
    assert interval_frequencies(values, rounded_minutes=True)["exact_zero"][0] == 2
    pd.testing.assert_series_equal(values, original)


def test_empty_alignment_is_unavailable_not_perfect():
    rows = resolution_summary(pd.Series([np.nan]), "REAL_TRAIN", "train")
    assert all(np.isnan(row["grid_aligned_fraction"]) for row in rows)


def make_run(tmp_path):
    source = tmp_path / "real.csv"
    pd.DataFrame({FEATURE: [0, 1 / 1440, 2 / 1440, 1000]}).to_csv(source, index=False)
    config = {"data_path": str(source), "stage": "val", "continuous_cols": [FEATURE]}
    (tmp_path / "config.json").write_text(json.dumps(config))
    (tmp_path / "manifest.json").write_text(json.dumps({"data_sha256": file_sha256(source)}))
    (tmp_path / "split_indices.json").write_text(json.dumps({"train": [0, 1], "audit": [], "val": [2], "test": [3]}))
    pd.DataFrame({FEATURE: [2 / 1440, 2 / 1440]}).to_csv(tmp_path / "synthetic_A0.csv", index=False)
    return source


def test_standalone_uses_saved_stage_skips_missing_variants_preserves_inputs(tmp_path):
    source = make_run(tmp_path)
    original = {path: path.read_bytes() for path in tmp_path.iterdir()}
    artifacts = run_existing_resolution_diagnostics(tmp_path)
    assert len(artifacts) == 3
    assert all(path.read_bytes() == data for path, data in original.items())
    report = pd.read_csv(tmp_path / "duration_interval_frequencies.csv")
    assert set(report.source) == {"REAL", "A0"}
    assert set(report.evaluation_split) == {"val"}
    assert report.loc[report.region == ">24 hours", "count"].sum() == 0
    assert report.absolute_frequency_gap.max() == 0
    grid = pd.read_csv(tmp_path / "duration_resolution.csv")
    assert set(grid.loc[grid.source == "REAL_TRAIN", "rows"]) == {2}
    manifest = json.loads((tmp_path / "duration_diagnostics_manifest.json").read_text())
    assert manifest["rounding_applied_to_datasets"] is False
    source.write_text("changed")
    with pytest.raises(ValueError, match="hash"):
        run_existing_resolution_diagnostics(tmp_path)
