import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from xai_reweighting.shap_reporting import importance_shares
from xai_reweighting.scoring import normalize_signal


def test_shares_preserve_rank_and_do_not_change_weighting_normalization():
    raw = pd.Series([2.0, 6.0, 2.0], index=["a", "b", "c"])
    original = raw.copy()
    shares = importance_shares(raw)
    np.testing.assert_allclose(shares, [.2, .6, .2])
    np.testing.assert_allclose(normalize_signal(raw), [1 / 3, 1, 1 / 3])
    pd.testing.assert_series_equal(raw, original)
    pd.testing.assert_series_equal(shares.rank(), raw.rank())
    np.testing.assert_allclose(importance_shares(shares), shares)


@pytest.mark.parametrize("values", [[], [0., 0.], [np.nan, 0.]])
def test_empty_zero_shares_remain_zero(values):
    result = importance_shares(pd.Series(values, dtype=float))
    assert result.sum() == 0 and np.isfinite(result).all()


def test_legacy_report_refresh_preserves_scientific_artifacts(tmp_path):
    from xai_reweighting.run_shap_comparison import refresh_shap_comparisons
    baseline = pd.DataFrame({"feature": ["x", "y"], "importance_share": [1., .5],
                             "mean_signed_shap": [-.03, .01], "rank": [1, 2]})
    baseline.to_csv(tmp_path / "baseline_detector_shap.csv", index=False)
    pd.DataFrame({"feature": ["x", "y"] * 2, "epoch": [1, 1, 2, 2],
                  "primary_scope": [True] * 4, "importance_share": [.2, .8, .4, .6],
                  "mean_signed_shap": [-2., 1., -4., 3.]}).to_csv(tmp_path / "discriminator_shap_A0.csv", index=False)
    (tmp_path / "config.json").write_text(json.dumps({"weighting": {"top_k": 1}}))
    protected = {name: b"must stay unchanged" for name in ("feature_scores_A5.csv", "row_weights_A5.csv", "manifest.json")}
    for name, content in protected.items():
        (tmp_path / name).write_bytes(content)
    raw_snapshot = (tmp_path / "discriminator_shap_A0.csv").read_bytes()
    assert refresh_shap_comparisons(tmp_path) == ["A0"]
    result = pd.read_csv(tmp_path / "discriminator_detector_shap_comparison_A0.csv").set_index("feature")
    np.testing.assert_allclose(result.detector_importance_share, [2/3, 1/3])
    np.testing.assert_allclose(result.snapshot_importance_share, [.3, .7])
    np.testing.assert_allclose(result.detector_mean_signed_shap, [-.03, .01])
    np.testing.assert_allclose(result.snapshot_mean_signed_shap, [-3., 2.])
    assert result.detector_importance_share.sum() == pytest.approx(1)
    assert result.snapshot_importance_share.sum() == pytest.approx(1)
    assert (tmp_path / "discriminator_shap_A0.csv").read_bytes() == raw_snapshot
    for name, content in protected.items():
        assert (tmp_path / name).read_bytes() == content
    refresh_shap_comparisons(tmp_path)
    again = pd.read_csv(tmp_path / "discriminator_detector_shap_comparison_A0.csv").set_index("feature")
    pd.testing.assert_frame_equal(result, again)


def test_snapshot_notebook_compiles_and_normalizes_legacy_detector_on_load():
    path = Path(__file__).resolve().parents[1] / "notebooks" / "discriminator_snapshot_analysis.ipynb"
    notebook = json.loads(path.read_text(encoding="utf-8"))
    cells = ["".join(cell["source"]) for cell in notebook["cells"] if cell["cell_type"] == "code"]
    for index, source in enumerate(cells):
        compile(source, f"snapshot-notebook:{index}", "exec")
    assert any('detector_shap["importance_share"] = importance_shares' in source for source in cells)
