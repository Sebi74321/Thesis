import json
from pathlib import Path


def test_orchestrator_notebook_is_valid_and_code_cells_compile():
    path = Path(__file__).resolve().parents[1] / "notebooks" / "xai_retraining_orchestrator.ipynb"
    notebook = json.loads(path.read_text(encoding="utf-8"))

    assert notebook["nbformat"] == 4
    code_cells = [cell for cell in notebook["cells"] if cell["cell_type"] == "code"]
    assert code_cells
    for index, cell in enumerate(code_cells):
        compile("".join(cell["source"]), f"notebook-cell-{index}", "exec")

    source = "\n".join("".join(cell["source"]) for cell in code_cells)
    assert "RUN_EXPERIMENT = False" in source
    assert "RERUN_EVALUATION = False" in source
    assert "xai_reweighting.run_ablation" in source
    assert "xai_reweighting.run_evaluation" in source
    assert "ablation_summary.csv" in source
    assert "feature_scores_" in source
    assert "top_shap_feature_variant_metrics.csv" in source
    assert "delta_discrepancy_vs_A0" in source
    assert 'sharey=False' in source
    assert 'sharex=False' in source
    assert 'facet_kws={"sharey": False}' not in source
    assert 'axis.set_xlabel("Variant")' in source
    assert 'axis.set_ylabel("Discrepancy delta vs A0")' in source
    assert 'source_order = ["Real audit", "A0 synthetic", "A5 synthetic"]' in source
    assert 'a0_path = run_directory / "synthetic_A0.csv"' in source
    assert 'a5_path = run_directory / "synthetic_A5.csv"' in source
    assert source.count("hue_order=source_order") == 3
    assert source.count("palette=source_palette") == 3
    assert "A0 corrected synthetic" not in source
    assert "A0 raw synthetic audit" not in source


def test_rq1_notebook_is_valid_and_safe_by_default():
    path = Path(__file__).resolve().parents[1] / "notebooks" / "rq1_model_comparison.ipynb"
    notebook = json.loads(path.read_text(encoding="utf-8"))
    code_cells = [cell for cell in notebook["cells"] if cell["cell_type"] == "code"]
    for index, cell in enumerate(code_cells):
        compile("".join(cell["source"]), f"rq1-notebook-cell-{index}", "exec")
    source = "\n".join("".join(cell["source"]) for cell in code_cells)
    assert "RUN_EXPERIMENT = False" in source
    assert "xai_reweighting.run_model_comparison" in source
    assert "rq1_summary.csv" in source
