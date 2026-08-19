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
