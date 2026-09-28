"""Inspect saved duration values without training, correction, or classifier fits."""

import argparse
import json
from pathlib import Path

import pandas as pd

from .io_utils import file_sha256
from .resolution_diagnostics import FEATURE, save_resolution_diagnostics


def run_existing_resolution_diagnostics(run_dir):
    run_dir = Path(run_dir).expanduser().resolve()
    config = json.loads((run_dir / "config.json").read_text(encoding="utf-8"))
    manifest = json.loads((run_dir / "manifest.json").read_text(encoding="utf-8"))
    if config.get("smoke", False):
        raise ValueError("Standalone duration diagnostics require a non-smoke run with original source indices")
    if FEATURE not in config.get("continuous_cols", []):
        return []
    stage = config.get("stage", "val")
    if stage not in ("val", "test"):
        raise ValueError("Saved evaluation stage must be val or test")
    source = Path(__file__).resolve().parents[1] / config["data_path"]
    if file_sha256(source) != manifest.get("data_sha256"):
        raise ValueError("Source data hash does not match the selected run")
    data = pd.read_csv(source, usecols=[FEATURE])
    indices = json.loads((run_dir / "split_indices.json").read_text(encoding="utf-8"))
    paths = {f"A{i}": run_dir / f"synthetic_A{i}.csv" for i in range(6)}
    if not any(path.is_file() for path in paths.values()):
        raise FileNotFoundError("No saved ablation synthetic datasets found")
    return save_resolution_diagnostics(data.iloc[indices["train"]], data.iloc[indices[stage]],
                                       paths, run_dir, evaluation_split=stage)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", type=Path, required=True)
    args = parser.parse_args(argv)
    artifacts = run_existing_resolution_diagnostics(args.run_dir)
    print("Saved: " + ", ".join(artifacts) if artifacts else "No configured pre_icu_los_days feature; nothing to inspect.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
