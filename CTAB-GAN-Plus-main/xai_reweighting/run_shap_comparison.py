"""Refresh saved SHAP comparison reports without GAN training or SHAP fitting."""

import argparse
import json
from pathlib import Path

import pandas as pd

from .io_utils import atomic_write_csv, atomic_write_json
from .shap_reporting import importance_shares


def refresh_shap_comparisons(run_dir):
    from .run_ablation import _save_discriminator_detector_comparison

    run_dir = Path(run_dir).expanduser().resolve()
    baseline_path = run_dir / "baseline_detector_shap.csv"
    baseline = pd.read_csv(baseline_path)
    required = {"feature", "importance_share", "mean_signed_shap"}
    if not required.issubset(baseline) or baseline["feature"].duplicated().any():
        raise ValueError("Expected unique features and importance/signed columns in baseline_detector_shap.csv")
    baseline["importance_share"] = importance_shares(baseline["importance_share"])
    config = json.loads((run_dir / "config.json").read_text(encoding="utf-8"))
    reference = baseline.set_index("feature")
    variants = []
    for variant in (f"A{i}" for i in range(6)):
        path = run_dir / f"discriminator_shap_{variant}.csv"
        if not path.is_file():
            continue
        _save_discriminator_detector_comparison(
            run_dir, variant, reference["importance_share"], reference["mean_signed_shap"],
            int(config.get("weighting", {}).get("top_k", 5)),
            int(config.get("discriminator_shap", {}).get("late_window_snapshots", 3)),
        )
        variants.append(variant)
    atomic_write_csv(baseline_path, baseline)
    atomic_write_json(run_dir / "shap_reporting_manifest.json", {
        "importance_normalization": "sum_to_one_or_all_zero",
        "snapshot_variants_processed": variants,
        "training_performed": False, "shap_recomputed": False,
        "weighting_signals_changed": False, "raw_signed_shap_changed": False,
    })
    return variants


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", type=Path, required=True)
    args = parser.parse_args(argv)
    variants = refresh_shap_comparisons(args.run_dir)
    print("Normalized detector report; processed snapshot variants: " + (", ".join(variants) or "none available"))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
