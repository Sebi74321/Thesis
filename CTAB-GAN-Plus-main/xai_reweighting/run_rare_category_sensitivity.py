"""Report threshold impact without GAN training or validation/test inspection."""

import argparse
from pathlib import Path

import pandas as pd

from .data_split import create_data_splits
from .io_utils import atomic_write_csv, atomic_write_json, file_sha256
from .rare_categories import pooling_sensitivity, pooling_settings
from .run_ablation import _load_config


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--seed", type=int)
    parser.add_argument("--thresholds", default="3,6,10,20")
    args = parser.parse_args(argv)
    config = _load_config(args.config.resolve())
    seed = args.seed if args.seed is not None else int(config.get("split_seed", config.get("seed", 42)))
    config.setdefault("rare_categories", {})["sensitivity_thresholds"] = [int(x) for x in args.thresholds.split(",")]
    pooling_settings(config)
    data_path = Path(__file__).resolve().parents[1] / config["data_path"]
    data = pd.read_csv(data_path)
    splits = create_data_splits(data, config["target_col"], seed=seed, **config.get("split", {}))
    summary, counts = pooling_sensitivity(splits.train, config)
    if args.output_dir.exists():
        raise FileExistsError("Choose a new output directory for the sensitivity report")
    atomic_write_csv(args.output_dir / "rare_category_sensitivity.csv", summary)
    atomic_write_csv(args.output_dir / "rare_category_counts.csv", counts)
    atomic_write_json(args.output_dir / "manifest.json", {
        "status": "complete", "analysis": "training_only_preprocessing_impact",
        "gan_performance_evaluated": False, "data_sha256": file_sha256(data_path),
        "split_seed": seed, "train_indices": splits.indices["train"],
        "settings": pooling_settings(config),
    })
    print(summary[summary["scope"] == "any_feature"].to_string(index=False))
    print(args.output_dir.resolve())
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
