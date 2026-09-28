"""Training-only rare-category pooling shared by all experiment/evaluation paths."""

from __future__ import annotations

import hashlib
import json
from numbers import Integral
from pathlib import Path

import numpy as np
import pandas as pd

from .data_split import DataSplits
from .io_utils import atomic_write_csv, atomic_write_json


MAPPING_FILE = "rare_category_mapping.json"
MISSING = "__MISSING_CATEGORY__"


def _values(series):
    values = series.astype(object)
    return values.mask(values.isna() | values.eq(" "), MISSING).astype(str)


def pooling_settings(config):
    settings = dict(config.get("rare_categories") or {})
    enabled = settings.get("enabled", False)
    if not isinstance(enabled, bool):
        raise ValueError("rare_categories.enabled must be boolean")
    count = settings.get("min_count", 6)
    thresholds = settings.get("sensitivity_thresholds", [3, 6, 10, 20])
    for value in [count, *thresholds]:
        if isinstance(value, bool) or not isinstance(value, Integral) or value < 2:
            raise ValueError("Rare-category thresholds must be integers of at least 2")
    token = settings.get("pooled_label", "__OTHER_RARE__")
    if not isinstance(token, str) or not token.strip() or token == MISSING:
        raise ValueError("rare_categories.pooled_label must be a nonempty, distinct string")
    excluded = set(settings.get("exclude_features", [])) | {config["target_col"]}
    excluded.update(task["target_col"] for task in config.get("utility_tasks", []))
    return {
        "enabled": enabled, "min_count": int(count), "pooled_label": token,
        "sensitivity_thresholds": sorted(set([int(count), *map(int, thresholds)])),
        "excluded_features": sorted(excluded),
    }


def fit_pooling(train, config):
    """Fit solely on original real_train, before variant augmentation."""
    settings = pooling_settings(config)
    columns = {}
    for feature in config["categorical_cols"]:
        if feature in settings["excluded_features"] or not settings["enabled"]:
            continue
        values = _values(train[feature])
        if values.isin([settings["pooled_label"]]).any() or train[feature].eq(MISSING).any():
            raise ValueError(f"Reserved rare-category label already present in {feature!r}")
        counts = values.value_counts().sort_index()
        columns[feature] = {
            "counts": {value: int(count) for value, count in counts.items()},
            "retained": counts.index[counts >= settings["min_count"]].tolist(),
            "pooled": counts.index[counts < settings["min_count"]].tolist(),
        }
    return {
        "version": 1, **settings, "training_rows": len(train),
        "training_sha256": hashlib.sha256(pd.util.hash_pandas_object(train, index=False).values.tobytes()).hexdigest(),
        "fit_population": "original_real_train_before_augmentation",
        "evaluation_scope": "pooled_categorical_representation",
        "columns": columns,
    }


def transform_pooling(frame, mapping):
    """Apply a frozen mapping; unknown values share the pooled label.

    No rows are dropped, no numerical predictors/targets are modified, and
    applying the mapping again is safe (including generated OTHER_RARE values).
    """
    result = frame.copy(deep=True)
    if not mapping.get("enabled", False):
        return result
    for feature, info in mapping["columns"].items():
        values = _values(result[feature])
        pool = ~values.isin(info["retained"])
        if info["pooled"] or pool.any():
            result[feature] = values.where(~pool, mapping["pooled_label"])
    return result


def pooling_sensitivity(train, config):
    """Training support/pooled mass only; these are not GAN performance results."""
    settings = pooling_settings(config)
    rows, category_rows = [], []
    for threshold in settings["sensitivity_thresholds"]:
        any_pooled = np.zeros(len(train), dtype=bool)
        total_levels = pooled_levels = 0
        for feature in config["categorical_cols"]:
            if feature in settings["excluded_features"]:
                continue
            values = _values(train[feature])
            counts = values.value_counts()
            rare = counts.index[counts < threshold]
            mask = values.isin(rare).to_numpy()
            any_pooled |= mask
            total_levels += len(counts)
            pooled_levels += len(rare)
            rows.append({
                "threshold": threshold, "feature": feature, "scope": "feature",
                "training_rows": len(train), "category_levels": len(counts),
                "pooled_levels": len(rare), "retained_levels": len(counts) - len(rare),
                "pooled_rows": int(mask.sum()), "pooled_row_fraction": float(mask.mean()) if len(mask) else 0.0,
            })
            for category, count in counts.sort_index().items():
                category_rows.append({"threshold": threshold, "feature": feature,
                                      "category": category, "training_count": int(count),
                                      "pooled": bool(count < threshold)})
        rows.append({
            "threshold": threshold, "feature": "__ALL_FEATURES__", "scope": "any_feature",
            "training_rows": len(train), "category_levels": total_levels,
            "pooled_levels": pooled_levels, "retained_levels": total_levels - pooled_levels,
            "pooled_rows": int(any_pooled.sum()),
            "pooled_row_fraction": float(any_pooled.mean()) if len(train) else 0.0,
        })
    return pd.DataFrame(rows), pd.DataFrame(category_rows, columns=[
        "threshold", "feature", "category", "training_count", "pooled"
    ])


def prepare_pooled_splits(splits, config, output_dir, *, resume=False):
    """Persist one mapping per experiment, shared by every variant/model/seed."""
    if not pooling_settings(config)["enabled"]:
        return splits
    mapping = fit_pooling(splits.train, config)
    path = Path(output_dir) / MAPPING_FILE
    if resume:
        if not path.is_file() or json.loads(path.read_text(encoding="utf-8")) != mapping:
            raise ValueError("Saved rare-category mapping does not match training rows/settings")
    else:
        atomic_write_json(path, mapping)
    sensitivity, categories = pooling_sensitivity(splits.train, config)
    atomic_write_csv(Path(output_dir) / "rare_category_sensitivity.csv", sensitivity)
    atomic_write_csv(Path(output_dir) / "rare_category_counts.csv", categories)
    atomic_write_json(Path(output_dir) / "rare_category_sensitivity_manifest.json", {
        "fit_population": "real_train_only", "thresholds": mapping["sensitivity_thresholds"],
        "selected_threshold": mapping["min_count"], "gan_performance_evaluated": False,
        "interpretation": "preprocessing_impact_only_not_evidence_of_GAN_robustness",
    })
    return DataSplits(*(transform_pooling(getattr(splits, part), mapping)
                        for part in ("train", "audit", "val", "test")), indices=splits.indices)


def apply_saved_pooling(frame, run_dir, config=None):
    """Reevaluation must load the training mapping, never fit a new one."""
    if config is None:
        config = json.loads((Path(run_dir) / "config.json").read_text(encoding="utf-8"))
    settings = pooling_settings(config)
    if not settings["enabled"]:
        return frame.copy(deep=True)
    mapping = json.loads((Path(run_dir) / MAPPING_FILE).read_text(encoding="utf-8"))
    if mapping.get("version") != 1 or any(mapping.get(key) != value for key, value in settings.items()):
        raise ValueError("Saved rare-category mapping is incompatible with the frozen configuration")
    return transform_pooling(frame, mapping)


def override_pooling_threshold(config, threshold):
    if threshold is not None:
        config.setdefault("rare_categories", {}).update(enabled=True, min_count=threshold)
    pooling_settings(config)
