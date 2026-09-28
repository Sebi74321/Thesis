"""Standalone duration diagnostics. Never used to transform GAN training/output."""

from pathlib import Path

import numpy as np
import pandas as pd

from .io_utils import atomic_write_csv, atomic_write_json

FEATURE = "pre_icu_los_days"
TOLERANCE_SECONDS = 0.001  # Accommodates day values serialized to nine decimals.
GRIDS = {"second": 1.0, "minute": 60.0, "five_minutes": 300.0,
         "hour": 3600.0, "day": 86400.0}
REGIONS = ["negative", "exact_zero", "(0, 5] minutes", "(5, 60] minutes",
           "(1, 24] hours", ">24 hours", "missing", "nonfinite_or_invalid"]


def _numeric(series):
    return pd.to_numeric(series, errors="coerce").to_numpy(dtype=float, na_value=np.nan)


def resolution_summary(series, source, evaluation_split):
    values = _numeric(series)
    finite = values[np.isfinite(values)]
    nonzero = finite[finite != 0]
    rows = []
    for grid, step in GRIDS.items():
        def alignment(array):
            if not len(array):
                return np.nan
            seconds = array * 86400.0
            distance = np.abs(seconds - np.rint(seconds / step) * step)
            return float(np.mean(distance <= TOLERANCE_SECONDS))
        rows.append({
            "feature": FEATURE, "source": source, "evaluation_split": evaluation_split,
            "grid": grid, "step_seconds": step, "tolerance_seconds": TOLERANCE_SECONDS,
            "rows": len(series), "finite_rows": len(finite), "nonzero_finite_rows": len(nonzero),
            "grid_aligned_fraction": alignment(finite),
            "nonzero_grid_aligned_fraction": alignment(nonzero),
            "zero_fraction": float(np.mean(values == 0)) if len(values) else np.nan,
            "negative_fraction": float(np.mean(np.isfinite(values) & (values < 0))) if len(values) else np.nan,
        })
    return rows


def interval_frequencies(series, *, rounded_minutes=False):
    """Exhaustive disjoint bins; denominators include missing/invalid rows."""
    values = _numeric(series).copy()
    missing = series.isna().to_numpy()
    finite = np.isfinite(values)
    if rounded_minutes:
        values[finite] = np.rint(values[finite] * 1440.0) / 1440.0
    minutes = values * 1440.0
    for edge in (5.0, 60.0, 1440.0):
        near = np.isclose(minutes, edge, atol=TOLERANCE_SECONDS / 60, rtol=0)
        minutes[near] = edge
    masks = [finite & (minutes < 0), finite & (minutes == 0),
             finite & (minutes > 0) & (minutes <= 5),
             finite & (minutes > 5) & (minutes <= 60),
             finite & (minutes > 60) & (minutes <= 1440),
             finite & (minutes > 1440), missing, ~finite & ~missing]
    return {region: (int(mask.sum()), float(mask.mean()) if len(mask) else np.nan)
            for region, mask in zip(REGIONS, masks)}


def save_resolution_diagnostics(real_train, real_eval, synthetic_paths, output_dir,
                                *, evaluation_split):
    """Compare saved variants to the matching real holdout, one CSV at a time.

    Rounded views use copies of BOTH populations. They are counterfactual
    diagnostics, not corrections or replacements for authoritative metrics.
    """
    if FEATURE not in real_train or FEATURE not in real_eval:
        return []
    output_dir = Path(output_dir)
    grid_rows = resolution_summary(real_train[FEATURE], "REAL_TRAIN", "train")
    grid_rows += resolution_summary(real_eval[FEATURE], "REAL", evaluation_split)
    interval_rows = []
    views = {"as_saved": False, "minute_rounded_diagnostic_only": True}
    references = {view: interval_frequencies(real_eval[FEATURE], rounded_minutes=rounding)
                  for view, rounding in views.items()}

    def add_intervals(series, source):
        for view, rounding in views.items():
            frequencies = interval_frequencies(series, rounded_minutes=rounding)
            for order, region in enumerate(REGIONS):
                count, frequency = frequencies[region]
                real_count, reference = references[view][region]
                interval_rows.append({
                    "feature": FEATURE, "source": source, "evaluation_split": evaluation_split,
                    "view": view, "region": region, "region_order": order,
                    "rows": len(series), "count": count, "frequency": frequency,
                    "real_rows": len(real_eval), "real_count": real_count,
                    "real_frequency": reference, "frequency_gap_vs_real": frequency - reference,
                    "absolute_frequency_gap": abs(frequency - reference),
                })

    add_intervals(real_eval[FEATURE], "REAL")
    included = []
    for variant, path in synthetic_paths.items():
        if not Path(path).is_file():
            continue
        synthetic = pd.read_csv(path, usecols=[FEATURE])[FEATURE]
        grid_rows += resolution_summary(synthetic, variant, evaluation_split)
        add_intervals(synthetic, variant)
        included.append(variant)
    artifacts = ["duration_resolution.csv", "duration_interval_frequencies.csv", "duration_diagnostics_manifest.json"]
    atomic_write_csv(output_dir / artifacts[0], pd.DataFrame(grid_rows))
    atomic_write_csv(output_dir / artifacts[1], pd.DataFrame(interval_rows))
    atomic_write_json(output_dir / artifacts[2], {
        "feature": FEATURE, "unit": "days", "evaluation_split": evaluation_split,
        "variants": included, "training_resolution_source": "real_train_only",
        "rounding_applied_to_datasets": False, "rounding_rule": "numpy.rint; nearest minute, ties to even",
        "boundary_tolerance_seconds": TOLERANCE_SECONDS,
        "denominator": "all rows for intervals; finite rows for alignment",
        "interpretation": "Compare as-saved interval mass before attributing exact-value gaps to generation failure. "
                          "Rounded views are diagnostic only; alignment does not prove acquisition resolution.",
    })
    return artifacts
