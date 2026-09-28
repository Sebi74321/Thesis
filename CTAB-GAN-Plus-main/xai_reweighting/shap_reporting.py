"""Reporting normalization, deliberately separate from retraining signals."""

import numpy as np
import pandas as pd


def importance_shares(values: pd.Series) -> pd.Series:
    """Convert nonnegative feature magnitudes to sum-to-one shares.

    Works for raw magnitudes, legacy max-scaled values and already-normalized
    shares. Empty/all-zero vectors remain zero; never mutate caller data.
    """
    clean = pd.to_numeric(values, errors="raise").fillna(0.0).astype(float)
    if not np.isfinite(clean.to_numpy()).all() or (clean < 0).any():
        raise ValueError("SHAP importance shares require finite, nonnegative magnitudes")
    maximum = float(clean.max()) if len(clean) else 0.0
    if maximum == 0:
        return clean * 0.0
    scaled = clean / maximum
    return scaled / float(scaled.sum())
