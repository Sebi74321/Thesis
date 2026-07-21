"""Leakage-safe, reproducible dataset splitting."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split


@dataclass(frozen=True)
class DataSplits:
    train: pd.DataFrame
    audit: pd.DataFrame
    val: pd.DataFrame
    test: pd.DataFrame
    indices: Dict[str, List[int]]


def create_data_splits(
    df: pd.DataFrame,
    target_col: str,
    seed: int = 42,
    train_fraction: float = 0.60,
    audit_fraction: float = 0.20,
    val_fraction: float = 0.10,
    test_fraction: float = 0.10,
) -> DataSplits:
    fractions = np.array([train_fraction, audit_fraction, val_fraction, test_fraction], dtype=float)
    if not np.isclose(fractions.sum(), 1.0) or np.any(fractions <= 0):
        raise ValueError("Split fractions must be positive and sum to 1")
    if target_col not in df:
        raise KeyError(f"Target column '{target_col}' is missing")

    all_idx = np.arange(len(df))
    train_idx, remainder_idx = train_test_split(
        all_idx, train_size=train_fraction, random_state=seed, stratify=df[target_col]
    )
    audit_share = audit_fraction / (audit_fraction + val_fraction + test_fraction)
    audit_idx, final_idx = train_test_split(
        remainder_idx,
        train_size=audit_share,
        random_state=seed + 1,
        stratify=df.iloc[remainder_idx][target_col],
    )
    val_share = val_fraction / (val_fraction + test_fraction)
    val_idx, test_idx = train_test_split(
        final_idx,
        train_size=val_share,
        random_state=seed + 2,
        stratify=df.iloc[final_idx][target_col],
    )

    parts = {
        "train": sorted(int(x) for x in train_idx),
        "audit": sorted(int(x) for x in audit_idx),
        "val": sorted(int(x) for x in val_idx),
        "test": sorted(int(x) for x in test_idx),
    }
    flattened = [x for values in parts.values() for x in values]
    if len(flattened) != len(df) or len(set(flattened)) != len(df):
        raise RuntimeError("Data split indices are overlapping or incomplete")

    def take(name: str) -> pd.DataFrame:
        return df.iloc[parts[name]].copy(deep=True).reset_index(drop=True)

    return DataSplits(take("train"), take("audit"), take("val"), take("test"), parts)
