"""Uniform and weighted augmentation helpers."""

from __future__ import annotations

from typing import Dict, Tuple

import numpy as np
import pandas as pd


def _sample_augmentation(real_train, gamma, seed, probabilities):
    if gamma < 0:
        raise ValueError("gamma must be non-negative")
    n_extra = int(gamma * len(real_train))
    if n_extra == 0:
        return real_train.copy(deep=True).reset_index(drop=True), {}
    rng = np.random.default_rng(seed)
    selected = rng.choice(len(real_train), size=n_extra, replace=True, p=probabilities)
    extra = real_train.iloc[selected].copy(deep=True)
    retrain = pd.concat([real_train.copy(deep=True), extra], ignore_index=True)
    values, counts = np.unique(selected, return_counts=True)
    return retrain, {int(i): int(c) for i, c in zip(values, counts)}


def create_uniform_augmentation(
    real_train: pd.DataFrame, gamma: float = 0.25, seed: int = 42
) -> Tuple[pd.DataFrame, Dict[int, int]]:
    return _sample_augmentation(real_train, gamma, seed, None)


def create_weighted_augmentation(
    real_train: pd.DataFrame, row_weights, gamma: float = 0.25, seed: int = 42
) -> Tuple[pd.DataFrame, Dict[int, int]]:
    weights = np.asarray(row_weights, dtype=float)
    if len(weights) != len(real_train):
        raise ValueError("row_weights length must equal real_train length")
    if not np.all(np.isfinite(weights)) or np.any(weights < 0) or weights.sum() <= 0:
        raise ValueError("row_weights must be finite, non-negative, and have a positive sum")
    return _sample_augmentation(real_train, gamma, seed, weights / weights.sum())
