"""Deterministic binary-class balancing for utility evaluations."""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd


def replace_legacy_gender_task(
    tasks: list[dict[str, Any]], mortality_target: str
) -> list[dict[str, Any]]:
    """Upgrade saved pre-change run configs when evaluation is rerun."""
    resolved = []
    for task in tasks:
        item = dict(task)
        if item.get("name") == "gender" and item.get("target_col") == "gender":
            item.update(
                {
                    "name": "mortality_balanced",
                    "balance": "balanced",
                    "task_type": "classification",
                    "target_col": mortality_target,
                    "positive_label": "1",
                }
            )
        resolved.append(item)
    return resolved


def normalize_balance_mode(balance: str | None) -> str:
    mode = str(balance or "imbalanced").strip().lower()
    mode = {
        "natural": "imbalanced",
        "original": "imbalanced",
        "original_prevalence": "imbalanced",
        "none": "imbalanced",
        "50_50": "balanced",
        "50/50": "balanced",
    }.get(mode, mode)
    if mode not in {"imbalanced", "balanced"}:
        raise ValueError(
            f"Unknown utility balance mode {balance!r}; expected imbalanced or balanced"
        )
    return mode


def positive_fraction(
    frame: pd.DataFrame, target_col: str, positive_label: Any
) -> float:
    if target_col not in frame:
        raise ValueError(f"Utility frame is missing target column {target_col!r}")
    labels = frame[target_col].astype("string").fillna("<NA>")
    positive = str(positive_label)
    unique = set(labels.unique())
    if len(unique) != 2:
        raise ValueError(
            f"Utility balancing requires a binary target; {target_col!r} has {len(unique)} classes"
        )
    if positive not in unique:
        raise ValueError(
            f"Configured positive label {positive!r} is absent from {target_col!r}"
        )
    return float((labels == positive).mean())


def target_positive_fraction(
    reference: pd.DataFrame,
    target_col: str,
    positive_label: Any,
    balance: str | None,
) -> float:
    if normalize_balance_mode(balance) == "balanced":
        return 0.5
    return positive_fraction(reference, target_col, positive_label)


def allocate_positive_count(n: int, fraction: float) -> int:
    if n < 0:
        raise ValueError("Sample size cannot be negative")
    if not 0.0 <= float(fraction) <= 1.0:
        raise ValueError("Positive fraction must lie within [0, 1]")
    return int(np.floor(n * float(fraction) + 0.5))


def sample_binary_counts(
    frame: pd.DataFrame,
    n: int,
    target_col: str,
    positive_label: Any,
    positive_count: int,
    seed: int,
) -> pd.DataFrame:
    """Sample exact binary class counts, replacing only when a pool is too small."""
    if n < 0 or not 0 <= positive_count <= n:
        raise ValueError("Invalid binary sample counts")
    if n == 0:
        return frame.iloc[0:0].copy()
    labels = frame[target_col].astype("string").fillna("<NA>")
    positive = str(positive_label)
    unique = set(labels.unique())
    if len(unique) != 2 or positive not in unique:
        raise ValueError(
            f"Utility balancing requires binary {target_col!r} containing positive label {positive!r}"
        )
    pieces = []
    for offset, (is_positive, size) in enumerate(
        ((True, positive_count), (False, n - positive_count))
    ):
        if size == 0:
            continue
        pool = frame.loc[(labels == positive) == is_positive]
        if pool.empty:
            class_name = "positive" if is_positive else "negative"
            raise ValueError(f"Cannot sample {size} {class_name} rows from an empty class")
        pieces.append(
            pool.sample(
                n=size,
                replace=size > len(pool),
                random_state=seed + offset,
            )
        )
    return (
        pd.concat(pieces, axis=0)
        .sample(frac=1.0, random_state=seed + 997)
        .reset_index(drop=True)
    )


def resample_binary_fraction(
    frame: pd.DataFrame,
    n: int,
    target_col: str,
    positive_label: Any,
    fraction: float,
    seed: int,
) -> pd.DataFrame:
    return sample_binary_counts(
        frame,
        n,
        target_col,
        positive_label,
        allocate_positive_count(n, fraction),
        seed,
    )


def compose_fixed_prevalence_mixture(
    real: pd.DataFrame,
    synthetic: pd.DataFrame,
    real_count: int,
    synthetic_count: int,
    target_col: str,
    positive_label: Any,
    fraction: float,
    seed: int,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Create source portions with an exact combined positive-class count."""
    total = real_count + synthetic_count
    total_positive = allocate_positive_count(total, fraction)
    if total == 0:
        return real.iloc[0:0].copy(), synthetic.iloc[0:0].copy()

    real_positive = allocate_positive_count(real_count, total_positive / total)
    real_positive = max(0, min(real_count, real_positive, total_positive))
    synthetic_positive = total_positive - real_positive
    if synthetic_positive > synthetic_count:
        shift = synthetic_positive - synthetic_count
        real_positive += shift
        synthetic_positive -= shift

    real_part = sample_binary_counts(
        real, real_count, target_col, positive_label, real_positive, seed
    )
    synthetic_part = sample_binary_counts(
        synthetic,
        synthetic_count,
        target_col,
        positive_label,
        synthetic_positive,
        seed + 11,
    )
    achieved = int(
        (real_part[target_col].astype("string") == str(positive_label)).sum()
        + (synthetic_part[target_col].astype("string") == str(positive_label)).sum()
    )
    if achieved != total_positive:
        raise RuntimeError("Fixed-prevalence mixture did not achieve its class allocation")
    return real_part, synthetic_part


def class_balance_metadata(
    real_part: pd.DataFrame,
    synthetic_part: pd.DataFrame,
    target_col: str,
    positive_label: Any,
    target_fraction: float,
) -> dict[str, float | int]:
    positive = str(positive_label)

    def count(frame: pd.DataFrame) -> int:
        if frame.empty:
            return 0
        return int((frame[target_col].astype("string") == positive).sum())

    real_positive = count(real_part)
    synthetic_positive = count(synthetic_part)
    total = len(real_part) + len(synthetic_part)
    training_positive = real_positive + synthetic_positive
    return {
        "target_positive_fraction": float(target_fraction),
        "real_training_positive_rows": real_positive,
        "synthetic_training_positive_rows": synthetic_positive,
        "training_positive_rows": training_positive,
        "training_negative_rows": total - training_positive,
        "training_positive_fraction": (
            float(training_positive / total) if total else float("nan")
        ),
    }


def source_prevalence_metadata(
    real: pd.DataFrame,
    synthetic: pd.DataFrame,
    real_count: int,
    synthetic_count: int,
    target_col: str,
    positive_label: Any,
    achieved_fraction: float,
) -> dict[str, float | bool]:
    """Expose the prevalence that an unadjusted source mixture would contain."""
    real_rate = positive_fraction(real, target_col, positive_label)
    synthetic_rate = positive_fraction(synthetic, target_col, positive_label)
    total = real_count + synthetic_count
    unadjusted = (
        (real_count * real_rate + synthetic_count * synthetic_rate) / total
        if total
        else float("nan")
    )
    adjustment = float(achieved_fraction - unadjusted)
    return {
        "source_real_positive_fraction": real_rate,
        "source_synthetic_positive_fraction": synthetic_rate,
        "unadjusted_training_positive_fraction": float(unadjusted),
        "positive_fraction_adjustment": adjustment,
        "class_balance_resampling_applied": bool(not np.isclose(adjustment, 0.0)),
    }
