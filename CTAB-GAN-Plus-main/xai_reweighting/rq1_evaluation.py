"""RQ1-specific rare-region and real-reference evaluation helpers."""

from __future__ import annotations

from typing import Iterable

import numpy as np
import pandas as pd

from .evaluation import _cat, evaluate_fidelity_and_tails


def _scale(values: np.ndarray) -> float:
    q25, q75 = np.quantile(values, [0.25, 0.75])
    value = float(q75 - q25)
    if not np.isfinite(value) or value <= 0:
        value = float(np.std(values))
    return value if np.isfinite(value) and value > 0 else 1.0


def _rare_flags(
    frame: pd.DataFrame,
    reference: pd.DataFrame,
    categorical_cols: Iterable[str],
    continuous_cols: Iterable[str],
    rare_threshold: float,
) -> pd.DataFrame:
    flags = {}
    for column in continuous_cols:
        reference_values = pd.to_numeric(reference[column], errors="coerce").dropna()
        values = pd.to_numeric(frame[column], errors="coerce")
        if reference_values.empty:
            continue
        lower, upper = reference_values.quantile([0.05, 0.95])
        flags[f"{column}:tail"] = (values <= lower) | (values >= upper)
    for column in categorical_cols:
        frequencies = _cat(reference[column]).value_counts(normalize=True)
        rare = set(frequencies[frequencies <= rare_threshold].index)
        if rare:
            flags[f"{column}:rare"] = _cat(frame[column]).isin(rare)
    return pd.DataFrame(flags, index=frame.index).fillna(False).astype(bool)


def evaluate_rq1_specific(
    reference: pd.DataFrame,
    real_eval: pd.DataFrame,
    synthetic: pd.DataFrame,
    target_col: str,
    categorical_cols: Iterable[str],
    continuous_cols: Iterable[str],
    rare_threshold: float = 0.05,
) -> tuple[dict[str, float | str], pd.DataFrame]:
    """Evaluate fixed-reference tails, rare categories and rare-outcome subgroups."""
    categorical_cols = list(categorical_cols)
    continuous_cols = list(continuous_cols)
    records: list[dict] = []

    for column in continuous_cols:
        ref = pd.to_numeric(reference[column], errors="coerce").dropna().to_numpy()
        real = pd.to_numeric(real_eval[column], errors="coerce").dropna().to_numpy()
        syn = pd.to_numeric(synthetic[column], errors="coerce").dropna().to_numpy()
        if not len(ref) or not len(real) or not len(syn):
            continue
        scale = _scale(ref)
        for quantile in (0.05, 0.95, 0.99):
            threshold = float(np.quantile(ref, quantile))
            lower = quantile == 0.05
            real_mass = float(np.mean(real <= threshold) if lower else np.mean(real >= threshold))
            syn_mass = float(np.mean(syn <= threshold) if lower else np.mean(syn >= threshold))
            records.append(
                {
                    "feature": column,
                    "region": f"{'lower' if lower else 'upper'}_q{int(quantile * 100):02d}",
                    "kind": "continuous_tail",
                    "reference_threshold": threshold,
                    "real_mass": real_mass,
                    "synthetic_mass": syn_mass,
                    "mass_error": abs(syn_mass - real_mass),
                    "quantile_error_scaled": abs(float(np.quantile(syn, quantile)) - float(np.quantile(real, quantile))) / scale,
                }
            )

    for column in categorical_cols:
        ref_frequency = _cat(reference[column]).value_counts(normalize=True)
        real_frequency = _cat(real_eval[column]).value_counts(normalize=True)
        syn_frequency = _cat(synthetic[column]).value_counts(normalize=True)
        for value in ref_frequency[ref_frequency <= rare_threshold].index:
            real_mass = float(real_frequency.get(value, 0.0))
            syn_mass = float(syn_frequency.get(value, 0.0))
            records.append(
                {
                    "feature": column,
                    "region": str(value),
                    "kind": "rare_category",
                    "reference_threshold": float(ref_frequency[value]),
                    "real_mass": real_mass,
                    "synthetic_mass": syn_mass,
                    "mass_error": abs(syn_mass - real_mass),
                    "quantile_error_scaled": np.nan,
                }
            )

    real_target = _cat(real_eval[target_col])
    syn_target = _cat(synthetic[target_col])
    reference_target = _cat(reference[target_col])
    rare_label = reference_target.value_counts().idxmin()
    metrics: dict[str, float | str] = {"rq1_rare_outcome_label": str(rare_label)}
    real_rare = real_eval.loc[real_target == rare_label]
    syn_rare = synthetic.loc[syn_target == rare_label]
    metrics["rq1_real_rare_outcome_rows"] = float(len(real_rare))
    metrics["rq1_synthetic_rare_outcome_rows"] = float(len(syn_rare))
    if len(real_rare) and len(syn_rare):
        conditional_categorical = [column for column in categorical_cols if column != target_col]
        conditional, _ = evaluate_fidelity_and_tails(
            real_rare,
            syn_rare,
            conditional_categorical,
            continuous_cols,
            rare_threshold=rare_threshold,
        )
        for key, value in conditional.items():
            metrics[f"rq1_outcome_conditional_{key}"] = value

    real_flags = _rare_flags(
        real_eval, reference, categorical_cols, continuous_cols, rare_threshold
    )
    syn_flags = _rare_flags(
        synthetic, reference, categorical_cols, continuous_cols, rare_threshold
    )
    real_joint = float((real_flags.sum(axis=1) >= 2).mean()) if len(real_flags.columns) else 0.0
    syn_joint = float((syn_flags.sum(axis=1) >= 2).mean()) if len(syn_flags.columns) else 0.0
    metrics.update(
        {
            "rq1_joint_rare_region_mass_real": real_joint,
            "rq1_joint_rare_region_mass_synthetic": syn_joint,
            "rq1_joint_rare_region_mass_error": abs(syn_joint - real_joint),
            "rq1_rare_region_count": float(len(real_flags.columns)),
        }
    )

    details = pd.DataFrame(records)
    if not details.empty:
        tails = details[details["kind"] == "continuous_tail"]
        rare = details[details["kind"] == "rare_category"]
        metrics["rq1_mean_fixed_tail_mass_error"] = float(tails["mass_error"].mean())
        metrics["rq1_mean_fixed_tail_quantile_error_scaled"] = float(
            tails["quantile_error_scaled"].mean()
        )
        metrics["rq1_mean_rare_category_frequency_error"] = float(rare["mass_error"].mean())
    for key in (
        "rq1_mean_fixed_tail_mass_error",
        "rq1_mean_fixed_tail_quantile_error_scaled",
        "rq1_mean_rare_category_frequency_error",
    ):
        if not np.isfinite(float(metrics.get(key, np.nan))):
            metrics[key] = 0.0
    return metrics, details


def real_real_reference(
    reference: pd.DataFrame,
    real_eval: pd.DataFrame,
    target_col: str,
    categorical_cols: Iterable[str],
    continuous_cols: Iterable[str],
    seed: int,
) -> tuple[dict[str, float | str], pd.DataFrame]:
    """Estimate the finite-sample fidelity floor from two real bootstraps."""
    first = real_eval.sample(n=len(real_eval), replace=True, random_state=seed).reset_index(drop=True)
    second = real_eval.sample(n=len(real_eval), replace=True, random_state=seed + 1).reset_index(drop=True)
    metrics, _ = evaluate_fidelity_and_tails(first, second, categorical_cols, continuous_cols)
    specific, details = evaluate_rq1_specific(
        reference, first, second, target_col, categorical_cols, continuous_cols
    )
    metrics.update(specific)
    return metrics, details
