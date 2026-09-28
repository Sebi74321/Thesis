"""Diagnostics for explaining a strong real-vs-synthetic baseline detector."""

from __future__ import annotations

from typing import Callable, Iterable

import numpy as np
import pandas as pd
from scipy.spatial.distance import jensenshannon
from scipy.stats import ks_2samp, wasserstein_distance

from .detector import train_detector
from .scoring import MISSING


def _categorical(series: pd.Series) -> pd.Series:
    return series.astype("object").where(series.notna(), MISSING).astype(str)


def _scale(values: np.ndarray) -> float:
    q25, q75 = np.quantile(values, [0.25, 0.75])
    scale = float(q75 - q25)
    if not np.isfinite(scale) or scale <= 0:
        scale = float(np.std(values))
    return scale if np.isfinite(scale) and scale > 0 else 1.0


def _js_distance(real: pd.Series, synthetic: pd.Series) -> float:
    if real.empty or synthetic.empty:
        return 0.0 if real.empty and synthetic.empty else 1.0
    categories = sorted(set(real.unique()).union(synthetic.unique()))
    real_frequency = real.value_counts(normalize=True)
    synthetic_frequency = synthetic.value_counts(normalize=True)
    return float(
        jensenshannon(
            [float(real_frequency.get(value, 0.0)) for value in categories],
            [float(synthetic_frequency.get(value, 0.0)) for value in categories],
            base=2.0,
        )
    )


def baseline_detector_diagnostics(
    real_audit: pd.DataFrame,
    synthetic_audit: pd.DataFrame,
    components: pd.DataFrame,
    target_col: str,
    categorical_cols: Iterable[str],
    continuous_cols: Iterable[str],
    *,
    top_n: int = 10,
    seed: int = 42,
    n_estimators: int = 300,
    n_jobs: int = -1,
    detector_fn: Callable = train_detector,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Rank global SHAP drivers and diagnose their gaps within each outcome.

    The conditional detectors exclude ``target_col``. Their AUC therefore measures
    distinguishability that remains after mortality prevalence is held constant.
    """
    if target_col not in real_audit or target_col not in synthetic_audit:
        raise ValueError(f"Target column is required in both audit datasets: {target_col}")
    if top_n < 1:
        raise ValueError("diagnostics top_n must be at least one")

    ranking = components.copy()
    ranking = ranking.sort_values(
        ["shap_raw", "feature"], ascending=[False, True], kind="mergesort"
    ).reset_index(drop=True)
    ranking["shap_rank"] = np.arange(1, len(ranking) + 1)
    candidates = ranking.loc[ranking["feature"] != target_col, "feature"].head(top_n).tolist()
    ranking["selected_for_conditional_diagnostic"] = ranking["feature"].isin(candidates)

    categorical = set(categorical_cols)
    continuous = set(continuous_cols)
    real_target = _categorical(real_audit[target_col])
    synthetic_target = _categorical(synthetic_audit[target_col])
    outcomes = sorted(set(real_target.unique()).union(synthetic_target.unique()))
    feature_rows: list[dict] = []
    category_rows: list[dict] = []
    detector_rows: list[dict] = []

    detector_columns = [column for column in real_audit.columns if column != target_col]
    detector_categorical = [column for column in categorical if column != target_col]
    for outcome in outcomes:
        real_group = real_audit.loc[real_target == outcome]
        synthetic_group = synthetic_audit.loc[synthetic_target == outcome]
        summary = {
            "outcome": outcome,
            "real_rows": int(len(real_group)),
            "synthetic_rows": int(len(synthetic_group)),
            "real_frequency": float(len(real_group) / len(real_audit)),
            "synthetic_frequency": float(len(synthetic_group) / len(synthetic_audit)),
            "frequency_error": float(
                len(synthetic_group) / len(synthetic_audit) - len(real_group) / len(real_audit)
            ),
        }
        if min(len(real_group), len(synthetic_group)) < 4 or not detector_columns:
            summary.update(
                {
                    "status": "insufficient_rows",
                    "detector_auc": np.nan,
                    "detector_average_precision": np.nan,
                    "detector_accuracy": np.nan,
                }
            )
        else:
            result = detector_fn(
                real_group.loc[:, detector_columns],
                synthetic_group.loc[:, detector_columns],
                detector_categorical,
                seed=seed,
                n_estimators=n_estimators,
                compute_shap=False,
                n_jobs=n_jobs,
            )
            summary.update({"status": "ok", **result.metrics})
        detector_rows.append(summary)

        for rank, feature in enumerate(candidates, start=1):
            base = {
                "outcome": outcome,
                "feature": feature,
                "conditional_rank": rank,
                "shap_raw": float(
                    ranking.loc[ranking["feature"] == feature, "shap_raw"].iloc[0]
                ),
                "real_rows": int(len(real_group)),
                "synthetic_rows": int(len(synthetic_group)),
            }
            if feature in continuous:
                real_numeric = pd.to_numeric(real_group[feature], errors="coerce")
                synthetic_numeric = pd.to_numeric(synthetic_group[feature], errors="coerce")
                real_values = real_numeric.dropna().to_numpy(dtype=float)
                synthetic_values = synthetic_numeric.dropna().to_numpy(dtype=float)
                row = {
                    **base,
                    "kind": "continuous",
                    "real_missing_rate": float(real_numeric.isna().mean()),
                    "synthetic_missing_rate": float(synthetic_numeric.isna().mean()),
                }
                if len(real_values) and len(synthetic_values):
                    scale = _scale(real_values)
                    row.update(
                        {
                            "real_mean": float(np.mean(real_values)),
                            "synthetic_mean": float(np.mean(synthetic_values)),
                            "real_median": float(np.median(real_values)),
                            "synthetic_median": float(np.median(synthetic_values)),
                            "real_q05": float(np.quantile(real_values, 0.05)),
                            "synthetic_q05": float(np.quantile(synthetic_values, 0.05)),
                            "real_q95": float(np.quantile(real_values, 0.95)),
                            "synthetic_q95": float(np.quantile(synthetic_values, 0.95)),
                            "wasserstein_scaled": float(
                                wasserstein_distance(real_values, synthetic_values) / scale
                            ),
                            "ks": float(ks_2samp(real_values, synthetic_values).statistic),
                        }
                    )
                feature_rows.append(row)
            elif feature in categorical:
                real_values = _categorical(real_group[feature])
                synthetic_values = _categorical(synthetic_group[feature])
                real_frequency = real_values.value_counts(normalize=True)
                synthetic_frequency = synthetic_values.value_counts(normalize=True)
                values = sorted(set(real_values.unique()).union(synthetic_values.unique()))
                gaps = [
                    abs(float(real_frequency.get(value, 0.0) - synthetic_frequency.get(value, 0.0)))
                    for value in values
                ]
                feature_rows.append(
                    {
                        **base,
                        "kind": "categorical",
                        "jensen_shannon": _js_distance(real_values, synthetic_values),
                        "max_category_frequency_error": max(gaps, default=0.0),
                    }
                )
                for value in values:
                    real_p = float(real_frequency.get(value, 0.0))
                    synthetic_p = float(synthetic_frequency.get(value, 0.0))
                    category_rows.append(
                        {
                            "outcome": outcome,
                            "feature": feature,
                            "conditional_rank": rank,
                            "category": value,
                            "real_frequency": real_p,
                            "synthetic_frequency": synthetic_p,
                            "frequency_error": synthetic_p - real_p,
                            "absolute_frequency_error": abs(synthetic_p - real_p),
                        }
                    )

    return (
        ranking,
        pd.DataFrame(feature_rows),
        pd.DataFrame(category_rows),
        pd.DataFrame(detector_rows),
    )
