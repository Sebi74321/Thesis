"""Artifact-oriented diagnostics for the features selected by A2/A3/A4/A5."""

from __future__ import annotations

from typing import Callable, Iterable, Mapping

import numpy as np
import pandas as pd
from scipy.stats import chi2_contingency, ks_2samp, wasserstein_distance

from .detector import train_detector
from .diagnostics import _categorical, _js_distance, _scale
from .evaluation import evaluate_utility


def top_shap_feature_variant_metrics(
    ranking: pd.DataFrame,
    feature_metrics_by_variant: Mapping[str, pd.DataFrame],
) -> pd.DataFrame:
    """Track validation/test fidelity of A0's top SHAP features across variants.

    Continuous features use IQR-scaled Wasserstein distance and categorical
    features use Jensen-Shannon distance. Both are distribution discrepancies,
    so lower values and negative deltas versus A0 indicate improvement.
    """
    required = {"feature", "shap_rank", "shap_raw"}
    if not required.issubset(ranking.columns):
        raise ValueError(f"ranking must contain {sorted(required)}")
    if "selected_for_conditional_diagnostic" in ranking:
        selected = ranking[ranking["selected_for_conditional_diagnostic"].astype(bool)].copy()
    else:
        selected = ranking.copy()
    selected = selected.sort_values("shap_rank", kind="mergesort")

    rows: list[dict] = []
    for variant, metrics in feature_metrics_by_variant.items():
        if "feature" not in metrics:
            raise ValueError(f"Feature metrics for {variant} do not contain 'feature'")
        indexed = metrics.drop_duplicates("feature").set_index("feature")
        for top in selected.to_dict(orient="records"):
            feature = str(top["feature"])
            if feature not in indexed.index:
                continue
            values = indexed.loc[feature]
            kind = str(values.get("kind", "unknown"))
            if kind == "continuous" or pd.notna(values.get("wasserstein_scaled", np.nan)):
                metric_name = "wasserstein_scaled"
            elif kind == "categorical" or pd.notna(values.get("jensen_shannon", np.nan)):
                metric_name = "jensen_shannon"
            else:
                metric_name = "unavailable"
            discrepancy = (
                float(values.get(metric_name))
                if metric_name != "unavailable" and pd.notna(values.get(metric_name))
                else np.nan
            )
            rows.append(
                {
                    "variant": str(variant),
                    "feature": feature,
                    "kind": kind,
                    "shap_rank": int(top["shap_rank"]),
                    "shap_raw": float(top["shap_raw"]),
                    "discrepancy_metric": metric_name,
                    "distribution_discrepancy": discrepancy,
                }
            )

    result = pd.DataFrame(rows)
    if result.empty:
        return result
    baseline = (
        result[result["variant"] == "A0"]
        .drop_duplicates("feature")
        .set_index("feature")["distribution_discrepancy"]
    )
    result["a0_distribution_discrepancy"] = result["feature"].map(baseline)
    result["delta_discrepancy_vs_A0"] = (
        result["distribution_discrepancy"] - result["a0_distribution_discrepancy"]
    )
    denominator = result["a0_distribution_discrepancy"].replace(0.0, np.nan)
    result["relative_change_vs_A0"] = result["delta_discrepancy_vs_A0"] / denominator
    result["improved_vs_A0"] = result["delta_discrepancy_vs_A0"] < 0
    variant_order = {variant: index for index, variant in enumerate(("A0", "A1", "A2", "A3", "A4", "A5"))}
    result["_variant_order"] = result["variant"].map(variant_order).fillna(len(variant_order))
    return result.sort_values(["shap_rank", "_variant_order", "variant"]).drop(
        columns="_variant_order"
    ).reset_index(drop=True)


def _target_association(frame: pd.DataFrame, feature: str, target: str, categorical: bool) -> float:
    target_values = _categorical(frame[target])
    if target_values.nunique() < 2:
        return 0.0
    if categorical:
        table = pd.crosstab(_categorical(frame[feature]), target_values)
        if min(table.shape) < 2 or table.to_numpy().sum() == 0:
            return 0.0
        chi2 = float(chi2_contingency(table, correction=False)[0])
        denominator = float(table.to_numpy().sum() * min(table.shape[0] - 1, table.shape[1] - 1))
        return float(np.sqrt(chi2 / denominator)) if denominator > 0 else 0.0
    numeric = pd.to_numeric(frame[feature], errors="coerce")
    codes = pd.Series(pd.Categorical(target_values).codes, index=frame.index, dtype=float)
    valid = numeric.notna() & (codes >= 0)
    if valid.sum() < 3 or numeric[valid].nunique() < 2:
        return 0.0
    correlation = numeric[valid].corr(codes[valid])
    return abs(float(correlation)) if np.isfinite(correlation) else 0.0


def _resolution(values: np.ndarray) -> float:
    unique = np.unique(values)
    if len(unique) < 2:
        return 0.0
    differences = np.diff(unique)
    positive = differences[differences > 0]
    return float(np.median(positive)) if len(positive) else 0.0


def _continuous_summary(
    real: pd.Series, synthetic: pd.Series, real_train: pd.Series
) -> tuple[dict, list[dict]]:
    real_numeric = pd.to_numeric(real, errors="coerce")
    syn_numeric = pd.to_numeric(synthetic, errors="coerce")
    train_numeric = pd.to_numeric(real_train, errors="coerce")
    r = real_numeric.dropna().to_numpy(dtype=float)
    s = syn_numeric.dropna().to_numpy(dtype=float)
    train = train_numeric.dropna().to_numpy(dtype=float)
    row = {
        "real_missing_rate": float(real_numeric.isna().mean()),
        "synthetic_missing_rate": float(syn_numeric.isna().mean()),
        "missing_rate_gap": float(syn_numeric.isna().mean() - real_numeric.isna().mean()),
        "real_unique": int(real_numeric.nunique(dropna=True)),
        "synthetic_unique": int(syn_numeric.nunique(dropna=True)),
    }
    spikes: list[dict] = []
    if not len(r) or not len(s) or not len(train):
        return row, spikes
    scale = _scale(r)
    train_min, train_max = float(np.min(train)), float(np.max(train))
    real_frequency = real_numeric.value_counts(normalize=True, dropna=False)
    syn_frequency = syn_numeric.value_counts(normalize=True, dropna=False)
    spike_values = list(dict.fromkeys([*real_frequency.head(5).index, *syn_frequency.head(5).index]))
    for value in spike_values:
        real_p = float(real_frequency.get(value, 0.0))
        syn_p = float(syn_frequency.get(value, 0.0))
        spikes.append(
            {
                "value": value,
                "real_frequency": real_p,
                "synthetic_frequency": syn_p,
                "frequency_gap": syn_p - real_p,
                "absolute_frequency_gap": abs(syn_p - real_p),
            }
        )
    row.update(
        {
            "real_mean": float(np.mean(r)),
            "synthetic_mean": float(np.mean(s)),
            "real_std": float(np.std(r)),
            "synthetic_std": float(np.std(s)),
            "real_min": float(np.min(r)),
            "synthetic_min": float(np.min(s)),
            "real_max": float(np.max(r)),
            "synthetic_max": float(np.max(s)),
            "real_q01": float(np.quantile(r, 0.01)),
            "synthetic_q01": float(np.quantile(s, 0.01)),
            "real_q05": float(np.quantile(r, 0.05)),
            "synthetic_q05": float(np.quantile(s, 0.05)),
            "real_median": float(np.median(r)),
            "synthetic_median": float(np.median(s)),
            "real_q95": float(np.quantile(r, 0.95)),
            "synthetic_q95": float(np.quantile(s, 0.95)),
            "real_q99": float(np.quantile(r, 0.99)),
            "synthetic_q99": float(np.quantile(s, 0.99)),
            "wasserstein_scaled": float(wasserstein_distance(r, s) / scale),
            "ks": float(ks_2samp(r, s).statistic),
            "synthetic_below_train_min_rate": float(np.mean(s < train_min)),
            "synthetic_above_train_max_rate": float(np.mean(s > train_max)),
            "synthetic_out_of_train_range_rate": float(np.mean((s < train_min) | (s > train_max))),
            "real_train_min_spike_rate": float(np.mean(r == train_min)),
            "synthetic_train_min_spike_rate": float(np.mean(s == train_min)),
            "real_train_max_spike_rate": float(np.mean(r == train_max)),
            "synthetic_train_max_spike_rate": float(np.mean(s == train_max)),
            "boundary_spike_gap": float(
                max(abs(np.mean(s == train_min) - np.mean(r == train_min)),
                    abs(np.mean(s == train_max) - np.mean(r == train_max)))
            ),
            "real_resolution": _resolution(r),
            "synthetic_resolution": _resolution(s),
            "unique_ratio_synthetic_to_real": float(len(np.unique(s)) / max(len(np.unique(r)), 1)),
        }
    )
    return row, spikes


def _categorical_summary(real: pd.Series, synthetic: pd.Series) -> tuple[dict, list[dict]]:
    real_values, syn_values = _categorical(real), _categorical(synthetic)
    real_frequency = real_values.value_counts(normalize=True)
    syn_frequency = syn_values.value_counts(normalize=True)
    values = sorted(set(real_frequency.index).union(syn_frequency.index))
    spikes = []
    for value in values:
        real_p, syn_p = float(real_frequency.get(value, 0.0)), float(syn_frequency.get(value, 0.0))
        spikes.append(
            {
                "value": value,
                "real_frequency": real_p,
                "synthetic_frequency": syn_p,
                "frequency_gap": syn_p - real_p,
                "absolute_frequency_gap": abs(syn_p - real_p),
            }
        )
    return {
        "real_missing_rate": float(real.isna().mean()),
        "synthetic_missing_rate": float(synthetic.isna().mean()),
        "missing_rate_gap": float(synthetic.isna().mean() - real.isna().mean()),
        "real_unique": int(real_values.nunique()),
        "synthetic_unique": int(syn_values.nunique()),
        "jensen_shannon": _js_distance(real_values, syn_values),
        "unseen_synthetic_category_rate": float(
            sum(syn_frequency.get(value, 0.0) for value in set(syn_frequency.index) - set(real_frequency.index))
        ),
        "max_category_frequency_gap": max((row["absolute_frequency_gap"] for row in spikes), default=0.0),
        "unique_ratio_synthetic_to_real": float(len(syn_frequency) / max(len(real_frequency), 1)),
    }, spikes


def prioritized_feature_diagnostics(
    real_train: pd.DataFrame,
    real_audit: pd.DataFrame,
    synthetic_audit: pd.DataFrame,
    priorities: Mapping[str, pd.DataFrame],
    target_col: str,
    categorical_cols: Iterable[str],
    continuous_cols: Iterable[str],
    *,
    baseline_detector_auc: float,
    seed: int = 42,
    n_estimators: int = 100,
    n_jobs: int = -1,
    detector_fn: Callable = train_detector,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Explain whether selected priorities reflect gaps or likely measurement artifacts."""
    categorical = set(categorical_cols)
    continuous = set(continuous_cols)
    selected_features = sorted(
        {
            feature
            for priority in priorities.values()
            for feature in priority.loc[priority["selected"].astype(bool), "feature"]
        }
    )
    base_rows, spike_rows, detector_rows, correlation_rows = [], [], [], []
    numeric_correlation = real_audit[list(continuous)].corr().abs() if continuous else pd.DataFrame()
    diagnostic_full = detector_fn(
        real_audit,
        synthetic_audit,
        [column for column in categorical if column in real_audit],
        seed=seed,
        n_estimators=n_estimators,
        compute_shap=False,
        n_jobs=n_jobs,
    )
    diagnostic_full_auc = float(diagnostic_full.metrics["detector_auc"])

    for feature in selected_features:
        kind = "categorical" if feature in categorical else "continuous"
        if kind == "continuous":
            summary, spikes = _continuous_summary(
                real_audit[feature], synthetic_audit[feature], real_train[feature]
            )
            if feature in numeric_correlation:
                correlations = numeric_correlation[feature].drop(labels=[feature], errors="ignore").dropna()
                if len(correlations):
                    partner = correlations.idxmax()
                    summary["max_real_feature_correlation"] = float(correlations[partner])
                    summary["most_correlated_feature"] = partner
                    correlation_rows.extend(
                        {
                            "feature": feature,
                            "other_feature": other,
                            "absolute_real_correlation": float(value),
                        }
                        for other, value in correlations.sort_values(ascending=False).head(5).items()
                    )
        else:
            summary, spikes = _categorical_summary(real_audit[feature], synthetic_audit[feature])
        for spike in spikes:
            spike_rows.append({"feature": feature, "kind": kind, **spike})
        real_association = _target_association(real_audit, feature, target_col, kind == "categorical")
        syn_association = _target_association(synthetic_audit, feature, target_col, kind == "categorical")
        summary.update(
            {
                "feature": feature,
                "kind": kind,
                "real_target_association": real_association,
                "synthetic_target_association": syn_association,
                "target_association_gap": abs(syn_association - real_association),
            }
        )
        base_rows.append(summary)

        univariate = detector_fn(
            real_audit[[feature]], synthetic_audit[[feature]],
            [feature] if feature in categorical else [], seed=seed,
            n_estimators=n_estimators, compute_shap=False, n_jobs=n_jobs,
        )
        remaining = [column for column in real_audit if column != feature]
        without = detector_fn(
            real_audit[remaining], synthetic_audit[remaining],
            [column for column in categorical if column in remaining], seed=seed,
            n_estimators=n_estimators, compute_shap=False, n_jobs=n_jobs,
        )
        detector_rows.append(
            {
                "feature": feature,
                "univariate_detector_auc": float(univariate.metrics["detector_auc"]),
                "detector_auc_without_feature": float(without.metrics["detector_auc"]),
                "detector_auc_drop_when_removed": float(
                    diagnostic_full_auc - without.metrics["detector_auc"]
                ),
                "baseline_detector_auc": float(baseline_detector_auc),
                "diagnostic_full_detector_auc": diagnostic_full_auc,
            }
        )

    base = pd.DataFrame(base_rows)
    detector = pd.DataFrame(detector_rows)
    if len(base):
        base = base.merge(detector, on="feature", how="left")
        base["flag_missingness_gap"] = base["missing_rate_gap"].abs().fillna(0) > 0.02
        base["flag_out_of_range"] = base.get(
            "synthetic_out_of_train_range_rate", pd.Series(0.0, index=base.index)
        ).fillna(0) > 0.01
        base["flag_boundary_spike"] = base.get(
            "boundary_spike_gap", pd.Series(0.0, index=base.index)
        ).fillna(0) > 0.02
        base["flag_low_unique_support"] = base["unique_ratio_synthetic_to_real"].fillna(1) < 0.5
        base["flag_high_univariate_detector_auc"] = base["univariate_detector_auc"].fillna(0.5) >= 0.80
        base["flag_target_association_shift"] = base["target_association_gap"].fillna(0) > 0.10
        flag_columns = [column for column in base if column.startswith("flag_")]
        base["artifact_flag_count"] = base[flag_columns].sum(axis=1).astype(int)
        base["requires_artifact_review"] = base["artifact_flag_count"] > 0

    variant_rows = []
    for variant, priority in priorities.items():
        selected = priority.loc[priority["selected"].astype(bool)].copy()
        selected["priority_rank"] = selected["priority"].rank(method="first", ascending=False).astype(int)
        for row in selected.to_dict(orient="records"):
            diagnostic = base.loc[base["feature"] == row["feature"]]
            variant_rows.append(
                {"variant": variant, **row, **(diagnostic.iloc[0].to_dict() if len(diagnostic) else {})}
            )
    return (
        pd.DataFrame(variant_rows),
        pd.DataFrame(spike_rows),
        detector,
        pd.DataFrame(correlation_rows),
    )


def feature_exclusion_sensitivity(
    real_audit: pd.DataFrame,
    synthetic_audit: pd.DataFrame,
    real_eval: pd.DataFrame,
    synthetic_eval: pd.DataFrame,
    priorities: Mapping[str, pd.DataFrame],
    feature_groups: Mapping[str, str],
    categorical_cols: Iterable[str],
    utility_tasks: Iterable[Mapping[str, str]],
    *,
    seed: int = 42,
    n_estimators: int = 100,
    n_jobs: int = -1,
    detector_fn: Callable = train_detector,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Measure detector and utility sensitivity when selected feature families are ignored."""
    selected = sorted({
        str(feature)
        for priority in priorities.values()
        for feature in priority.loc[priority["selected"].astype(bool), "feature"]
    })
    grouped: dict[str, set[str]] = {}
    for feature in selected:
        label = feature_groups.get(feature, feature)
        grouped.setdefault(label, set()).update(
            candidate for candidate, group in feature_groups.items() if group == label
        )
        grouped[label].add(feature)
    exclusion_sets: list[tuple[str, list[str]]] = [("none", [])]
    exclusion_sets.extend(
        (f"family:{label}", sorted(features)) for label, features in sorted(grouped.items())
    )
    exclusion_sets.append(("all_selected", selected))

    detector_rows = []
    full_auc = None
    for label, excluded in exclusion_sets:
        remaining = [column for column in real_audit if column not in set(excluded)]
        if remaining:
            result = detector_fn(
                real_audit[remaining], synthetic_audit[remaining],
                [column for column in categorical_cols if column in remaining],
                seed=seed, n_estimators=n_estimators, compute_shap=False, n_jobs=n_jobs,
            )
            auc = float(result.metrics["detector_auc"])
            status = "ok"
        else:
            auc = float("nan")
            status = "no_features_remaining"
        if label == "none":
            full_auc = auc
        detector_rows.append({
            "exclusion": label,
            "excluded_features": "|".join(excluded),
            "excluded_feature_count": len(excluded),
            "detector_auc": auc,
            "detector_auc_drop_vs_none": float(full_auc - auc) if full_auc is not None else 0.0,
            "status": status,
        })

    utility_rows = []
    for task in utility_tasks:
        task_baseline = None
        task_rows = []
        for label, excluded in exclusion_sets:
            try:
                metrics = evaluate_utility(
                    synthetic_eval, real_eval, str(task["target_col"]), categorical_cols,
                    str(task["positive_label"]), metric_prefix="", exclude_predictors=excluded,
                    seed=seed, n_estimators=n_estimators, n_jobs=n_jobs,
                )
                status = "ok"
            except ValueError as exc:
                if "at least one predictor" not in str(exc):
                    raise
                metrics = {
                    "target": task["target_col"], "positive_label": task["positive_label"],
                    "decision_rule": "unavailable", **{
                        metric: float("nan") for metric in (
                            "roc_auc", "pr_auc", "accuracy", "balanced_accuracy",
                            "precision_macro", "recall_macro", "f1_macro",
                            "positive_precision", "positive_recall", "positive_f1",
                        )
                    },
                }
                status = "no_predictors_remaining"
            row = {
                "utility_task": task["name"],
                "target_balance": task.get("balance", "unspecified"),
                "exclusion": label,
                "excluded_features": "|".join(excluded),
                "excluded_feature_count": len(excluded),
                "status": status,
                **metrics,
            }
            if label == "none":
                task_baseline = row.copy()
            task_rows.append(row)
        for row in task_rows:
            if task_baseline is not None:
                for metric in ("roc_auc", "pr_auc", "accuracy", "balanced_accuracy", "f1_macro", "positive_recall"):
                    row[f"delta_{metric}_vs_none"] = float(row[metric] - task_baseline[metric])
            utility_rows.append(row)
    return pd.DataFrame(detector_rows), pd.DataFrame(utility_rows)
