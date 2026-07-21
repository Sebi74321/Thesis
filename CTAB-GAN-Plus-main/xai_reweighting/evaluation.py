"""Common fidelity, tail, utility, privacy, and audit evaluation."""

from __future__ import annotations

from typing import Any, Dict, Iterable, Tuple

import numpy as np
import pandas as pd
from scipy.spatial.distance import jensenshannon
from scipy.stats import ks_2samp, wasserstein_distance
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.neighbors import NearestNeighbors
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import LabelEncoder, OneHotEncoder, RobustScaler

from .detector import train_detector
from .scoring import MISSING


def _encoder():
    try:
        return OneHotEncoder(handle_unknown="ignore", sparse_output=False)
    except TypeError:
        return OneHotEncoder(handle_unknown="ignore", sparse=False)


def _cat(series: pd.Series) -> pd.Series:
    return series.astype("object").where(series.notna(), MISSING).astype(str)


def _js(real: pd.Series, syn: pd.Series) -> float:
    real, syn = _cat(real), _cat(syn)
    categories = sorted(set(real.unique()).union(syn.unique()))
    rp, sp = real.value_counts(normalize=True), syn.value_counts(normalize=True)
    return float(
        jensenshannon(
            [float(rp.get(x, 0.0)) for x in categories],
            [float(sp.get(x, 0.0)) for x in categories],
            base=2.0,
        )
    )


def _cdf_tail_divergence(real: np.ndarray, synthetic: np.ndarray, tau: float = 0.90) -> float:
    threshold = float(np.quantile(real, tau))
    maximum = float(np.max(real))
    if maximum <= threshold:
        return 0.0
    grid = np.linspace(threshold, maximum, 200)
    real_sorted = np.sort(real)
    synthetic_sorted = np.sort(synthetic)
    real_cdf = np.searchsorted(real_sorted, grid, side="right") / len(real_sorted)
    synthetic_cdf = np.searchsorted(synthetic_sorted, grid, side="right") / len(synthetic_sorted)
    return float(np.max(np.abs(real_cdf - synthetic_cdf)))


def evaluate_fidelity_and_tails(
    real: pd.DataFrame,
    synthetic: pd.DataFrame,
    categorical_cols: Iterable[str],
    continuous_cols: Iterable[str],
    rare_threshold: float = 0.05,
) -> Tuple[Dict[str, float], pd.DataFrame]:
    records = []
    for col in continuous_cols:
        r = pd.to_numeric(real[col], errors="coerce").dropna().to_numpy()
        s = pd.to_numeric(synthetic[col], errors="coerce").dropna().to_numpy()
        if len(r) == 0 or len(s) == 0:
            continue
        q25, q75 = np.quantile(r, [0.25, 0.75])
        scale = float(q75 - q25)
        if not np.isfinite(scale) or scale <= 0:
            scale = float(np.std(r))
        if not np.isfinite(scale) or scale <= 0:
            scale = 1.0
        record: Dict[str, Any] = {
            "feature": col,
            "kind": "continuous",
            "wasserstein": float(wasserstein_distance(r, s)),
            "wasserstein_scaled": float(wasserstein_distance(r, s) / scale),
            "ks": float(ks_2samp(r, s).statistic),
            "cdf_tail_divergence": _cdf_tail_divergence(r, s),
        }
        for quantile in (0.05, 0.95, 0.99):
            threshold = float(np.quantile(r, quantile))
            side = "lower" if quantile == 0.05 else "upper"
            expected = quantile if side == "lower" else 1.0 - quantile
            real_mass = float(np.mean(r <= threshold) if side == "lower" else np.mean(r >= threshold))
            syn_mass = float(np.mean(s <= threshold) if side == "lower" else np.mean(s >= threshold))
            label = f"q{int(quantile * 100):02d}"
            record[f"{label}_{side}_mass_real"] = real_mass
            record[f"{label}_{side}_mass_syn"] = syn_mass
            record[f"{label}_{side}_mass_error"] = abs(syn_mass - real_mass)
            record[f"{label}_{side}_coverage"] = min(syn_mass / expected, 1.0) if expected > 0 else np.nan
            record[f"{label}_quantile_error_scaled"] = abs(float(np.quantile(s, quantile)) - threshold) / scale
        records.append(record)

    for col in categorical_cols:
        r, s = _cat(real[col]), _cat(synthetic[col])
        rp, sp = r.value_counts(normalize=True), s.value_counts(normalize=True)
        rare = [value for value, probability in rp.items() if probability <= rare_threshold]
        rare_error = float(sum(abs(float(rp.get(x, 0.0) - sp.get(x, 0.0))) for x in rare))
        records.append(
            {
                "feature": col,
                "kind": "categorical",
                "jensen_shannon": _js(r, s),
                "rare_category_frequency_error": rare_error,
            }
        )

    details = pd.DataFrame(records)
    metrics: Dict[str, float] = {}
    for column in details.select_dtypes(include=[np.number]).columns:
        metrics[f"mean_{column}"] = float(details[column].mean(skipna=True))

    numeric = [c for c in continuous_cols if c in real and c in synthetic]
    if len(numeric) >= 2:
        real_corr = real[numeric].corr().fillna(0.0).to_numpy()
        syn_corr = synthetic[numeric].corr().fillna(0.0).to_numpy()
        metrics["correlation_distance"] = float(
            np.linalg.norm(real_corr - syn_corr, ord="fro") / len(numeric)
        )
    else:
        metrics["correlation_distance"] = float("nan")
    return metrics, details


def _predictor_preprocessor(columns, categorical_cols):
    categorical = [c for c in categorical_cols if c in columns]
    continuous = [c for c in columns if c not in categorical]
    return ColumnTransformer(
        [
            (
                "num",
                Pipeline(
                    [("imputer", SimpleImputer(strategy="median")), ("scale", RobustScaler())]
                ),
                continuous,
            ),
            (
                "cat",
                Pipeline(
                    [("imputer", SimpleImputer(strategy="most_frequent")), ("onehot", _encoder())]
                ),
                categorical,
            ),
        ]
    )


def evaluate_utility(
    synthetic: pd.DataFrame,
    real_eval: pd.DataFrame,
    target_col: str,
    categorical_cols: Iterable[str],
    seed: int = 42,
    n_estimators: int = 300,
    n_jobs: int = -1,
) -> Dict[str, float]:
    predictors = [c for c in real_eval.columns if c != target_col]
    X_syn, y_syn = synthetic[predictors].copy(), _cat(synthetic[target_col])
    X_real, y_real = real_eval[predictors].copy(), _cat(real_eval[target_col])
    for column in categorical_cols:
        if column != target_col and column in predictors:
            X_syn[column] = _cat(X_syn[column])
            X_real[column] = _cat(X_real[column])
    rare_label = y_real.value_counts().idxmin()
    label_encoder = LabelEncoder().fit(pd.concat([y_syn, y_real], ignore_index=True))
    y_syn_encoded = label_encoder.transform(y_syn)
    y_real_encoded = label_encoder.transform(y_real)
    rare_class = int(label_encoder.transform([rare_label])[0])
    preprocessor = _predictor_preprocessor(predictors, categorical_cols)
    X_syn_p = preprocessor.fit_transform(X_syn)
    X_real_p = preprocessor.transform(X_real)
    model = RandomForestClassifier(
        n_estimators=n_estimators,
        random_state=seed,
        n_jobs=n_jobs,
        class_weight="balanced",
    )
    model.fit(X_syn_p, y_syn_encoded)
    prediction = model.predict(X_real_p)
    if rare_class not in model.classes_:
        probability = np.zeros(len(y_real_encoded))
    else:
        probability = model.predict_proba(X_real_p)[:, list(model.classes_).index(rare_class)]
    truth_binary = (y_real_encoded == rare_class).astype(int)
    return {
        "utility_roc_auc": float(roc_auc_score(truth_binary, probability)),
        "utility_pr_auc": float(average_precision_score(truth_binary, probability)),
        "utility_f1": float(
            f1_score(y_real_encoded, prediction, pos_label=rare_class, zero_division=0)
        ),
        "utility_accuracy": float(accuracy_score(y_real_encoded, prediction)),
        "utility_precision": float(
            precision_score(y_real_encoded, prediction, pos_label=rare_class, zero_division=0)
        ),
        "utility_recall": float(
            recall_score(y_real_encoded, prediction, pos_label=rare_class, zero_division=0)
        ),
        "rare_event_recall": float(
            recall_score(y_real_encoded, prediction, pos_label=rare_class, zero_division=0)
        ),
        "rare_class": str(rare_label),
    }


def _nearest_distances(reference, queries, chunk_size=1024, n_jobs=-1):
    model = NearestNeighbors(n_neighbors=1, n_jobs=n_jobs).fit(reference)
    chunks = []
    for start in range(0, len(queries), chunk_size):
        chunks.append(model.kneighbors(queries[start : start + chunk_size], return_distance=True)[0][:, 0])
    return np.concatenate(chunks) if chunks else np.array([], dtype=float)


def evaluate_privacy(
    real_train: pd.DataFrame,
    real_eval: pd.DataFrame,
    synthetic: pd.DataFrame,
    categorical_cols: Iterable[str],
    n_jobs: int = -1,
) -> Dict[str, float]:
    columns = real_train.columns.tolist()
    real_train = real_train[columns].copy()
    real_eval = real_eval[columns].copy()
    synthetic = synthetic[columns].copy()
    for column in categorical_cols:
        if column in columns:
            real_train[column] = _cat(real_train[column])
            real_eval[column] = _cat(real_eval[column])
            synthetic[column] = _cat(synthetic[column])
    preprocessor = _predictor_preprocessor(columns, categorical_cols)
    train_p = np.asarray(preprocessor.fit_transform(real_train[columns]), dtype=float)
    eval_p = np.asarray(preprocessor.transform(real_eval[columns]), dtype=float)
    syn_p = np.asarray(preprocessor.transform(synthetic[columns]), dtype=float)
    scale = np.sqrt(max(1, train_p.shape[1]))
    syn_dist = _nearest_distances(train_p, syn_p, n_jobs=n_jobs) / scale
    eval_dist = _nearest_distances(train_p, eval_p, n_jobs=n_jobs) / scale
    train_hashes = set(pd.util.hash_pandas_object(real_train[columns], index=False).astype(str))
    syn_hashes = pd.util.hash_pandas_object(synthetic[columns], index=False).astype(str)
    metrics = {"privacy_exact_match_rate": float(syn_hashes.isin(train_hashes).mean())}
    for name, distances in (("synthetic", syn_dist), ("heldout", eval_dist)):
        for percentile in (5, 50, 95):
            metrics[f"privacy_{name}_nn_p{percentile}"] = float(np.percentile(distances, percentile))
    heldout_median = metrics["privacy_heldout_nn_p50"]
    metrics["privacy_median_distance_ratio"] = (
        metrics["privacy_synthetic_nn_p50"] / heldout_median if heldout_median > 0 else float("nan")
    )
    return metrics


def evaluate_variant(
    real_train: pd.DataFrame,
    real_eval: pd.DataFrame,
    synthetic: pd.DataFrame,
    target_col: str,
    categorical_cols: Iterable[str],
    continuous_cols: Iterable[str],
    seed: int = 42,
    n_jobs: int = -1,
    n_estimators: int = 300,
) -> Tuple[Dict[str, Any], pd.DataFrame]:
    synthetic = synthetic.loc[:, real_train.columns]
    metrics, feature_details = evaluate_fidelity_and_tails(
        real_eval, synthetic, categorical_cols, continuous_cols
    )
    metrics.update(
        evaluate_utility(
            synthetic, real_eval, target_col, categorical_cols, seed,
            n_estimators=n_estimators, n_jobs=n_jobs,
        )
    )
    metrics.update(evaluate_privacy(real_train, real_eval, synthetic, categorical_cols, n_jobs))
    audit = train_detector(
        real_eval,
        synthetic,
        categorical_cols,
        seed=seed,
        n_estimators=n_estimators,
        compute_shap=False,
        n_jobs=n_jobs,
    )
    metrics.update(audit.metrics)
    target_real = _cat(real_eval[target_col]).value_counts(normalize=True)
    target_syn = _cat(synthetic[target_col]).value_counts(normalize=True)
    rare_target = target_real.idxmin()
    metrics["rare_outcome_frequency_error"] = abs(
        float(target_real.get(rare_target, 0.0) - target_syn.get(rare_target, 0.0))
    )
    return metrics, feature_details
