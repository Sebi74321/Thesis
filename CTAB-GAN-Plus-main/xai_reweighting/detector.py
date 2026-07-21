"""Real-vs-synthetic detector and original-feature TreeSHAP aggregation."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Iterable, Optional

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.metrics import accuracy_score, average_precision_score, roc_auc_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder

from .scoring import normalize_signal


def _one_hot_encoder():
    try:
        return OneHotEncoder(handle_unknown="ignore", sparse_output=False)
    except TypeError:  # scikit-learn < 1.2
        return OneHotEncoder(handle_unknown="ignore", sparse=False)


@dataclass(frozen=True)
class DetectorResult:
    metrics: Dict[str, float]
    shap_importance: pd.Series


def train_detector(
    real_df: pd.DataFrame,
    synthetic_df: pd.DataFrame,
    categorical_cols: Iterable[str],
    seed: int = 42,
    n_estimators: int = 300,
    test_size: float = 0.3,
    shap_max_rows: int = 2000,
    compute_shap: bool = True,
    n_jobs: int = -1,
) -> DetectorResult:
    common = [c for c in real_df.columns if c in synthetic_df.columns]
    if not common:
        raise ValueError("Real and synthetic data have no common columns")
    n = min(len(real_df), len(synthetic_df))
    if n < 4:
        raise ValueError("At least four real and synthetic rows are required")
    real = real_df.loc[:, common].sample(n=n, random_state=seed).copy()
    synthetic = synthetic_df.loc[:, common].sample(n=n, random_state=seed).copy()
    categorical = [c for c in categorical_cols if c in common]
    for column in categorical:
        real[column] = real[column].astype("object").where(real[column].notna(), "__missing__").astype(str)
        synthetic[column] = synthetic[column].astype("object").where(
            synthetic[column].notna(), "__missing__"
        ).astype(str)
    real["__is_real__"] = 1
    synthetic["__is_real__"] = 0
    combined = pd.concat([real, synthetic], ignore_index=True)
    X, y = combined.drop(columns="__is_real__"), combined["__is_real__"]

    continuous = [c for c in common if c not in categorical]
    numeric_pipe = Pipeline([("imputer", SimpleImputer(strategy="median"))])
    categorical_pipe = Pipeline(
        [("imputer", SimpleImputer(strategy="most_frequent")), ("onehot", _one_hot_encoder())]
    )
    preprocessor = ColumnTransformer(
        [("num", numeric_pipe, continuous), ("cat", categorical_pipe, categorical)]
    )
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=seed, stratify=y
    )
    X_train_p = preprocessor.fit_transform(X_train)
    X_test_p = preprocessor.transform(X_test)
    detector = RandomForestClassifier(
        n_estimators=n_estimators,
        random_state=seed,
        n_jobs=n_jobs,
        class_weight="balanced",
    )
    detector.fit(X_train_p, y_train)
    probabilities = detector.predict_proba(X_test_p)[:, 1]
    predictions = (probabilities >= 0.5).astype(int)
    metrics = {
        "detector_auc": float(roc_auc_score(y_test, probabilities)),
        "detector_average_precision": float(average_precision_score(y_test, probabilities)),
        "detector_accuracy": float(accuracy_score(y_test, predictions)),
        "n_real": int(n),
        "n_synthetic": int(n),
    }

    importance = pd.Series(0.0, index=common, dtype=float)
    if compute_shap:
        try:
            import shap
        except ImportError as exc:
            raise RuntimeError("SHAP is required for A4/A5; install the base requirements") from exc
        if len(X_test_p) > shap_max_rows:
            rng = np.random.default_rng(seed)
            selected = rng.choice(len(X_test_p), shap_max_rows, replace=False)
            X_shap = X_test_p[selected]
        else:
            X_shap = X_test_p
        values = shap.TreeExplainer(detector).shap_values(X_shap)
        if isinstance(values, list):
            values = values[1]
        elif isinstance(values, np.ndarray) and values.ndim == 3:
            values = values[:, :, 1]
        mean_abs = np.abs(np.asarray(values)).mean(axis=0)

        mapping = list(continuous)
        if categorical:
            encoder = preprocessor.named_transformers_["cat"].named_steps["onehot"]
            for feature, categories in zip(categorical, encoder.categories_):
                mapping.extend([feature] * len(categories))
        if len(mapping) != len(mean_abs):
            raise RuntimeError("Could not map encoded SHAP values to original features")
        for feature, value in zip(mapping, mean_abs):
            importance.loc[feature] += float(value)
        importance = normalize_signal(importance)
    return DetectorResult(metrics, importance)
