"""Real-vs-synthetic detector and original-feature TreeSHAP aggregation."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, Iterable

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


PRIMARY_SHAP_SCOPE = "correct_synthetic_holdout_only"
OUTCOME_GROUPS = (
    "correct_real",
    "false_real_as_synthetic",
    "correct_synthetic",
    "false_synthetic_as_real",
)


def _one_hot_encoder():
    try:
        return OneHotEncoder(handle_unknown="ignore", sparse_output=False)
    except TypeError:  # scikit-learn < 1.2
        return OneHotEncoder(handle_unknown="ignore", sparse=False)


@dataclass(frozen=True)
class DetectorResult:
    metrics: Dict[str, Any]
    shap_importance: pd.Series
    shap_signed: pd.Series


def prepare_detector_probe(
    real_df: pd.DataFrame,
    synthetic_df: pd.DataFrame,
    *,
    seed: int,
    test_size: float,
) -> tuple[list[str], pd.DataFrame, np.ndarray, np.ndarray, np.ndarray]:
    """Create the deterministic balanced probe shared by both discriminators."""
    common = [column for column in real_df.columns if column in synthetic_df.columns]
    if not common:
        raise ValueError("Real and synthetic data have no common columns")
    n = min(len(real_df), len(synthetic_df))
    if n < 4:
        raise ValueError("At least four real and synthetic rows are required")
    real = real_df.loc[:, common].sample(n=n, random_state=seed).copy()
    synthetic = synthetic_df.loc[:, common].sample(n=n, random_state=seed).copy()
    combined = pd.concat([real, synthetic], ignore_index=True)
    truth = np.concatenate(
        [np.ones(n, dtype=int), np.zeros(n, dtype=int)]
    )
    indices = np.arange(len(combined), dtype=int)
    calibration_indices, holdout_indices = train_test_split(
        indices,
        test_size=test_size,
        random_state=seed,
        stratify=truth,
    )
    return common, combined, truth, calibration_indices, holdout_indices


def outcome_group_masks(
    truth: np.ndarray, predictions: np.ndarray
) -> dict[str, np.ndarray]:
    """Return the four real/synthetic classification outcome masks."""
    truth = np.asarray(truth, dtype=int)
    predictions = np.asarray(predictions, dtype=int)
    if truth.shape != predictions.shape:
        raise ValueError("Truth and prediction arrays must have identical shapes")
    return {
        "correct_real": (truth == 1) & (predictions == 1),
        "false_real_as_synthetic": (truth == 1) & (predictions == 0),
        "correct_synthetic": (truth == 0) & (predictions == 0),
        "false_synthetic_as_real": (truth == 0) & (predictions == 1),
    }


def deterministic_row_subset(
    indices: np.ndarray, maximum: int, seed: int
) -> np.ndarray:
    """Select a reproducible sorted subset without changing global RNG state."""
    indices = np.asarray(indices, dtype=int)
    if len(indices) <= maximum:
        return np.sort(indices)
    rng = np.random.default_rng(int(seed))
    return np.sort(rng.choice(indices, int(maximum), replace=False))


def _aggregate_encoded_shap(
    values: np.ndarray,
    continuous: Iterable[str],
    categorical: Iterable[str],
    category_sizes: Iterable[int],
) -> pd.Series:
    """Aggregate signed encoded SHAP contributions before measuring magnitude."""
    statistics = _aggregate_encoded_shap_statistics(
        values, continuous, categorical, category_sizes
    )
    return statistics.set_index("feature")["mean_abs_shap"]


def _aggregate_encoded_shap_statistics(
    values: np.ndarray,
    continuous: Iterable[str],
    categorical: Iterable[str],
    category_sizes: Iterable[int],
) -> pd.DataFrame:
    """Return comparable signed and absolute original-feature SHAP values."""
    values = np.asarray(values, dtype=float)
    if values.ndim != 2:
        raise ValueError("Encoded SHAP values must be a two-dimensional array")
    continuous = list(continuous)
    categorical = list(categorical)
    category_sizes = [int(size) for size in category_sizes]
    if len(categorical) != len(category_sizes):
        raise ValueError("Each categorical feature must have one encoded category count")

    expected_columns = len(continuous) + sum(category_sizes)
    if values.shape[1] != expected_columns:
        raise RuntimeError(
            "Could not map encoded SHAP values to original features: "
            f"received {values.shape[1]} columns, expected {expected_columns}"
        )

    rows = []
    offset = 0
    for feature in continuous:
        contribution = values[:, offset]
        rows.append(
            {
                "feature": feature,
                "mean_abs_shap": float(np.mean(np.abs(contribution))),
                "mean_signed_shap": float(np.mean(contribution)),
            }
        )
        offset += 1
    for feature, size in zip(categorical, category_sizes):
        # A categorical feature is represented by several one-hot columns.
        # Preserve their signs while grouping each row, then measure the
        # magnitude of the original feature's combined contribution.
        grouped_contribution = values[:, offset:offset + size].sum(axis=1)
        rows.append(
            {
                "feature": feature,
                "mean_abs_shap": float(np.mean(np.abs(grouped_contribution))),
                "mean_signed_shap": float(np.mean(grouped_contribution)),
            }
        )
        offset += size
    return pd.DataFrame(rows)


def train_detector(
    real_df: pd.DataFrame,
    synthetic_df: pd.DataFrame,
    categorical_cols: Iterable[str],
    seed: int = 42,
    n_estimators: int = 300,
    test_size: float = 0.3,
    shap_max_rows: int = 100,
    shap_scope: str = PRIMARY_SHAP_SCOPE,
    compute_shap: bool = True,
    n_jobs: int = -1,
) -> DetectorResult:
    if shap_scope != PRIMARY_SHAP_SCOPE:
        raise ValueError(
            f"Detector SHAP scope must be {PRIMARY_SHAP_SCOPE!r} so weighting "
            "targets artifacts in synthetic rows the detector identifies correctly"
        )
    if compute_shap and shap_max_rows < 1:
        raise ValueError("shap_max_rows must be at least one when SHAP is enabled")
    common, combined, truth_all, train_indices, test_indices = prepare_detector_probe(
        real_df, synthetic_df, seed=seed, test_size=test_size
    )
    n = len(combined) // 2
    X = combined.copy(deep=True)
    categorical = [c for c in categorical_cols if c in common]
    for column in categorical:
        X[column] = (
            X[column]
            .astype("object")
            .where(X[column].notna(), "__missing__")
            .astype(str)
        )

    continuous = [c for c in common if c not in categorical]
    numeric_pipe = Pipeline([("imputer", SimpleImputer(strategy="median"))])
    categorical_pipe = Pipeline(
        [("imputer", SimpleImputer(strategy="most_frequent")), ("onehot", _one_hot_encoder())]
    )
    preprocessor = ColumnTransformer(
        [("num", numeric_pipe, continuous), ("cat", categorical_pipe, categorical)]
    )
    X_train = X.iloc[train_indices]
    X_test = X.iloc[test_indices]
    y_train = truth_all[train_indices]
    y_test = truth_all[test_indices]
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
    truth = np.asarray(y_test, dtype=int)
    misclassified = predictions != truth
    groups = outcome_group_masks(truth, predictions)
    candidate_indices = np.flatnonzero(groups["correct_synthetic"])
    metrics = {
        "detector_auc": float(roc_auc_score(y_test, probabilities)),
        "detector_average_precision": float(average_precision_score(y_test, probabilities)),
        "detector_accuracy": float(accuracy_score(y_test, predictions)),
        "n_real": int(n),
        "n_synthetic": int(n),
        "detector_holdout_rows": int(len(y_test)),
        "detector_misclassified_rows": int(misclassified.sum()),
        "detector_misclassification_rate": float(misclassified.mean()),
        "detector_correct_real": int(groups["correct_real"].sum()),
        "detector_false_real_as_synthetic": int(
            groups["false_real_as_synthetic"].sum()
        ),
        "detector_correct_synthetic": int(groups["correct_synthetic"].sum()),
        "detector_false_synthetic_as_real": int(
            groups["false_synthetic_as_real"].sum()
        ),
        "detector_shap_scope": shap_scope,
        "detector_shap_candidate_rows": int(len(candidate_indices)),
        "detector_shap_rows": 0,
        "detector_shap_status": "disabled" if not compute_shap else "pending",
    }

    importance = pd.Series(0.0, index=common, dtype=float)
    signed_importance = pd.Series(0.0, index=common, dtype=float)
    if compute_shap:
        try:
            import shap
        except ImportError as exc:
            raise RuntimeError("SHAP is required for A3/A4/A5; install the base requirements") from exc
        if len(candidate_indices) == 0:
            metrics["detector_shap_status"] = "no_correct_synthetic_holdout_rows"
            return DetectorResult(metrics, importance, signed_importance)
        selected = deterministic_row_subset(candidate_indices, shap_max_rows, seed)
        X_shap = X_test_p[selected]
        metrics["detector_shap_rows"] = int(len(selected))
        metrics["detector_shap_status"] = (
            "correct_synthetic_holdout_rows_explained"
        )
        values = shap.TreeExplainer(detector).shap_values(X_shap)
        if isinstance(values, list):
            values = values[1]
        elif isinstance(values, np.ndarray) and values.ndim == 3:
            values = values[:, :, 1]
        category_sizes = []
        if categorical:
            encoder = preprocessor.named_transformers_["cat"].named_steps["onehot"]
            category_sizes = [len(categories) for categories in encoder.categories_]
        grouped_statistics = _aggregate_encoded_shap_statistics(
            np.asarray(values), continuous, categorical, category_sizes
        ).set_index("feature")
        importance.loc[grouped_statistics.index] = grouped_statistics[
            "mean_abs_shap"
        ]
        signed_importance.loc[grouped_statistics.index] = grouped_statistics[
            "mean_signed_shap"
        ]
        importance = normalize_signal(importance)
    return DetectorResult(metrics, importance, signed_importance)
