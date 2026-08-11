"""Real/synthetic mixture utility curves with leakage-safe threshold tuning."""

from __future__ import annotations

import sys
from typing import Iterable, Sequence

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    balanced_accuracy_score,
    confusion_matrix,
    f1_score,
    fbeta_score,
    precision_recall_curve,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.preprocessing import LabelEncoder
from tqdm.auto import tqdm

from .evaluation import _cat, _predictor_preprocessor


class _ProgressReporter:
    def __init__(self, total: int, label: str, mode: str):
        if mode not in {"auto", "on", "off"}:
            raise ValueError("progress must be auto, on, or off")
        self.total = total
        self.label = label
        self.mode = mode
        self.current = 0
        interactive = mode == "on" or (mode == "auto" and sys.stderr.isatty())
        self.bar = tqdm(total=total, desc=label, unit="fit", disable=not interactive)

    def advance(self) -> None:
        self.current += 1
        if not self.bar.disable:
            self.bar.update(1)
        elif self.mode == "auto":
            print(f"[{self.label}] {self.current}/{self.total} classifier fits", flush=True)

    def close(self) -> None:
        self.bar.close()


def _stratified_sample(
    frame: pd.DataFrame, n: int, target_col: str, seed: int
) -> pd.DataFrame:
    """Sample exactly ``n`` rows without replacement, approximately stratified."""
    if not 0 <= n <= len(frame):
        raise ValueError(f"Requested {n} rows from a frame containing {len(frame)} rows")
    if n == 0:
        return frame.iloc[0:0].copy()
    if n == len(frame):
        return frame.sample(frac=1.0, random_state=seed).reset_index(drop=True)

    labels = _cat(frame[target_col])
    counts = labels.value_counts(sort=False)
    ideal = counts.astype(float) * (n / len(frame))
    allocation = np.floor(ideal).astype(int)
    remaining = n - int(allocation.sum())
    order = sorted(
        counts.index,
        key=lambda value: (-(ideal[value] - allocation[value]), str(value)),
    )
    for value in order:
        if remaining == 0:
            break
        if allocation[value] < counts[value]:
            allocation[value] += 1
            remaining -= 1
    if remaining:
        raise RuntimeError("Could not allocate the requested stratified sample")

    pieces = []
    for offset, value in enumerate(sorted(counts.index, key=str)):
        size = int(allocation[value])
        if size:
            pieces.append(
                frame.loc[labels == value].sample(n=size, random_state=seed + offset)
            )
    return (
        pd.concat(pieces, axis=0)
        .sample(frac=1.0, random_state=seed + 997)
        .reset_index(drop=True)
    )


def _select_fbeta_threshold(
    truth: np.ndarray, probability: np.ndarray, beta: float
) -> tuple[float, float]:
    if beta <= 0:
        raise ValueError("threshold beta must be positive")
    precision, recall, thresholds = precision_recall_curve(truth, probability)
    if len(thresholds) == 0:
        return 0.5, 0.0
    precision, recall = precision[:-1], recall[:-1]
    beta_squared = beta**2
    denominator = beta_squared * precision + recall
    scores = np.divide(
        (1.0 + beta_squared) * precision * recall,
        denominator,
        out=np.zeros_like(denominator, dtype=float),
        where=denominator > 0,
    )
    best = np.flatnonzero(np.isclose(scores, np.max(scores)))
    best_recall = np.max(recall[best])
    best = best[np.isclose(recall[best], best_recall)]
    best_precision = np.max(precision[best])
    best = best[np.isclose(precision[best], best_precision)]
    index = int(best[np.argmax(thresholds[best])])
    return float(thresholds[index]), float(scores[index])


def _binary_metrics(
    truth: np.ndarray, probability: np.ndarray, threshold: float, prefix: str, beta: float
) -> dict[str, float | int]:
    prediction = (probability >= threshold).astype(int)
    tn, fp, fn, tp = confusion_matrix(truth, prediction, labels=[0, 1]).ravel()
    return {
        f"{prefix}_threshold": float(threshold),
        f"{prefix}_accuracy": float(accuracy_score(truth, prediction)),
        f"{prefix}_balanced_accuracy": float(balanced_accuracy_score(truth, prediction)),
        f"{prefix}_precision": float(precision_score(truth, prediction, zero_division=0)),
        f"{prefix}_recall": float(recall_score(truth, prediction, zero_division=0)),
        f"{prefix}_f1": float(f1_score(truth, prediction, zero_division=0)),
        f"{prefix}_f2": float(fbeta_score(truth, prediction, beta=beta, zero_division=0)),
        f"{prefix}_specificity": float(tn / (tn + fp)) if tn + fp else float("nan"),
        f"{prefix}_tn": int(tn),
        f"{prefix}_fp": int(fp),
        f"{prefix}_fn": int(fn),
        f"{prefix}_tp": int(tp),
    }


def evaluate_training_utility(
    training: pd.DataFrame,
    real_threshold: pd.DataFrame,
    real_eval: pd.DataFrame,
    target_col: str,
    categorical_cols: Iterable[str],
    *,
    seed: int = 42,
    n_estimators: int = 300,
    n_jobs: int = -1,
    threshold_beta: float = 2.0,
) -> dict[str, float | int | str]:
    """Fit on one training mixture, tune on development real data, evaluate on held-out real."""
    predictors = [column for column in real_eval.columns if column != target_col]
    frames = [training, real_threshold, real_eval]
    for frame in frames:
        missing = [column for column in [*predictors, target_col] if column not in frame]
        if missing:
            raise ValueError(f"Utility frame is missing columns: {missing}")

    X_train = training[predictors].copy()
    X_threshold = real_threshold[predictors].copy()
    X_eval = real_eval[predictors].copy()
    for column in categorical_cols:
        if column != target_col and column in predictors:
            X_train[column] = _cat(X_train[column])
            X_threshold[column] = _cat(X_threshold[column])
            X_eval[column] = _cat(X_eval[column])

    y_train = _cat(training[target_col])
    y_threshold = _cat(real_threshold[target_col])
    y_eval = _cat(real_eval[target_col])
    rare_label = y_threshold.value_counts().idxmin()
    encoder = LabelEncoder().fit(pd.concat([y_train, y_threshold, y_eval], ignore_index=True))
    if len(encoder.classes_) != 2:
        raise ValueError("Mixed utility currently requires a binary target")
    rare_class = int(encoder.transform([rare_label])[0])
    y_train_encoded = encoder.transform(y_train)
    threshold_truth = (encoder.transform(y_threshold) == rare_class).astype(int)
    eval_truth = (encoder.transform(y_eval) == rare_class).astype(int)

    preprocessor = _predictor_preprocessor(predictors, categorical_cols)
    X_train_p = preprocessor.fit_transform(X_train)
    X_threshold_p = preprocessor.transform(X_threshold)
    X_eval_p = preprocessor.transform(X_eval)
    model = RandomForestClassifier(
        n_estimators=n_estimators,
        random_state=seed,
        n_jobs=n_jobs,
        class_weight="balanced",
    )
    model.fit(X_train_p, y_train_encoded)

    def rare_probability(values) -> np.ndarray:
        if rare_class not in model.classes_:
            return np.zeros(values.shape[0], dtype=float)
        index = list(model.classes_).index(rare_class)
        return model.predict_proba(values)[:, index]

    threshold_probability = rare_probability(X_threshold_p)
    eval_probability = rare_probability(X_eval_p)
    selected_threshold, development_fbeta = _select_fbeta_threshold(
        threshold_truth, threshold_probability, threshold_beta
    )
    result: dict[str, float | int | str] = {
        "rare_class": str(rare_label),
        "threshold_beta": float(threshold_beta),
        "development_fbeta": development_fbeta,
        "roc_auc": float(roc_auc_score(eval_truth, eval_probability)),
        "pr_auc": float(average_precision_score(eval_truth, eval_probability)),
    }
    result.update(_binary_metrics(eval_truth, eval_probability, 0.5, "default", threshold_beta))
    result.update(
        _binary_metrics(
            eval_truth, eval_probability, selected_threshold, "tuned", threshold_beta
        )
    )
    return result


def evaluate_real_only_baseline(
    real_train: pd.DataFrame,
    real_threshold: pd.DataFrame,
    real_eval: pd.DataFrame,
    target_col: str,
    categorical_cols: Iterable[str],
    *,
    repeats: int = 3,
    seed: int = 42,
    n_estimators: int = 300,
    n_jobs: int = -1,
    threshold_beta: float = 2.0,
    progress: str = "off",
) -> pd.DataFrame:
    if repeats < 1:
        raise ValueError("mixed utility repeats must be at least one")
    rows = []
    reporter = _ProgressReporter(repeats, "real-only utility", progress)
    try:
        for repeat in range(repeats):
            repeat_seed = seed + repeat * 1000
            metrics = evaluate_training_utility(
                real_train,
                real_threshold,
                real_eval,
                target_col,
                categorical_cols,
                seed=repeat_seed,
                n_estimators=n_estimators,
                n_jobs=n_jobs,
                threshold_beta=threshold_beta,
            )
            rows.append(
                {
                    "protocol": "real_only",
                    "synthetic_fraction": 0.0,
                    "synthetic_share_of_training": 0.0,
                    "repeat": repeat,
                    "seed": repeat_seed,
                    "real_training_rows": len(real_train),
                    "synthetic_training_rows": 0,
                    "training_rows": len(real_train),
                    **metrics,
                }
            )
            reporter.advance()
    finally:
        reporter.close()
    return pd.DataFrame(rows)


def evaluate_mixed_utility_curve(
    variant: str,
    real_train: pd.DataFrame,
    synthetic: pd.DataFrame,
    real_threshold: pd.DataFrame,
    real_eval: pd.DataFrame,
    target_col: str,
    categorical_cols: Iterable[str],
    real_only: pd.DataFrame,
    *,
    additive_fractions: Sequence[float] = (0.0, 0.25, 0.5, 1.0),
    replacement_fractions: Sequence[float] = (0.0, 0.25, 0.5, 0.75, 1.0),
    repeats: int = 3,
    seed: int = 42,
    n_estimators: int = 300,
    n_jobs: int = -1,
    threshold_beta: float = 2.0,
    progress: str = "off",
) -> pd.DataFrame:
    if len(synthetic) < len(real_train):
        raise ValueError("Mixed utility requires at least len(real_train) synthetic rows")
    if repeats != len(real_only):
        raise ValueError("Real-only baseline must contain one row per repeat")
    for fraction in [*additive_fractions, *replacement_fractions]:
        if not 0.0 <= float(fraction) <= 1.0:
            raise ValueError("Mixed utility fractions must lie within [0, 1]")

    records = []
    fit_count = repeats * (
        sum(not np.isclose(float(x), 0.0) for x in set(additive_fractions))
        + sum(not np.isclose(float(x), 0.0) for x in set(replacement_fractions))
    )
    reporter = _ProgressReporter(fit_count, f"{variant} mixed utility", progress)
    try:
        for protocol, fractions in (
            ("additive", sorted(set(float(x) for x in additive_fractions))),
            ("replacement", sorted(set(float(x) for x in replacement_fractions))),
        ):
            for point, fraction in enumerate(fractions):
                if np.isclose(fraction, 0.0):
                    for baseline in real_only.to_dict(orient="records"):
                        baseline.update(
                            {
                                "variant": variant,
                                "protocol": protocol,
                                "synthetic_fraction": 0.0,
                                "synthetic_share_of_training": 0.0,
                            }
                        )
                        records.append(baseline)
                    continue
                for repeat in range(repeats):
                    repeat_seed = seed + repeat * 1000
                    sample_seed = repeat_seed + point * 31 + (
                        0 if protocol == "additive" else 503
                    )
                    if protocol == "additive":
                        real_part = real_train.copy(deep=True)
                        synthetic_count = int(np.floor(fraction * len(real_train)))
                    else:
                        synthetic_count = int(round(fraction * len(real_train)))
                        real_count = len(real_train) - synthetic_count
                        real_part = _stratified_sample(
                            real_train, real_count, target_col, sample_seed
                        )
                    synthetic_part = _stratified_sample(
                        synthetic, synthetic_count, target_col, sample_seed + 11
                    )
                    training = (
                        pd.concat([real_part, synthetic_part], ignore_index=True)
                        .sample(frac=1.0, random_state=sample_seed + 23)
                        .reset_index(drop=True)
                    )
                    metrics = evaluate_training_utility(
                        training,
                        real_threshold,
                        real_eval,
                        target_col,
                        categorical_cols,
                        seed=repeat_seed,
                        n_estimators=n_estimators,
                        n_jobs=n_jobs,
                        threshold_beta=threshold_beta,
                    )
                    records.append(
                        {
                            "variant": variant,
                            "protocol": protocol,
                            "synthetic_fraction": fraction,
                            "synthetic_share_of_training": float(
                                len(synthetic_part) / len(training)
                            ),
                            "repeat": repeat,
                            "seed": repeat_seed,
                            "real_training_rows": len(real_part),
                            "synthetic_training_rows": len(synthetic_part),
                            "training_rows": len(training),
                            **metrics,
                        }
                    )
                    reporter.advance()
    finally:
        reporter.close()
    return pd.DataFrame(records)


def summarize_mixed_utility(results: pd.DataFrame) -> pd.DataFrame:
    keys = ["variant", "protocol", "synthetic_fraction"]
    counts = ["real_training_rows", "synthetic_training_rows", "training_rows"]
    excluded = {*keys, *counts, "repeat", "seed"}
    metrics = [
        column
        for column in results.select_dtypes(include=[np.number]).columns
        if column not in excluded
    ]
    aggregation = {column: "first" for column in counts}
    aggregation.update({column: ["mean", "std"] for column in metrics})
    summary = results.groupby(keys, as_index=False).agg(aggregation)
    summary.columns = [
        "_".join(str(part) for part in column if part)
        if isinstance(column, tuple)
        else str(column)
        for column in summary.columns
    ]
    return summary.fillna({column: 0.0 for column in summary if column.endswith("_std")})
