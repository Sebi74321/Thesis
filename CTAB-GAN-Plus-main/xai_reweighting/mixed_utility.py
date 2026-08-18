"""Real/synthetic mixture utility curves with a fixed, data-agnostic decision rule."""

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
    f1_score,
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


def evaluate_training_utility(
    training: pd.DataFrame,
    real_eval: pd.DataFrame,
    target_col: str,
    categorical_cols: Iterable[str],
    *,
    positive_label: str,
    seed: int = 42,
    n_estimators: int = 300,
    n_jobs: int = -1,
) -> dict[str, float | int | str]:
    """Fit once and evaluate using the classifier's fixed argmax predictions."""
    predictors = [column for column in real_eval.columns if column != target_col]
    frames = [training, real_eval]
    for frame in frames:
        missing = [column for column in [*predictors, target_col] if column not in frame]
        if missing:
            raise ValueError(f"Utility frame is missing columns: {missing}")

    X_train = training[predictors].copy()
    X_eval = real_eval[predictors].copy()
    for column in categorical_cols:
        if column != target_col and column in predictors:
            X_train[column] = _cat(X_train[column])
            X_eval[column] = _cat(X_eval[column])

    y_train = _cat(training[target_col])
    y_eval = _cat(real_eval[target_col])
    if y_eval.nunique() != 2:
        raise ValueError("Mixed utility currently requires a binary target")
    positive_label = str(positive_label)
    if positive_label not in set(y_eval):
        raise ValueError(f"Configured positive label {positive_label!r} is absent from evaluation data")
    encoder = LabelEncoder().fit(pd.concat([y_train, y_eval], ignore_index=True))
    y_train_encoded = encoder.transform(y_train)
    y_eval_encoded = encoder.transform(y_eval)
    positive_class = int(encoder.transform([positive_label])[0])

    preprocessor = _predictor_preprocessor(predictors, categorical_cols)
    X_train_p = preprocessor.fit_transform(X_train)
    X_eval_p = preprocessor.transform(X_eval)
    model = RandomForestClassifier(
        n_estimators=n_estimators,
        random_state=seed,
        n_jobs=n_jobs,
        class_weight="balanced",
    )
    model.fit(X_train_p, y_train_encoded)
    prediction = model.predict(X_eval_p)

    def positive_probability(values) -> np.ndarray:
        if positive_class not in model.classes_:
            return np.zeros(values.shape[0], dtype=float)
        index = list(model.classes_).index(positive_class)
        return model.predict_proba(values)[:, index]

    eval_probability = positive_probability(X_eval_p)
    eval_truth = (y_eval_encoded == positive_class).astype(int)
    return {
        "target_col": target_col,
        "positive_label": positive_label,
        "decision_rule": "random_forest_argmax",
        "roc_auc": float(roc_auc_score(eval_truth, eval_probability)),
        "pr_auc": float(average_precision_score(eval_truth, eval_probability)),
        "accuracy": float(accuracy_score(y_eval_encoded, prediction)),
        "balanced_accuracy": float(balanced_accuracy_score(y_eval_encoded, prediction)),
        "precision_macro": float(precision_score(y_eval_encoded, prediction, average="macro", zero_division=0)),
        "recall_macro": float(recall_score(y_eval_encoded, prediction, average="macro", zero_division=0)),
        "f1_macro": float(f1_score(y_eval_encoded, prediction, average="macro", zero_division=0)),
        "positive_precision": float(precision_score(y_eval_encoded, prediction, pos_label=positive_class, zero_division=0)),
        "positive_recall": float(recall_score(y_eval_encoded, prediction, pos_label=positive_class, zero_division=0)),
        "positive_f1": float(f1_score(y_eval_encoded, prediction, pos_label=positive_class, zero_division=0)),
    }


def evaluate_real_only_baseline(
    real_train: pd.DataFrame,
    real_eval: pd.DataFrame,
    target_col: str,
    categorical_cols: Iterable[str],
    *,
    positive_label: str,
    repeats: int = 3,
    seed: int = 42,
    n_estimators: int = 300,
    n_jobs: int = -1,
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
                real_eval,
                target_col,
                categorical_cols,
                positive_label=positive_label,
                seed=repeat_seed,
                n_estimators=n_estimators,
                n_jobs=n_jobs,
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
    real_eval: pd.DataFrame,
    target_col: str,
    categorical_cols: Iterable[str],
    real_only: pd.DataFrame,
    *,
    positive_label: str,
    additive_fractions: Sequence[float] = (0.0, 0.25, 0.5, 1.0),
    replacement_fractions: Sequence[float] = (0.0, 0.25, 0.5, 0.75, 1.0),
    repeats: int = 3,
    seed: int = 42,
    n_estimators: int = 300,
    n_jobs: int = -1,
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
                        real_eval,
                        target_col,
                        categorical_cols,
                        positive_label=positive_label,
                        seed=repeat_seed,
                        n_estimators=n_estimators,
                        n_jobs=n_jobs,
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
    keys = [
        column
        for column in ["variant", "utility_task", "target_balance", "protocol", "synthetic_fraction"]
        if column in results.columns
    ]
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
