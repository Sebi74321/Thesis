"""GradientSHAP trajectories for CTAB-GAN+ discriminator snapshots.

This module moves the internal-discriminator analysis out of the MIMIC
experiment notebook.  It deliberately remains separate from ``detector.py``:
that module explains a post-hoc real-vs-synthetic random forest, whereas this
module explains the discriminator that was trained with the GAN.
"""

from __future__ import annotations

from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass
from typing import Any

import numpy as np
import pandas as pd
import torch
from scipy.stats import ks_2samp, wasserstein_distance
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    balanced_accuracy_score,
    roc_auc_score,
    roc_curve,
)

from model.synthesizer.ctabgan_synthesizer import (
    Discriminator,
    Generator,
    apply_activate,
    determine_layers_disc,
    determine_layers_gen,
)
from .detector import (
    OUTCOME_GROUPS,
    deterministic_row_subset,
    outcome_group_masks,
    prepare_detector_probe,
)


RESULT_COLUMNS = [
    "epoch",
    "outcome_group",
    "primary_scope",
    "feature",
    "mean_abs_shap",
    "mean_signed_shap",
    "importance_share",
    "rank",
    "encoded_dimensions",
    "background_rows",
    "candidate_rows",
    "explained_rows",
]

METRIC_COLUMNS = [
    "epoch",
    "calibration_rows",
    "holdout_rows",
    "threshold",
    "auc",
    "average_precision",
    "accuracy",
    "balanced_accuracy",
    "orientation_free_separability",
    "real_recall",
    "synthetic_recall",
    "predicted_real_fraction",
    "real_score_mean",
    "synthetic_score_mean",
    "score_mean_gap",
    "standardized_score_mean_gap",
    "score_ks_statistic",
    "score_wasserstein_scaled",
    "auc_bootstrap_ci_low",
    "auc_bootstrap_ci_high",
    *[f"rows_{group}" for group in OUTCOME_GROUPS],
]

LATE_WINDOW_COLUMNS = [
    "probe_mode",
    "window_snapshots",
    "first_epoch",
    "last_epoch",
    "metric",
    "mean",
    "std",
    "slope_per_epoch",
]


@dataclass(frozen=True)
class DiscriminatorSnapshotEvaluation:
    trajectory: pd.DataFrame
    metrics: pd.DataFrame
    predictions: pd.DataFrame
    epoch_matched_metrics: pd.DataFrame
    epoch_matched_predictions: pd.DataFrame
    late_window_summary: pd.DataFrame
    shap_stability: pd.DataFrame


class DiscriminatorTabularWrapper(torch.nn.Module):
    """Expose a CTAB+ image discriminator as an encoded-tabular model.

    CTAB+ trains its discriminator on the encoded row concatenated with a
    conditional vector and reshaped as an image. The wrapper averages each
    row's critic score over a fixed reproducible set of valid sampled
    conditions, while SHAP varies only encoded feature coordinates.
    """

    def __init__(
        self,
        discriminator,
        image_transformer,
        conditions: torch.Tensor | None = None,
    ):
        super().__init__()
        self.discriminator = discriminator
        self.image_transformer = image_transformer
        if conditions is not None and conditions.ndim != 2:
            raise ValueError("Marginalized conditions must be two-dimensional")
        self.register_buffer("conditions", conditions)

    def forward(self, encoded: torch.Tensor) -> torch.Tensor:
        if encoded.ndim != 2:
            raise ValueError("Encoded discriminator inputs must be two-dimensional")
        condition_count = 1
        if self.conditions is not None:
            condition_count = int(self.conditions.shape[0])
            expanded = encoded[:, None, :].expand(-1, condition_count, -1)
            conditions = self.conditions[None, :, :].expand(encoded.shape[0], -1, -1)
            encoded = torch.cat([expanded, conditions], dim=2).reshape(
                encoded.shape[0] * condition_count, -1
            )
        scores, _ = self.discriminator(self.image_transformer.transform(encoded))
        # GradientExplainer expects one scalar output per row.  CTAB+'s
        # convolutional discriminator returns singleton spatial dimensions.
        scores = scores.reshape(scores.shape[0], -1).mean(dim=1)
        return scores.reshape(-1, condition_count).mean(dim=1, keepdim=True)


def encoded_feature_slices(transformer, feature_names: Sequence[str]) -> dict[str, slice]:
    """Map each original column to its contiguous CTAB+ encoded coordinates.

    CTAB+'s native ``DataTransformer.output_info`` is a *flat* list of
    activation spans, not a list with exactly one item per source column.
    Continuous mixture-model columns and mixed columns each emit two spans
    (a scalar value and a component/modal one-hot vector), while categorical
    and general continuous columns emit one. Use the transformer's original
    column metadata to group those spans before aggregating SHAP values.

    A nested one-entry-per-column representation is retained as a fallback for
    lightweight/test transformers that do not expose CTAB+'s ``meta`` field.
    """
    output_info = list(transformer.output_info)
    metadata = getattr(transformer, "meta", None)

    def span_width(span: Any) -> int:
        # Native CTAB+: (width, activation[, tag]). Older test doubles and
        # CTGAN-like transformers may group several such tuples in a list.
        if not isinstance(span, (tuple, list)) or not span:
            raise ValueError(f"Invalid transformer output span: {span!r}")
        if isinstance(span[0], (tuple, list)):
            return sum(span_width(item) for item in span)
        width = int(span[0])
        if width <= 0:
            raise ValueError(f"Transformer output span has invalid width: {span!r}")
        return width

    if metadata is None:
        if len(output_info) != len(feature_names):
            raise ValueError(
                "Transformer without original-column metadata must provide one "
                "output_info entry per feature: "
                f"{len(output_info)} entries for {len(feature_names)} features"
            )
        grouped_spans = [[span] for span in output_info]
    else:
        metadata = list(metadata)
        if len(metadata) != len(feature_names):
            raise ValueError(
                "Transformer original-column metadata does not match the feature count: "
                f"{len(metadata)} columns for {len(feature_names)} features"
            )
        general_columns = {
            int(index) for index in getattr(transformer, "general_columns", [])
        }
        grouped_spans = []
        span_index = 0
        for column_index, column_metadata in enumerate(metadata):
            if not isinstance(column_metadata, Mapping):
                raise ValueError(
                    "Transformer original-column metadata entries must be mappings; "
                    f"column {column_index} is {column_metadata!r}"
                )
            column_type = column_metadata.get("type")
            if column_type not in {"categorical", "continuous", "mixed"}:
                raise ValueError(
                    "Unsupported transformer column type for "
                    f"{feature_names[column_index]!r}: {column_type!r}"
                )
            span_count = 2 if (
                column_type == "mixed"
                or (column_type == "continuous" and column_index not in general_columns)
            ) else 1
            next_index = span_index + span_count
            if next_index > len(output_info):
                raise ValueError(
                    "Transformer output metadata ended while mapping original column "
                    f"{feature_names[column_index]!r}"
                )
            grouped_spans.append(output_info[span_index:next_index])
            span_index = next_index
        if span_index != len(output_info):
            raise ValueError(
                "Transformer output metadata contains unmapped activation spans: "
                f"mapped {span_index} of {len(output_info)}"
            )

    result: dict[str, slice] = {}
    start = 0
    for feature, spans in zip(feature_names, grouped_spans):
        width = sum(span_width(span) for span in spans)
        result[str(feature)] = slice(start, start + width)
        start += width

    output_dim = getattr(transformer, "output_dim", start)
    if int(output_dim) != start:
        raise ValueError(
            f"Transformer metadata covers {start} coordinates but output_dim is {output_dim}"
        )
    return result


def _single_output_shap(values: Any) -> np.ndarray:
    """Normalize SHAP version-specific single-output return shapes."""
    if isinstance(values, list):
        if len(values) != 1:
            raise ValueError(f"Expected one discriminator output, received {len(values)}")
        values = values[0]
    array = np.asarray(values)
    if array.ndim == 3 and array.shape[-1] == 1:
        array = array[:, :, 0]
    if array.ndim != 2:
        raise ValueError(
            "Expected SHAP values with shape (rows, encoded_features); "
            f"received {array.shape}"
        )
    return array


def aggregate_encoded_shap(
    encoded_shap: np.ndarray,
    feature_slices: Mapping[str, slice],
) -> pd.DataFrame:
    """Group signed encoded contributions, then summarize original features."""
    values = _single_output_shap(encoded_shap)
    expected = max((feature_slice.stop for feature_slice in feature_slices.values()), default=0)
    if values.shape[1] != expected:
        raise ValueError(
            f"Received {values.shape[1]} encoded SHAP columns; expected {expected}"
        )

    rows = []
    for feature, feature_slice in feature_slices.items():
        contribution = values[:, feature_slice].sum(axis=1)
        rows.append(
            {
                "feature": feature,
                "mean_abs_shap": float(np.mean(np.abs(contribution))),
                "mean_signed_shap": float(np.mean(contribution)),
                "encoded_dimensions": int(feature_slice.stop - feature_slice.start),
            }
        )
    return pd.DataFrame(rows)


def reconstruct_discriminator(snapshot: Mapping[str, Any], synthesizer, device) -> Discriminator:
    """Recreate a discriminator without reusing the trained layer instances."""
    if "state_dict" not in snapshot or "epoch" not in snapshot:
        raise ValueError("Each discriminator snapshot requires epoch and state_dict fields")
    if synthesizer.dside is None:
        raise ValueError("The fitted CTAB+ synthesizer has no discriminator side length")
    layers = determine_layers_disc(int(synthesizer.dside), int(synthesizer.num_channels))
    discriminator = Discriminator(int(synthesizer.dside), layers).to(device)
    discriminator.load_state_dict(snapshot["state_dict"])
    discriminator.eval()
    return discriminator


def reconstruct_generator(snapshot: Mapping[str, Any], synthesizer, device) -> Generator:
    """Recreate the generator paired with a discriminator snapshot."""
    if "generator_state_dict" not in snapshot or "epoch" not in snapshot:
        raise ValueError(
            "Epoch-matched evaluation requires snapshots containing epoch and "
            "generator_state_dict"
        )
    if synthesizer.gside is None:
        raise ValueError("The fitted CTAB+ synthesizer has no generator side length")
    input_dim = int(synthesizer.random_dim) + int(synthesizer.cond_generator.n_opt)
    layers = determine_layers_gen(
        int(synthesizer.gside), input_dim, int(synthesizer.num_channels)
    )
    generator = Generator(int(synthesizer.gside), layers).to(device)
    generator.load_state_dict(snapshot["generator_state_dict"])
    generator.eval()
    return generator


def _prepare_probe_frame(adapter, frame: pd.DataFrame) -> pd.DataFrame:
    """Apply the fitted DataPrep transformations to previously unseen rows."""
    data_prep = adapter.data_prep
    columns = list(data_prep.df.columns)
    missing = [column for column in columns if column not in frame]
    if missing:
        raise ValueError(f"Discriminator probe is missing columns: {missing}")
    prepared = frame.loc[:, columns].copy(deep=True)
    prepared = prepared.replace(r" ", np.nan).fillna("empty")

    categorical = set(data_prep.categorical_columns)
    for column in columns:
        if column not in categorical:
            prepared[column] = prepared[column].map(
                lambda value: -9999999 if value == "empty" else value
            )

    for column in data_prep.log_columns:
        lower = float(data_prep.lower_bounds[column])

        def transform_log(value):
            if value == -9999999:
                return value
            value = float(value)
            if lower > 0:
                return np.log(value)
            if lower == 0:
                return np.log(value + 1.0)
            return np.log(value - lower + 1.0)

        prepared[column] = prepared[column].map(transform_log)

    encoders = {
        item["column"]: item["label_encoder"]
        for item in data_prep.label_encoder_list
    }
    for column, encoder in encoders.items():
        values = prepared[column].astype(str)
        unknown = sorted(set(values) - set(encoder.classes_))
        if unknown:
            raise ValueError(
                f"Discriminator probe contains unseen categories for {column!r}: "
                f"{unknown[:5]}"
            )
        prepared[column] = encoder.transform(values)
    return prepared


def _encode_probe(adapter, frame: pd.DataFrame, seed: int) -> np.ndarray:
    transformer = adapter.synthesizer.transformer
    prepare_for_discriminator = getattr(adapter, "prepare_discriminator_probe", None)
    if prepare_for_discriminator is not None:
        frame = prepare_for_discriminator(frame, seed=int(seed) + 32452843)
    prepared = _prepare_probe_frame(adapter, frame)
    numpy_state = np.random.get_state()
    ordering = getattr(transformer, "ordering", None)
    ordering_length = len(ordering) if ordering is not None else None
    feature_names = [str(column) for column in adapter.data_prep.df.columns]
    feature_slices = encoded_feature_slices(transformer, feature_names)
    fitted_ordering = None
    if ordering is not None and len(ordering) >= len(feature_names):
        fitted_ordering = list(ordering[: len(feature_names)])
    original_filters = getattr(transformer, "filter_arr", None)
    probe_filters = []
    if original_filters is not None:
        for column_index, metadata in enumerate(transformer.meta):
            if metadata.get("type") == "mixed":
                probe_filters.append(
                    ~np.isin(
                        prepared.iloc[:, column_index].to_numpy(),
                        np.asarray(metadata.get("modal", [])),
                    )
                )
    try:
        np.random.seed(int(seed) % (2**32))
        if original_filters is not None:
            transformer.filter_arr = probe_filters
        encoded = np.asarray(
            transformer.transform(prepared.values), dtype=np.float32
        )
        if fitted_ordering is not None:
            probe_ordering = list(
                ordering[ordering_length : ordering_length + len(feature_names)]
            )
            if len(probe_ordering) != len(feature_names):
                raise ValueError(
                    "CTAB+ preprocessing did not record one probe ordering per feature"
                )
            # CTAB+ frequency-sorts mixture/modal one-hot coordinates every
            # time ``transform`` is called.  The discriminator was trained on
            # the ordering learned from real_train, so audit rows must be
            # realigned to that fitted ordering rather than retaining a new
            # audit-specific order.
            for feature_index, feature in enumerate(feature_names):
                fitted = fitted_ordering[feature_index]
                probe = probe_ordering[feature_index]
                if fitted is None and probe is None:
                    continue
                if fitted is None or probe is None:
                    raise ValueError(
                        f"CTAB+ ordering type changed for feature {feature!r}"
                    )
                fitted = np.asarray(fitted, dtype=int)
                probe = np.asarray(probe, dtype=int)
                feature_slice = feature_slices[feature]
                onehot_start = int(feature_slice.start) + 1
                onehot_stop = int(feature_slice.stop)
                if (
                    len(fitted) != onehot_stop - onehot_start
                    or len(probe) != len(fitted)
                    or set(fitted.tolist()) != set(probe.tolist())
                ):
                    raise ValueError(
                        f"CTAB+ ordering metadata is invalid for feature {feature!r}"
                    )
                probe_position = {
                    int(original_index): position
                    for position, original_index in enumerate(probe)
                }
                positions = [probe_position[int(index)] for index in fitted]
                encoded[:, onehot_start:onehot_stop] = encoded[
                    :, onehot_start:onehot_stop
                ][:, positions]
    finally:
        np.random.set_state(numpy_state)
        if original_filters is not None:
            transformer.filter_arr = original_filters
        if ordering is not None and ordering_length is not None:
            del ordering[ordering_length:]
    if encoded.ndim != 2 or len(encoded) == 0:
        raise ValueError("CTAB+ preprocessing produced no explainable probe rows")
    if not np.isfinite(encoded).all():
        raise ValueError(
            "CTAB+ preprocessing produced non-finite discriminator probe values"
        )
    return encoded


def _sample_conditions(synthesizer, count: int, seed: int, device) -> torch.Tensor | None:
    generator = synthesizer.cond_generator
    if int(getattr(generator, "n_opt", 0)) == 0:
        return None
    if count <= 0:
        raise ValueError("condition_samples must be positive for conditional models")
    numpy_state = np.random.get_state()
    try:
        np.random.seed(int(seed) % (2**32))
        conditions = np.asarray(generator.sample(int(count)), dtype=np.float32)
    finally:
        np.random.set_state(numpy_state)
    return torch.as_tensor(conditions, device=device)


def _balanced_background_indices(
    calibration_indices: np.ndarray,
    truth: np.ndarray,
    size: int,
    seed: int,
) -> np.ndarray:
    half = max(1, int(size) // 2)
    real = calibration_indices[truth[calibration_indices] == 1]
    synthetic = calibration_indices[truth[calibration_indices] == 0]
    real_selected = deterministic_row_subset(real, min(half, len(real)), seed)
    synthetic_selected = deterministic_row_subset(
        synthetic, min(half, len(synthetic)), seed + 1
    )
    selected = np.sort(np.concatenate([real_selected, synthetic_selected]))
    if len(selected) == 0:
        raise ValueError("No calibration rows are available for the SHAP background")
    return selected


def _critic_scores(
    wrapper: torch.nn.Module,
    encoded: np.ndarray,
    device,
    batch_size: int = 1024,
) -> np.ndarray:
    parts = []
    with torch.no_grad():
        for start in range(0, len(encoded), int(batch_size)):
            batch = torch.as_tensor(
                encoded[start:start + int(batch_size)], device=device
            )
            parts.append(wrapper(batch).detach().cpu().numpy().reshape(-1))
    return np.concatenate(parts) if parts else np.empty(0, dtype=float)


def _calibrated_threshold(truth: np.ndarray, scores: np.ndarray) -> float:
    """Select the calibration-only threshold maximizing Youden's J statistic."""
    fpr, tpr, thresholds = roc_curve(truth, scores)
    finite = np.isfinite(thresholds)
    if not finite.any():
        raise ValueError("Could not determine a finite discriminator threshold")
    objective = np.where(finite, tpr - fpr, -np.inf)
    return float(thresholds[int(np.argmax(objective))])


def _fixed_generator_inputs(synthesizer, count: int, seed: int) -> tuple[torch.Tensor, torch.Tensor]:
    """Create one CPU latent/condition probe reused at every saved epoch."""
    if count <= 0:
        raise ValueError("Epoch-matched generator probe size must be positive")
    torch_generator = torch.Generator(device="cpu")
    torch_generator.manual_seed(int(seed))
    noise = torch.randn(
        count, int(synthesizer.random_dim), generator=torch_generator, dtype=torch.float32
    )
    n_opt = int(synthesizer.cond_generator.n_opt)
    if n_opt:
        numpy_state = np.random.get_state()
        try:
            np.random.seed(int(seed) % (2**32))
            condition_array = np.asarray(
                synthesizer.cond_generator.sample(count), dtype=np.float32
            )
        finally:
            np.random.set_state(numpy_state)
        conditions = torch.from_numpy(condition_array)
    else:
        conditions = torch.empty((count, 0), dtype=torch.float32)
    return noise, conditions


def _generate_snapshot_encoded(
    snapshot: Mapping[str, Any],
    synthesizer,
    noise: torch.Tensor,
    conditions: torch.Tensor,
    device,
    batch_size: int = 1024,
    seed: int = 42,
) -> np.ndarray:
    """Generate encoded rows from the generator saved at the same epoch."""
    generator = reconstruct_generator(snapshot, synthesizer, device)
    outputs = []
    cuda_devices = []
    if torch.device(device).type == "cuda":
        cuda_devices = [torch.device(device).index or 0]
    with torch.random.fork_rng(devices=cuda_devices):
        # apply_activate uses Gumbel-softmax, so fixing only the latent vector
        # would not provide a genuinely shared stochastic probe.
        torch.manual_seed(int(seed))
        with torch.no_grad():
            for start in range(0, len(noise), int(batch_size)):
                latent = noise[start:start + int(batch_size)].to(device)
                condition_batch = conditions[start:start + int(batch_size)].to(device)
                if condition_batch.shape[1]:
                    latent = torch.cat([latent, condition_batch], dim=1)
                latent = latent.reshape(latent.shape[0], latent.shape[1], 1, 1)
                generated = generator(latent)
                generated = synthesizer.Gtransformer.inverse_transform(generated)
                outputs.append(
                    apply_activate(generated, synthesizer.transformer.output_info)
                    .detach()
                    .cpu()
                    .numpy()
                )
    del generator
    encoded = np.concatenate(outputs, axis=0).astype(np.float32, copy=False)
    if encoded.ndim != 2 or len(encoded) != len(noise) or not np.isfinite(encoded).all():
        raise ValueError("A generator snapshot produced an invalid encoded probe")
    return encoded


def _bootstrap_auc_interval(
    truth: np.ndarray,
    scores: np.ndarray,
    *,
    repeats: int,
    seed: int,
) -> tuple[float, float]:
    """Return a deterministic stratified percentile interval for ROC-AUC."""
    if repeats <= 0:
        return np.nan, np.nan
    truth = np.asarray(truth, dtype=int)
    scores = np.asarray(scores, dtype=float)
    negative = np.flatnonzero(truth == 0)
    positive = np.flatnonzero(truth == 1)
    if not len(negative) or not len(positive):
        return np.nan, np.nan
    rng = np.random.default_rng(int(seed))
    estimates = np.empty(int(repeats), dtype=float)
    for index in range(int(repeats)):
        sampled = np.concatenate(
            [
                rng.choice(negative, len(negative), replace=True),
                rng.choice(positive, len(positive), replace=True),
            ]
        )
        estimates[index] = roc_auc_score(truth[sampled], scores[sampled])
    low, high = np.quantile(estimates, [0.025, 0.975])
    return float(low), float(high)


def _snapshot_metric_row(
    *,
    epoch: int,
    truth: np.ndarray,
    scores: np.ndarray,
    predictions: np.ndarray,
    threshold: float,
    calibration_rows: int,
    bootstrap_repeats: int,
    seed: int,
) -> tuple[dict[str, Any], dict[str, np.ndarray]]:
    groups = outcome_group_masks(truth, predictions)
    auc = float(roc_auc_score(truth, scores))
    real_scores = scores[truth == 1]
    synthetic_scores = scores[truth == 0]
    real_std = float(np.std(real_scores, ddof=0))
    synthetic_std = float(np.std(synthetic_scores, ddof=0))
    pooled_scale = float(np.sqrt((real_std**2 + synthetic_std**2) / 2.0))
    if not np.isfinite(pooled_scale) or pooled_scale <= 0:
        pooled_scale = 1.0
    mean_gap = float(np.mean(real_scores) - np.mean(synthetic_scores))
    ci_low, ci_high = _bootstrap_auc_interval(
        truth,
        scores,
        repeats=int(bootstrap_repeats),
        seed=int(seed) + int(epoch),
    )
    real_total = max(1, int((truth == 1).sum()))
    synthetic_total = max(1, int((truth == 0).sum()))
    row = {
        "epoch": int(epoch),
        "calibration_rows": int(calibration_rows),
        "holdout_rows": int(len(truth)),
        "threshold": float(threshold),
        "auc": auc,
        "average_precision": float(average_precision_score(truth, scores)),
        "accuracy": float(accuracy_score(truth, predictions)),
        "balanced_accuracy": float(balanced_accuracy_score(truth, predictions)),
        "orientation_free_separability": float(2.0 * abs(auc - 0.5)),
        "real_recall": float(groups["correct_real"].sum() / real_total),
        "synthetic_recall": float(groups["correct_synthetic"].sum() / synthetic_total),
        "predicted_real_fraction": float(np.mean(predictions == 1)),
        "real_score_mean": float(np.mean(real_scores)),
        "synthetic_score_mean": float(np.mean(synthetic_scores)),
        "score_mean_gap": mean_gap,
        "standardized_score_mean_gap": float(mean_gap / pooled_scale),
        "score_ks_statistic": float(ks_2samp(real_scores, synthetic_scores).statistic),
        "score_wasserstein_scaled": float(
            wasserstein_distance(real_scores, synthetic_scores) / pooled_scale
        ),
        "auc_bootstrap_ci_low": ci_low,
        "auc_bootstrap_ci_high": ci_high,
        **{f"rows_{group}": int(mask.sum()) for group, mask in groups.items()},
    }
    return row, groups


def summarize_late_window(
    metrics: pd.DataFrame,
    *,
    probe_mode: str,
    window_snapshots: int,
) -> pd.DataFrame:
    """Summarize level, variability, and slope across the last snapshots."""
    if metrics.empty:
        return pd.DataFrame(columns=LATE_WINDOW_COLUMNS)
    count = min(max(2, int(window_snapshots)), len(metrics))
    window = metrics.sort_values("epoch", kind="mergesort").tail(count)
    epochs = pd.to_numeric(window["epoch"], errors="coerce").to_numpy(dtype=float)
    tracked = [
        "auc",
        "orientation_free_separability",
        "average_precision",
        "balanced_accuracy",
        "real_recall",
        "synthetic_recall",
        "predicted_real_fraction",
        "standardized_score_mean_gap",
        "score_ks_statistic",
        "score_wasserstein_scaled",
    ]
    rows = []
    for metric in tracked:
        values = pd.to_numeric(window[metric], errors="coerce").to_numpy(dtype=float)
        finite = np.isfinite(values) & np.isfinite(epochs)
        if not finite.any():
            continue
        slope = 0.0
        if finite.sum() >= 2 and np.ptp(epochs[finite]) > 0:
            slope = float(np.polyfit(epochs[finite], values[finite], 1)[0])
        rows.append(
            {
                "probe_mode": str(probe_mode),
                "window_snapshots": int(count),
                "first_epoch": int(window["epoch"].min()),
                "last_epoch": int(window["epoch"].max()),
                "metric": metric,
                "mean": float(np.mean(values[finite])),
                "std": float(np.std(values[finite], ddof=0)),
                "slope_per_epoch": slope,
            }
        )
    return pd.DataFrame(rows, columns=LATE_WINDOW_COLUMNS)


def summarize_shap_stability(
    trajectory: pd.DataFrame,
    *,
    window_snapshots: int,
    top_k: int = 5,
) -> pd.DataFrame:
    """Compare feature rankings between adjacent late-window snapshots."""
    primary = trajectory[trajectory["primary_scope"].astype(bool)].copy()
    if primary.empty:
        return pd.DataFrame(
            columns=["epoch_from", "epoch_to", "spearman", "top_k_jaccard"]
        )
    epochs = sorted(pd.to_numeric(primary["epoch"], errors="coerce").dropna().astype(int).unique())
    epochs = epochs[-min(max(2, int(window_snapshots)), len(epochs)):]
    rows = []
    for epoch_from, epoch_to in zip(epochs[:-1], epochs[1:]):
        left = primary[primary["epoch"] == epoch_from].set_index("feature")
        right = primary[primary["epoch"] == epoch_to].set_index("feature")
        features = sorted(set(left.index) | set(right.index))
        left_values = left["importance_share"].reindex(features, fill_value=0.0)
        right_values = right["importance_share"].reindex(features, fill_value=0.0)
        correlation = left_values.corr(right_values, method="spearman")
        count = min(int(top_k), len(features))
        left_top = set(left_values.nlargest(count).index)
        right_top = set(right_values.nlargest(count).index)
        union = left_top | right_top
        rows.append(
            {
                "epoch_from": int(epoch_from),
                "epoch_to": int(epoch_to),
                "spearman": float(correlation) if np.isfinite(correlation) else np.nan,
                "top_k": int(count),
                "top_k_overlap_count": int(len(left_top & right_top)),
                "top_k_jaccard": float(len(left_top & right_top) / len(union)) if union else 0.0,
            }
        )
    return pd.DataFrame(rows)


def evaluate_discriminator_snapshots(
    adapter,
    real_probe: pd.DataFrame,
    synthetic_probe: pd.DataFrame,
    *,
    background_size: int = 50,
    explain_size: int = 100,
    test_size: float = 0.3,
    condition_samples: int = 8,
    bootstrap_repeats: int = 200,
    late_window_snapshots: int = 3,
    seed: int = 42,
    exclude_features: Iterable[str] = (),
) -> DiscriminatorSnapshotEvaluation:
    """Evaluate and explain snapshot decisions on a shared labeled audit probe.

    Every snapshot uses the same balanced real/synthetic probe split, balanced
    SHAP background, marginalized valid conditional vectors, and deterministic
    outcome-group sampling.  The primary scope matches detector weighting:
    correctly classified synthetic holdout rows.
    """
    if background_size < 2 or explain_size <= 0:
        raise ValueError(
            "background_size must be at least two and explain_size must be positive"
        )
    synthesizer = getattr(adapter, "synthesizer", None)
    data_prep = getattr(adapter, "data_prep", None)
    snapshots = list(getattr(adapter, "discriminator_snapshots", []))
    if synthesizer is None or data_prep is None:
        raise ValueError("A fitted CTABGANPlusAdapter is required")
    if not snapshots:
        raise ValueError(
            "No discriminator snapshots are available; set generator.snapshot_frq "
            "to a positive epoch interval or configure generator.snapshot_schedule"
        )

    feature_names = [str(column) for column in data_prep.df.columns]
    slices = encoded_feature_slices(synthesizer.transformer, feature_names)
    common, probe, truth, calibration_indices, holdout_indices = (
        prepare_detector_probe(
            real_probe,
            synthetic_probe,
            seed=seed,
            test_size=test_size,
        )
    )
    if common != feature_names:
        raise ValueError(
            "Discriminator probe schema must exactly match the fitted CTAB+ schema"
        )
    encoded = _encode_probe(adapter, probe, seed)
    background_indices = _balanced_background_indices(
        calibration_indices,
        truth,
        min(int(background_size), len(calibration_indices)),
        seed,
    )
    device = torch.device(getattr(adapter, "device", "cpu"))
    background = torch.as_tensor(encoded[background_indices], device=device)
    conditions = _sample_conditions(
        synthesizer, int(condition_samples), seed, device
    )
    excluded = {str(feature) for feature in exclude_features}
    if excluded.issuperset(feature_names):
        raise ValueError("Discriminator SHAP cannot exclude every original feature")

    try:
        import shap
    except ImportError as exc:  # pragma: no cover - dependency error is environment-specific
        raise RuntimeError("SHAP is required for discriminator snapshot evaluation") from exc

    ordered_snapshots = sorted(snapshots, key=lambda item: int(item["epoch"]))
    epoch_matched_available = all(
        "generator_state_dict" in snapshot for snapshot in ordered_snapshots
    )
    paired_noise = paired_conditions = None
    paired_real_encoded = None
    if epoch_matched_available:
        rows_per_source = len(truth) // 2
        paired_real_encoded = encoded[:rows_per_source]
        paired_noise, paired_conditions = _fixed_generator_inputs(
            synthesizer, rows_per_source, int(seed) + 7919
        )

    results = []
    metric_rows = []
    prediction_rows = []
    epoch_matched_metric_rows = []
    epoch_matched_prediction_rows = []
    for snapshot in ordered_snapshots:
        epoch = int(snapshot["epoch"])
        discriminator = reconstruct_discriminator(snapshot, synthesizer, device)
        wrapper = DiscriminatorTabularWrapper(
            discriminator,
            synthesizer.Dtransformer,
            conditions=conditions,
        ).to(device)
        wrapper.eval()
        scores = _critic_scores(wrapper, encoded, device)
        threshold = _calibrated_threshold(
            truth[calibration_indices], scores[calibration_indices]
        )
        holdout_truth = truth[holdout_indices]
        holdout_scores = scores[holdout_indices]
        holdout_predictions = (holdout_scores >= threshold).astype(int)
        metric_row, groups = _snapshot_metric_row(
            epoch=epoch,
            truth=holdout_truth,
            scores=holdout_scores,
            predictions=holdout_predictions,
            threshold=threshold,
            calibration_rows=len(calibration_indices),
            bootstrap_repeats=bootstrap_repeats,
            seed=seed,
        )
        metric_rows.append(metric_row)
        group_for_row = np.full(len(holdout_indices), "", dtype=object)
        for group, mask in groups.items():
            group_for_row[mask] = group
        prediction_rows.extend(
            {
                "epoch": epoch,
                "probe_row": int(probe_index),
                "source": "real" if row_truth == 1 else "synthetic",
                "truth_is_real": int(row_truth),
                "predicted_is_real": int(row_prediction),
                "critic_score": float(row_score),
                "threshold": threshold,
                "outcome_group": str(group),
            }
            for probe_index, row_truth, row_prediction, row_score, group in zip(
                holdout_indices,
                holdout_truth,
                holdout_predictions,
                holdout_scores,
                group_for_row,
            )
        )

        if epoch_matched_available:
            historical_synthetic = _generate_snapshot_encoded(
                snapshot,
                synthesizer,
                paired_noise,
                paired_conditions,
                device,
                seed=int(seed) + 15485863,
            )
            paired_encoded = np.concatenate(
                [paired_real_encoded, historical_synthetic], axis=0
            )
            paired_scores = _critic_scores(wrapper, paired_encoded, device)
            paired_threshold = _calibrated_threshold(
                truth[calibration_indices], paired_scores[calibration_indices]
            )
            paired_holdout_scores = paired_scores[holdout_indices]
            paired_predictions = (
                paired_holdout_scores >= paired_threshold
            ).astype(int)
            paired_row, paired_groups = _snapshot_metric_row(
                epoch=epoch,
                truth=holdout_truth,
                scores=paired_holdout_scores,
                predictions=paired_predictions,
                threshold=paired_threshold,
                calibration_rows=len(calibration_indices),
                bootstrap_repeats=bootstrap_repeats,
                seed=seed + 104729,
            )
            epoch_matched_metric_rows.append(paired_row)
            paired_group_for_row = np.full(len(holdout_indices), "", dtype=object)
            for group, mask in paired_groups.items():
                paired_group_for_row[mask] = group
            epoch_matched_prediction_rows.extend(
                {
                    "epoch": epoch,
                    "probe_row": int(probe_index),
                    "source": "real" if row_truth == 1 else "synthetic",
                    "truth_is_real": int(row_truth),
                    "predicted_is_real": int(row_prediction),
                    "critic_score": float(row_score),
                    "threshold": paired_threshold,
                    "outcome_group": str(group),
                }
                for probe_index, row_truth, row_prediction, row_score, group in zip(
                    holdout_indices,
                    holdout_truth,
                    paired_predictions,
                    paired_holdout_scores,
                    paired_group_for_row,
                )
            )
        explainer = shap.GradientExplainer(wrapper, background)
        for group_position, group in enumerate(OUTCOME_GROUPS):
            candidate_positions = np.flatnonzero(groups[group])
            selected_positions = deterministic_row_subset(
                candidate_positions,
                int(explain_size),
                seed + group_position * 1009,
            )
            if len(selected_positions) == 0:
                continue
            explain_indices = holdout_indices[selected_positions]
            to_explain = torch.as_tensor(encoded[explain_indices], device=device)
            # GradientExplainer uses Monte Carlo interpolation. Reset and
            # restore RNGs so groups and epochs are exactly reproducible.
            shap_numpy_state = np.random.get_state()
            cuda_devices = []
            if device.type == "cuda":
                cuda_devices = [device.index if device.index is not None else 0]
            group_seed = int(seed + group_position * 1009)
            try:
                np.random.seed(group_seed % (2**32))
                with torch.random.fork_rng(devices=cuda_devices):
                    torch.manual_seed(group_seed)
                    shap_values = _single_output_shap(
                        explainer.shap_values(to_explain)
                    )
            finally:
                np.random.set_state(shap_numpy_state)
            importance = aggregate_encoded_shap(shap_values, slices)
            importance = importance[
                ~importance["feature"].isin(excluded)
            ].copy()
            importance = importance.sort_values(
                ["mean_abs_shap", "feature"],
                ascending=[False, True],
                kind="mergesort",
            ).reset_index(drop=True)
            total = float(importance["mean_abs_shap"].sum())
            importance["importance_share"] = (
                importance["mean_abs_shap"] / total if total > 0 else 0.0
            )
            importance["rank"] = np.arange(1, len(importance) + 1, dtype=int)
            importance.insert(0, "primary_scope", group == "correct_synthetic")
            importance.insert(0, "outcome_group", group)
            importance.insert(0, "epoch", epoch)
            importance["background_rows"] = len(background_indices)
            importance["candidate_rows"] = len(candidate_positions)
            importance["explained_rows"] = len(selected_positions)
            results.append(importance)

    trajectory = (
        pd.concat(results, ignore_index=True).loc[:, RESULT_COLUMNS]
        if results
        else pd.DataFrame(columns=RESULT_COLUMNS)
    )
    fixed_metrics = pd.DataFrame(metric_rows, columns=METRIC_COLUMNS)
    epoch_matched_metrics = pd.DataFrame(
        epoch_matched_metric_rows, columns=METRIC_COLUMNS
    )
    late_summaries = [
        summarize_late_window(
            fixed_metrics,
            probe_mode="fixed_final_generator",
            window_snapshots=late_window_snapshots,
        )
    ]
    if not epoch_matched_metrics.empty:
        late_summaries.append(
            summarize_late_window(
                epoch_matched_metrics,
                probe_mode="epoch_matched_generator",
                window_snapshots=late_window_snapshots,
            )
        )
    return DiscriminatorSnapshotEvaluation(
        trajectory=trajectory,
        metrics=fixed_metrics,
        predictions=pd.DataFrame(prediction_rows),
        epoch_matched_metrics=epoch_matched_metrics,
        epoch_matched_predictions=pd.DataFrame(epoch_matched_prediction_rows),
        late_window_summary=pd.concat(late_summaries, ignore_index=True),
        shap_stability=summarize_shap_stability(
            trajectory,
            window_snapshots=late_window_snapshots,
        ),
    )
