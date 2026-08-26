"""GradientSHAP trajectories for CTAB-GAN+ discriminator snapshots.

This module moves the internal-discriminator analysis out of the MIMIC
experiment notebook.  It deliberately remains separate from ``detector.py``:
that module explains a post-hoc real-vs-synthetic random forest, whereas this
module explains the discriminator that was trained with the GAN.
"""

from __future__ import annotations

from collections.abc import Iterable, Mapping, Sequence
from typing import Any

import numpy as np
import pandas as pd
import torch

from model.synthesizer.ctabgan_synthesizer import (
    Discriminator,
    determine_layers_disc,
)


RESULT_COLUMNS = [
    "epoch",
    "feature",
    "mean_abs_shap",
    "importance_share",
    "rank",
    "encoded_dimensions",
    "background_rows",
    "explained_rows",
]


class DiscriminatorTabularWrapper(torch.nn.Module):
    """Expose a CTAB+ image discriminator as an encoded-tabular model.

    CTAB+ trains its discriminator on the encoded row concatenated with a
    conditional vector and reshaped as an image.  The all-zero conditional
    vector matches the notebook analysis and keeps conditional coordinates
    fixed while SHAP varies only encoded feature coordinates.
    """

    def __init__(self, discriminator, image_transformer, cond_dim: int = 0):
        super().__init__()
        self.discriminator = discriminator
        self.image_transformer = image_transformer
        self.cond_dim = int(cond_dim)

    def forward(self, encoded: torch.Tensor) -> torch.Tensor:
        if encoded.ndim != 2:
            raise ValueError("Encoded discriminator inputs must be two-dimensional")
        if self.cond_dim:
            condition = torch.zeros(
                (encoded.shape[0], self.cond_dim),
                dtype=encoded.dtype,
                device=encoded.device,
            )
            encoded = torch.cat([encoded, condition], dim=1)
        scores, _ = self.discriminator(self.image_transformer.transform(encoded))
        # GradientExplainer expects one scalar output per row.  CTAB+'s
        # convolutional discriminator returns singleton spatial dimensions.
        return scores.reshape(scores.shape[0], -1).mean(dim=1, keepdim=True)


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
    """Aggregate mean absolute encoded attribution mass to original features."""
    values = _single_output_shap(encoded_shap)
    expected = max((feature_slice.stop for feature_slice in feature_slices.values()), default=0)
    if values.shape[1] != expected:
        raise ValueError(
            f"Received {values.shape[1]} encoded SHAP columns; expected {expected}"
        )

    rows = []
    for feature, feature_slice in feature_slices.items():
        contribution = np.abs(values[:, feature_slice]).mean(axis=0).sum()
        rows.append(
            {
                "feature": feature,
                "mean_abs_shap": float(contribution),
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


def evaluate_discriminator_snapshots(
    adapter,
    *,
    background_size: int = 50,
    explain_size: int = 100,
    seed: int = 42,
    exclude_features: Iterable[str] = (),
) -> pd.DataFrame:
    """Compute a tidy original-feature SHAP trajectory over CTAB+ snapshots.

    The same background and explained rows are reused for every epoch, making
    temporal changes attributable to the discriminator rather than row
    resampling.  Raw magnitudes and within-epoch shares are both returned;
    shares are useful because the discriminator output scale can drift.
    """
    if background_size <= 0 or explain_size <= 0:
        raise ValueError("background_size and explain_size must be positive")
    synthesizer = getattr(adapter, "synthesizer", None)
    data_prep = getattr(adapter, "data_prep", None)
    snapshots = list(getattr(adapter, "discriminator_snapshots", []))
    if synthesizer is None or data_prep is None:
        raise ValueError("A fitted CTABGANPlusAdapter is required")
    if not snapshots:
        raise ValueError(
            "No discriminator snapshots are available; set generator.snapshot_frq "
            "to a positive epoch interval"
        )

    feature_names = [str(column) for column in data_prep.df.columns]
    slices = encoded_feature_slices(synthesizer.transformer, feature_names)
    # DataTransformer samples mixture-component encodings with NumPy's global
    # RNG and appends ordering metadata on each transform.  Make this analysis
    # reproducible without changing later pipeline randomness or transformer
    # state.
    numpy_state = np.random.get_state()
    ordering = getattr(synthesizer.transformer, "ordering", None)
    ordering_length = len(ordering) if ordering is not None else None
    try:
        np.random.seed(int(seed) % (2**32))
        encoded = np.asarray(
            synthesizer.transformer.transform(data_prep.df.values), dtype=np.float32
        )
    finally:
        np.random.set_state(numpy_state)
        if ordering is not None and ordering_length is not None:
            del ordering[ordering_length:]
    if encoded.ndim != 2 or len(encoded) == 0:
        raise ValueError("CTAB+ preprocessing produced no explainable rows")

    rng = np.random.default_rng(int(seed))
    background_indices = rng.choice(
        len(encoded), size=min(int(background_size), len(encoded)), replace=False
    )
    explain_indices = rng.choice(
        len(encoded), size=min(int(explain_size), len(encoded)), replace=False
    )
    device = torch.device(getattr(adapter, "device", "cpu"))
    background = torch.as_tensor(encoded[background_indices], device=device)
    to_explain = torch.as_tensor(encoded[explain_indices], device=device)
    excluded = {str(feature) for feature in exclude_features}
    if excluded.issuperset(feature_names):
        raise ValueError("Discriminator SHAP cannot exclude every original feature")

    try:
        import shap
    except ImportError as exc:  # pragma: no cover - dependency error is environment-specific
        raise RuntimeError("SHAP is required for discriminator snapshot evaluation") from exc

    results = []
    for snapshot in sorted(snapshots, key=lambda item: int(item["epoch"])):
        discriminator = reconstruct_discriminator(snapshot, synthesizer, device)
        wrapper = DiscriminatorTabularWrapper(
            discriminator,
            synthesizer.Dtransformer,
            cond_dim=int(getattr(synthesizer.cond_generator, "n_opt", 0)),
        ).to(device)
        wrapper.eval()
        explainer = shap.GradientExplainer(wrapper, background)
        # GradientExplainer uses Monte Carlo interpolation.  Reset and restore
        # its RNGs so every epoch uses the same draws and this diagnostic does
        # not perturb the rest of the experiment.
        shap_numpy_state = np.random.get_state()
        cuda_devices = []
        if device.type == "cuda":
            cuda_devices = [device.index if device.index is not None else 0]
        try:
            np.random.seed(int(seed) % (2**32))
            with torch.random.fork_rng(devices=cuda_devices):
                torch.manual_seed(int(seed))
                shap_values = _single_output_shap(explainer.shap_values(to_explain))
        finally:
            np.random.set_state(shap_numpy_state)
        importance = aggregate_encoded_shap(shap_values, slices)
        importance = importance[~importance["feature"].isin(excluded)].copy()
        importance = importance.sort_values(
            ["mean_abs_shap", "feature"], ascending=[False, True], kind="mergesort"
        ).reset_index(drop=True)
        total = float(importance["mean_abs_shap"].sum())
        importance["importance_share"] = (
            importance["mean_abs_shap"] / total if total > 0 else 0.0
        )
        importance["rank"] = np.arange(1, len(importance) + 1, dtype=int)
        importance.insert(0, "epoch", int(snapshot["epoch"]))
        importance["background_rows"] = len(background_indices)
        importance["explained_rows"] = len(explain_indices)
        results.append(importance)

    if not results:
        return pd.DataFrame(columns=RESULT_COLUMNS)
    return pd.concat(results, ignore_index=True).loc[:, RESULT_COLUMNS]
