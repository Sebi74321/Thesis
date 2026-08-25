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
    """Map each original column to its contiguous CTAB+ encoded coordinates."""
    output_info = list(transformer.output_info)
    if len(output_info) != len(feature_names):
        raise ValueError(
            "Transformer output metadata does not match the original feature count: "
            f"{len(output_info)} metadata entries for {len(feature_names)} features"
        )

    result: dict[str, slice] = {}
    start = 0
    for feature, info in zip(feature_names, output_info):
        width = sum(int(item[0]) for item in info)
        if width <= 0:
            raise ValueError(f"Feature {feature!r} has no encoded coordinates")
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
