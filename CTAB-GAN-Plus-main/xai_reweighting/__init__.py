"""Model-agnostic XAI-guided weighted retraining utilities."""

from .augmentation import create_uniform_augmentation, create_weighted_augmentation
from .data_split import DataSplits, create_data_splits
from .scoring import (
    build_region_definitions,
    compute_feature_components,
    compute_feature_priority,
    compute_row_weights,
)

__all__ = [
    "CTABGANPlusAdapter",
    "DataSplits",
    "GeneratorAdapter",
    "build_region_definitions",
    "compute_feature_components",
    "compute_feature_priority",
    "compute_row_weights",
    "create_data_splits",
    "create_uniform_augmentation",
    "create_weighted_augmentation",
]


def __getattr__(name):
    """Load torch-backed adapters only when explicitly requested."""
    if name in {"GeneratorAdapter", "CTABGANPlusAdapter"}:
        from .generator_adapters import CTABGANPlusAdapter, GeneratorAdapter

        return {"GeneratorAdapter": GeneratorAdapter, "CTABGANPlusAdapter": CTABGANPlusAdapter}[name]
    raise AttributeError(name)
