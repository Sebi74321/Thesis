import numpy as np
import pandas as pd
import pytest

from xai_reweighting.augmentation import (
    create_uniform_augmentation,
    create_weighted_augmentation,
)
from xai_reweighting.data_split import create_data_splits


def test_stratified_splits_are_complete_and_disjoint():
    df = pd.DataFrame({"x": range(100), "target": [0] * 80 + [1] * 20})
    splits = create_data_splits(df, "target", seed=42)
    assert [len(splits.train), len(splits.audit), len(splits.val), len(splits.test)] == [60, 20, 10, 10]
    flattened = [i for values in splits.indices.values() for i in values]
    assert sorted(flattened) == list(range(100))
    assert all(np.isclose(part["target"].mean(), 0.2) for part in (splits.train, splits.audit, splits.val, splits.test))


def test_augmentations_preserve_original_rows_and_size():
    df = pd.DataFrame({"value": range(20)})
    uniform, _ = create_uniform_augmentation(df, gamma=0.25, seed=1)
    weighted, _ = create_weighted_augmentation(df, np.linspace(1, 2, len(df)), gamma=0.25, seed=1)
    assert len(uniform) == len(weighted) == 25
    pd.testing.assert_frame_equal(uniform.iloc[:20].reset_index(drop=True), df)
    pd.testing.assert_frame_equal(weighted.iloc[:20].reset_index(drop=True), df)


def test_device_resolution_and_invalid_spec():
    pytest.importorskip("torch")
    from xai_reweighting.device import resolve_device

    assert str(resolve_device("cpu")) == "cpu"
    with pytest.raises(ValueError):
        resolve_device("gpu")
