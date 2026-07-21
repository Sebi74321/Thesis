"""Adapters exposing a common dataframe-in/dataframe-out generator API."""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any, Dict, Optional

import pandas as pd
import torch
from pandas.api.types import is_bool_dtype, is_numeric_dtype

from model.pipeline.data_preparation import DataPrep
from model.synthesizer.ctabgan_synthesizer import CTABGANSynthesizer

from .device import seed_everything


class GeneratorAdapter(ABC):
    @abstractmethod
    def fit(self, df: pd.DataFrame) -> None:
        """Fit the generator on every row in ``df``."""

    @abstractmethod
    def sample(self, n: int) -> pd.DataFrame:
        """Return exactly ``n`` synthetic rows."""


class CTABGANPlusAdapter(GeneratorAdapter):
    """CTAB-GAN+ adapter that bypasses its legacy internal train/test split."""

    def __init__(
        self,
        *,
        categorical_columns,
        log_columns=None,
        mixed_columns=None,
        general_columns=None,
        non_categorical_columns=None,
        integer_columns=None,
        problem_type=None,
        class_dim=(256, 256),
        random_dim=100,
        num_channels=64,
        l2scale=1e-5,
        batch_size=512,
        epochs=150,
        snapshot_frq: Optional[int] = None,
        device: torch.device | str = "cpu",
        seed: int = 42,
        deterministic: bool = True,
        allow_tf32: bool = False,
        progress: str = "auto",
        progress_label: str = "CTAB-GAN+",
    ):
        self.categorical_columns = list(categorical_columns)
        self.log_columns = list(log_columns or [])
        self.mixed_columns = dict(mixed_columns or {})
        self.general_columns = list(general_columns or [])
        self.non_categorical_columns = list(non_categorical_columns or [])
        self.integer_columns = list(integer_columns or [])
        self.problem_type: Dict[str, Any] = dict(problem_type or {None: None})
        self.synthesizer_kwargs = {
            "class_dim": tuple(class_dim),
            "random_dim": random_dim,
            "num_channels": num_channels,
            "l2scale": l2scale,
            "batch_size": batch_size,
            "epochs": epochs,
            "snapshot_frq": snapshot_frq,
            "device": str(device),
            "progress": progress,
            "progress_label": progress_label,
        }
        self.device = torch.device(device)
        self.seed = int(seed)
        self.deterministic = bool(deterministic)
        self.allow_tf32 = bool(allow_tf32)
        self.columns = None
        self.dtypes = None
        self.data_prep = None
        self.synthesizer = None
        self.discriminator_snapshots = []
        self._sample_calls = 0

    def fit(self, df: pd.DataFrame) -> None:
        if df.empty:
            raise ValueError("Cannot fit CTAB-GAN+ on an empty dataframe")
        seed_everything(self.seed, self.deterministic, self.allow_tf32)
        train_df = df.copy(deep=True).reset_index(drop=True)
        self.columns = train_df.columns.tolist()
        self.dtypes = train_df.dtypes.to_dict()

        # Passing a null problem type prevents DataPrep from performing its
        # legacy split. The real problem type is still supplied to the
        # synthesizer below, preserving conditional classification training.
        self.data_prep = DataPrep(
            train_df,
            self.categorical_columns,
            self.log_columns,
            self.mixed_columns.copy(),
            self.general_columns,
            self.non_categorical_columns,
            self.integer_columns,
            {None: None},
            0.0,
        )
        if len(self.data_prep.df) != len(train_df):
            raise RuntimeError("Adapter preprocessing unexpectedly discarded training rows")

        self.synthesizer = CTABGANSynthesizer(**self.synthesizer_kwargs)
        self._sample_calls = 0
        self.discriminator_snapshots = self.synthesizer.fit(
            train_data=self.data_prep.df,
            categorical=self.data_prep.column_types["categorical"],
            mixed=self.data_prep.column_types["mixed"],
            general=self.data_prep.column_types["general"],
            non_categorical=self.data_prep.column_types["non_categorical"],
            type=self.problem_type,
        )

    def sample(self, n: int) -> pd.DataFrame:
        if self.synthesizer is None or self.data_prep is None or self.columns is None:
            raise RuntimeError("fit() must be called before sample()")
        if n < 0:
            raise ValueError("n must be non-negative")
        if n == 0:
            return pd.DataFrame(columns=self.columns)
        seed_everything(self.seed + self._sample_calls, self.deterministic, self.allow_tf32)
        self._sample_calls += 1
        encoded = self.synthesizer.sample(n)
        result = self.data_prep.inverse_prep(encoded).loc[:, self.columns].reset_index(drop=True)
        for column, dtype in self.dtypes.items():
            if is_bool_dtype(dtype):
                normalized = result[column].astype(str).str.lower()
                if not normalized.isin({"true", "false"}).all():
                    raise RuntimeError(f"Cannot restore boolean dtype for generated column '{column}'")
                result[column] = normalized.map({"true": True, "false": False}).astype(dtype)
            elif is_numeric_dtype(dtype):
                result[column] = pd.to_numeric(result[column], errors="raise").astype(dtype)
            else:
                result[column] = result[column].astype(dtype)
        if len(result) != n:
            raise RuntimeError(f"Generator returned {len(result)} rows; expected {n}")
        return result
