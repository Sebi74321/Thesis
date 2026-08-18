"""Adapters exposing a common dataframe-in/dataframe-out generator API."""

from __future__ import annotations

from abc import ABC, abstractmethod
from contextlib import contextmanager, redirect_stdout
import os
from pathlib import Path
import re
import sys
from typing import Any, Dict, Mapping, Optional
import warnings

import numpy as np
import pandas as pd
import torch
from pandas.api.types import is_bool_dtype, is_float_dtype, is_numeric_dtype
from sklearn.exceptions import ConvergenceWarning

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


def _restore_schema(result: pd.DataFrame, columns, dtypes) -> pd.DataFrame:
    """Restore the fitted dataframe's order and practical pandas dtypes."""
    missing = [column for column in columns if column not in result]
    if missing:
        raise RuntimeError(f"Generator output is missing columns: {missing}")
    result = result.loc[:, columns].copy().reset_index(drop=True)
    for column, dtype in dtypes.items():
        if is_bool_dtype(dtype):
            normalized = result[column].astype(str).str.lower()
            if not normalized.isin({"true", "false", "0", "1"}).all():
                raise RuntimeError(f"Cannot restore boolean dtype for generated column '{column}'")
            result[column] = normalized.map(
                {"true": True, "false": False, "1": True, "0": False}
            ).astype(dtype)
        elif is_numeric_dtype(dtype):
            result[column] = pd.to_numeric(result[column], errors="raise")
            if not is_float_dtype(dtype):
                result[column] = result[column].round()
            result[column] = result[column].astype(dtype)
        else:
            result[column] = result[column].astype(dtype)
    return result


def _decimal_places(series: pd.Series, coverage: float = 0.99, max_places: int = 6) -> int:
    """Infer measurement precision without forcing genuinely continuous columns onto a grid."""
    if not is_float_dtype(series.dtype):
        return 0
    numeric = pd.to_numeric(series, errors="coerce").dropna().to_numpy(dtype=float)
    if len(numeric):
        for places in range(max_places + 1):
            rounded = np.round(numeric, places)
            tolerance = max(1e-10, 10.0 ** (-(places + 7)))
            if float(np.mean(np.abs(numeric - rounded) <= tolerance)) >= coverage:
                return places
    maximum = 0
    for value in series.dropna().astype(str):
        mantissa = value.lower().split("e", 1)[0]
        if "." in mantissa:
            maximum = max(maximum, len(mantissa.rsplit(".", 1)[1].rstrip("0")))
    return min(maximum, 12)


@contextmanager
def _working_directory(path: Path):
    previous = Path.cwd()
    path.mkdir(parents=True, exist_ok=True)
    os.chdir(path)
    try:
        yield
    finally:
        os.chdir(previous)


class _Tee:
    def __init__(self, stream):
        self.stream = stream
        self.parts = []

    def write(self, value):
        self.parts.append(value)
        return self.stream.write(value)

    def flush(self):
        self.stream.flush()

    @property
    def text(self):
        return "".join(self.parts)


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
        mixture_max_iter: int = 500,
        mixture_n_init: int = 3,
        mixture_tol: float = 1e-3,
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
            "mixture_max_iter": mixture_max_iter,
            "mixture_n_init": mixture_n_init,
            "mixture_tol": mixture_tol,
        }
        self.device = torch.device(device)
        self.seed = int(seed)
        self.deterministic = bool(deterministic)
        self.allow_tf32 = bool(allow_tf32)
        self.columns = None
        self.dtypes = None
        self.decimals = {}
        self.last_raw_sample = None
        self.data_prep = None
        self.synthesizer = None
        self.discriminator_snapshots = []
        self.training_history = pd.DataFrame()
        self.mixture_diagnostics = []
        self._sample_calls = 0

    def fit(self, df: pd.DataFrame) -> None:
        if df.empty:
            raise ValueError("Cannot fit CTAB-GAN+ on an empty dataframe")
        seed_everything(self.seed, self.deterministic, self.allow_tf32)
        train_df = df.copy(deep=True).reset_index(drop=True)
        self.columns = train_df.columns.tolist()
        self.dtypes = train_df.dtypes.to_dict()
        self.decimals = {column: _decimal_places(train_df[column]) for column in self.columns}

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
        try:
            self.discriminator_snapshots = self.synthesizer.fit(
                train_data=self.data_prep.df,
                categorical=self.data_prep.column_types["categorical"],
                mixed=self.data_prep.column_types["mixed"],
                general=self.data_prep.column_types["general"],
                non_categorical=self.data_prep.column_types["non_categorical"],
                type=self.problem_type,
            )
        finally:
            self.training_history = pd.DataFrame(self.synthesizer.training_history)
            self.mixture_diagnostics = list(self.synthesizer.mixture_diagnostics)

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
        self.last_raw_sample = result.copy(deep=True)
        for column, decimals in self.decimals.items():
            if column in result and is_numeric_dtype(self.dtypes[column]):
                result[column] = pd.to_numeric(result[column], errors="raise").round(decimals)
        result = _restore_schema(result, self.columns, self.dtypes)
        if len(result) != n:
            raise RuntimeError(f"Generator returned {len(result)} rows; expected {n}")
        return result

    def save_checkpoint(self, path: Path) -> None:
        if self.synthesizer is None or not hasattr(self.synthesizer, "generator"):
            raise RuntimeError("fit() must be called before save_checkpoint()")
        torch.save(
            {
                "backend": "ctabgan_plus",
                "generator_state_dict": self.synthesizer.generator.state_dict(),
                "synthesizer_kwargs": self.synthesizer_kwargs,
                "columns": self.columns,
                "dtypes": {key: str(value) for key, value in self.dtypes.items()},
                "measurement_decimals": self.decimals,
            },
            path,
        )


class CTGANAdapter(GeneratorAdapter):
    """Adapter for the standalone ``ctgan`` package."""

    def __init__(
        self,
        *,
        categorical_columns,
        embedding_dim=128,
        generator_dim=(256, 256),
        discriminator_dim=(256, 256),
        generator_lr=2e-4,
        discriminator_lr=2e-4,
        batch_size=500,
        discriminator_steps=1,
        log_frequency=True,
        verbose=True,
        epochs=300,
        pac=10,
        device: torch.device | str = "cpu",
        seed: int = 42,
        deterministic: bool = True,
        allow_tf32: bool = False,
        progress: str = "auto",
        progress_label: str = "CTGAN",
        **unused,
    ):
        if batch_size % pac:
            raise ValueError("CTGAN batch_size must be divisible by pac")
        self.categorical_columns = list(categorical_columns)
        self.device = torch.device(device)
        self.seed = int(seed)
        self.deterministic = bool(deterministic)
        self.allow_tf32 = bool(allow_tf32)
        self.progress = progress
        self.progress_label = progress_label
        self.model_kwargs = {
            "embedding_dim": int(embedding_dim),
            "generator_dim": tuple(generator_dim),
            "discriminator_dim": tuple(discriminator_dim),
            "generator_lr": float(generator_lr),
            "discriminator_lr": float(discriminator_lr),
            "batch_size": int(batch_size),
            "discriminator_steps": int(discriminator_steps),
            "log_frequency": bool(log_frequency),
            "verbose": bool(verbose and progress != "off"),
            "epochs": int(epochs),
            "pac": int(pac),
            "enable_gpu": self.device.type == "cuda",
        }
        self.columns = None
        self.dtypes = None
        self.model = None
        self.training_history = pd.DataFrame()
        self.mixture_diagnostics = []
        self.convergence_warnings = []
        self._sample_calls = 0

    def fit(self, df: pd.DataFrame) -> None:
        if df.empty:
            raise ValueError("Cannot fit CTGAN on an empty dataframe")
        try:
            from ctgan import CTGAN
        except ImportError as exc:
            raise ImportError("CTGAN requires ctgan==0.12.1; rerun setup_env.sh") from exc
        seed_everything(self.seed, self.deterministic, self.allow_tf32)
        train_df = df.copy(deep=True).reset_index(drop=True)
        self.columns = train_df.columns.tolist()
        self.dtypes = train_df.dtypes.to_dict()
        self.model = CTGAN(**self.model_kwargs)
        self.model.set_device(str(self.device))
        self._sample_calls = 0
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always", ConvergenceWarning)
            self.model.fit(train_df, discrete_columns=self.categorical_columns)
        self.convergence_warnings = [
            str(item.message) for item in caught if issubclass(item.category, ConvergenceWarning)
        ]
        losses = getattr(self.model, "loss_values", None)
        if isinstance(losses, pd.DataFrame):
            self.training_history = losses.copy()

    def sample(self, n: int) -> pd.DataFrame:
        if self.model is None or self.columns is None:
            raise RuntimeError("fit() must be called before sample()")
        if n < 0:
            raise ValueError("n must be non-negative")
        if n == 0:
            return pd.DataFrame(columns=self.columns)
        seed_everything(self.seed + self._sample_calls, self.deterministic, self.allow_tf32)
        self._sample_calls += 1
        result = _restore_schema(self.model.sample(n), self.columns, self.dtypes)
        if len(result) != n:
            raise RuntimeError(f"Generator returned {len(result)} rows; expected {n}")
        return result

    def save_checkpoint(self, path: Path) -> None:
        self.model.save(str(path))


class DPCGANAdapter(GeneratorAdapter):
    """Adapter for ``dp-cgans`` 0.2.0 using its upstream private mode."""

    NOISE_MULTIPLIER = 1.0
    DELTA = 2e-6

    def __init__(
        self,
        *,
        categorical_columns,
        generator_dim=(128, 128, 128),
        discriminator_dim=(128, 128, 128),
        generator_lr=2e-4,
        discriminator_lr=2e-4,
        batch_size=500,
        discriminator_steps=10,
        log_frequency=True,
        verbose=True,
        epochs=100,
        pac=10,
        private=True,
        saved_transformer=None,
        device: torch.device | str = "cpu",
        seed: int = 42,
        deterministic: bool = True,
        allow_tf32: bool = False,
        progress: str = "auto",
        progress_label: str = "DP-CGAN",
        work_dir: Path | str | None = None,
        **unused,
    ):
        if private is not True:
            raise ValueError("dp_cgan is reserved for upstream private=True runs")
        if batch_size % pac:
            raise ValueError("DP-CGAN batch_size must be divisible by pac")
        self.categorical_columns = list(categorical_columns)
        self.device = torch.device(device)
        self.seed = int(seed)
        self.deterministic = bool(deterministic)
        self.allow_tf32 = bool(allow_tf32)
        self.progress = progress
        self.progress_label = progress_label
        self.work_dir = Path(work_dir or f"dp_cgan_seed_{seed}").resolve()
        self.model_kwargs = {
            "generator_dim": tuple(generator_dim),
            "discriminator_dim": tuple(discriminator_dim),
            "generator_lr": float(generator_lr),
            "discriminator_lr": float(discriminator_lr),
            "batch_size": int(batch_size),
            "discriminator_steps": int(discriminator_steps),
            "log_frequency": bool(log_frequency),
            "verbose": bool(verbose and progress != "off"),
            "epochs": int(epochs),
            "pac": int(pac),
            "private": True,
            "saved_transformer": saved_transformer,
            "cuda": str(self.device) if self.device.type == "cuda" else False,
        }
        self.columns = None
        self.dtypes = None
        self.decimals = {}
        self.model = None
        self.training_history = pd.DataFrame()
        self.mixture_diagnostics = []
        self.convergence_warnings = []
        self.upstream_stdout = ""
        self.upstream_reported_epsilon = None
        self._sample_calls = 0

    def fit(self, df: pd.DataFrame) -> None:
        if df.empty:
            raise ValueError("Cannot fit DP-CGAN on an empty dataframe")
        try:
            from dp_cgans import DP_CGAN
        except ImportError as exc:
            raise ImportError("DP-CGAN requires dp-cgans==0.2.0; rerun setup_env.sh") from exc
        seed_everything(self.seed, self.deterministic, self.allow_tf32)
        train_df = df.copy(deep=True).reset_index(drop=True)
        self.columns = train_df.columns.tolist()
        self.dtypes = train_df.dtypes.to_dict()
        self.decimals = {column: _decimal_places(train_df[column]) for column in self.columns}
        for column in self.categorical_columns:
            train_df[column] = train_df[column].astype("object")
        self.model = DP_CGAN(**self.model_kwargs)
        self._sample_calls = 0
        tee = _Tee(sys.stdout)
        with _working_directory(self.work_dir), warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always", ConvergenceWarning)
            with redirect_stdout(tee):
                self.model.fit(train_df)
        self.upstream_stdout = tee.text
        matches = re.findall(r"differential privacy with eps\s*=\s*([0-9.eE+-]+)", tee.text)
        if matches:
            self.upstream_reported_epsilon = float(matches[-1])
        self.convergence_warnings = [
            str(item.message) for item in caught if issubclass(item.category, ConvergenceWarning)
        ]
        loss_files = sorted(self.work_dir.glob("*loss*.csv"))
        if loss_files:
            try:
                self.training_history = pd.read_csv(loss_files[-1])
            except Exception:
                self.training_history = pd.DataFrame()

    def sample(self, n: int) -> pd.DataFrame:
        if self.model is None or self.columns is None:
            raise RuntimeError("fit() must be called before sample()")
        if n < 0:
            raise ValueError("n must be non-negative")
        if n == 0:
            return pd.DataFrame(columns=self.columns)
        seed_everything(self.seed + self._sample_calls, self.deterministic, self.allow_tf32)
        self._sample_calls += 1
        with _working_directory(self.work_dir):
            result = self.model.sample(n)
        for column, decimals in self.decimals.items():
            if decimals and column in result:
                result[column] = pd.to_numeric(result[column], errors="coerce").round(decimals)
        result = _restore_schema(result, self.columns, self.dtypes)
        if len(result) != n:
            raise RuntimeError(f"Generator returned {len(result)} rows; expected {n}")
        return result

    def save_checkpoint(self, path: Path) -> None:
        self.model.save(str(path))


GENERATOR_NAMES = ("ctabgan_plus", "ctgan", "dp_cgan")


def create_generator(
    name: str,
    config: Mapping[str, Any],
    *,
    device: torch.device | str,
    seed: int,
    deterministic: bool = True,
    allow_tf32: bool = False,
    progress: str = "auto",
    progress_label: str | None = None,
    work_dir: Path | None = None,
) -> GeneratorAdapter:
    """Construct a fresh model adapter without importing unused backends."""
    normalized = name.strip().lower()
    if normalized not in GENERATOR_NAMES:
        raise ValueError(f"Unknown generator '{name}'; choose from {GENERATOR_NAMES}")
    classes = {
        "ctabgan_plus": CTABGANPlusAdapter,
        "ctgan": CTGANAdapter,
        "dp_cgan": DPCGANAdapter,
    }
    kwargs = dict(config)
    common = {
        "device": device,
        "seed": seed,
        "deterministic": deterministic,
        "allow_tf32": allow_tf32,
        "progress": progress,
        "progress_label": progress_label or normalized,
    }
    if normalized == "dp_cgan":
        common["work_dir"] = work_dir
    return classes[normalized](**kwargs, **common)
