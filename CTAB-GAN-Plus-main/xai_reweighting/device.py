"""Torch device resolution and reproducibility controls."""

from __future__ import annotations

import os
import platform
import random
from typing import Any, Dict

import numpy as np
import torch


def resolve_device(spec: str = "auto") -> torch.device:
    normalized = spec.strip().lower()
    if normalized == "auto":
        return torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    if normalized == "cpu":
        return torch.device("cpu")
    if normalized == "cuda":
        normalized = "cuda:0"
    if normalized.startswith("cuda:"):
        if not torch.cuda.is_available():
            raise RuntimeError(
                f"CUDA device '{spec}' was requested, but torch.cuda.is_available() is false. "
                "Use --device cpu or install a CUDA-enabled PyTorch build."
            )
        try:
            index = int(normalized.split(":", 1)[1])
        except ValueError as exc:
            raise ValueError(f"Invalid device '{spec}'; expected cuda or cuda:N") from exc
        if index < 0 or index >= torch.cuda.device_count():
            raise RuntimeError(
                f"CUDA device index {index} is unavailable; {torch.cuda.device_count()} visible device(s)."
            )
        return torch.device(normalized)
    raise ValueError(f"Unsupported device '{spec}'; use auto, cpu, cuda, or cuda:N")


def seed_everything(seed: int, deterministic: bool = True, allow_tf32: bool = False) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    if hasattr(torch.backends, "cudnn"):
        torch.backends.cudnn.deterministic = deterministic
        torch.backends.cudnn.benchmark = not deterministic
        torch.backends.cudnn.allow_tf32 = allow_tf32
    if hasattr(torch.backends, "cuda") and hasattr(torch.backends.cuda, "matmul"):
        torch.backends.cuda.matmul.allow_tf32 = allow_tf32


def device_manifest(device: torch.device, deterministic: bool, allow_tf32: bool) -> Dict[str, Any]:
    info: Dict[str, Any] = {
        "device": str(device),
        "precision": "float32",
        "deterministic": bool(deterministic),
        "allow_tf32": bool(allow_tf32),
        "python": platform.python_version(),
        "torch": torch.__version__,
        "cuda_runtime": torch.version.cuda,
        "cuda_available": bool(torch.cuda.is_available()),
        "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES"),
        "cuda_device_count": int(torch.cuda.device_count()) if torch.cuda.is_available() else 0,
    }
    if device.type == "cuda":
        info["cuda_device_name"] = torch.cuda.get_device_name(device)
    return info
