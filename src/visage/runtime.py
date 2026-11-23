"""Explicit reproducibility, project-local caches, and environment evidence."""

from __future__ import annotations

import hashlib
import importlib.metadata
import json
import os
import platform
import random
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def configure_caches(root: Path) -> None:
    for name, relative in {
        "HF_HOME": ".cache/huggingface", "TORCH_HOME": ".cache/torch",
        "MPLCONFIGDIR": ".cache/matplotlib", "PIP_CACHE_DIR": ".cache/pip",
    }.items():
        path = root.resolve() / relative
        path.mkdir(parents=True, exist_ok=True)
        os.environ.setdefault(name, str(path))
    os.environ.setdefault("HF_HUB_DISABLE_XET", "1")
    os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")


def seed_everything(seed: int = 42, *, threads: int = 4) -> None:
    import numpy as np
    import torch
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.set_num_threads(threads)
    torch.use_deterministic_algorithms(True)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False


def sha256_file(path: Path) -> str:
    with Path(path).open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def write_json(path: Path, value) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    partial = path.with_suffix(path.suffix + ".partial")
    partial.write_text(json.dumps(value, indent=2, allow_nan=False), encoding="utf-8")
    partial.replace(path)


def environment_info() -> dict:
    import torch
    packages = {}
    for name in ("torch", "torchvision", "timm", "numpy", "scipy", "pillow", "pytest"):
        try:
            packages[name] = importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError:
            packages[name] = None
    result = {
        "recorded_utc": utc_now(), "python": sys.version, "python_executable": sys.executable,
        "platform": platform.platform(), "packages": packages,
        "torch_cuda_build": torch.version.cuda, "cuda_available": torch.cuda.is_available(),
        "torch_threads": torch.get_num_threads(), "deterministic_algorithms": torch.are_deterministic_algorithms_enabled(),
    }
    if torch.cuda.is_available():
        device = torch.cuda.get_device_properties(0)
        result["gpu"] = {"name": device.name, "vram_bytes": device.total_memory,
                         "capability": list(torch.cuda.get_device_capability(0))}
    try:
        query = subprocess.run(["nvidia-smi", "--query-gpu=name,driver_version,memory.total", "--format=csv,noheader"],
                               capture_output=True, text=True, check=True)
        result["nvidia_smi"] = query.stdout.strip()
    except (OSError, subprocess.CalledProcessError):
        result["nvidia_smi"] = None
    return result
