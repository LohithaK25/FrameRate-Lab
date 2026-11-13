from __future__ import annotations

import platform
import random
import shutil
import subprocess
import sys
from contextlib import nullcontext
from typing import Any

import numpy as np
import torch


def seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def resolve_device(requested: str) -> torch.device:
    if requested == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if requested == "cuda" and not torch.cuda.is_available():
        raise RuntimeError(
            "CUDA was requested but PyTorch cannot see an NVIDIA GPU. "
            "Use --device cpu or run this command in a CUDA-enabled runtime."
        )
    return torch.device(requested)


def autocast_context(device: torch.device, precision: str):
    if precision == "fp32":
        return nullcontext()
    if device.type != "cuda":
        raise ValueError(f"{precision} autocast is only enabled for CUDA in this project.")
    if precision == "fp16":
        return torch.autocast(device_type="cuda", dtype=torch.float16)
    if precision == "bf16":
        if not torch.cuda.is_bf16_supported():
            raise ValueError("This CUDA device does not report BF16 support.")
        return torch.autocast(device_type="cuda", dtype=torch.bfloat16)
    raise ValueError(f"Unknown precision: {precision}")


def synchronize(device: torch.device) -> None:
    if device.type == "cuda":
        torch.cuda.synchronize(device)


def environment_info(device: torch.device) -> dict[str, Any]:
    info: dict[str, Any] = {
        "python": sys.version.split()[0],
        "platform": platform.platform(),
        "torch": torch.__version__,
        "torchvision": _torchvision_version(),
        "device": str(device),
        "cuda_available": torch.cuda.is_available(),
    }
    if device.type == "cuda":
        props = torch.cuda.get_device_properties(device)
        info.update(
            {
                "gpu_name": props.name,
                "gpu_total_memory_bytes": props.total_memory,
                "cuda_runtime": torch.version.cuda,
                "bf16_supported": torch.cuda.is_bf16_supported(),
                "nvidia_smi": _nvidia_smi_summary(),
            }
        )
    else:
        info["gpu_name"] = None
    return info


def _torchvision_version() -> str:
    try:
        import torchvision

        return torchvision.__version__
    except Exception:
        return "unavailable"


def _nvidia_smi_summary() -> str | None:
    executable = shutil.which("nvidia-smi")
    if not executable:
        return None
    try:
        completed = subprocess.run(
            [executable, "--query-gpu=driver_version,name,memory.total", "--format=csv,noheader"],
            check=True,
            capture_output=True,
            text=True,
            timeout=5,
        )
        return completed.stdout.strip()
    except (subprocess.SubprocessError, OSError):
        return None
