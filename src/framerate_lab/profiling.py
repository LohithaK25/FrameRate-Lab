from __future__ import annotations

from pathlib import Path
from typing import Any

import torch

from .runtime import autocast_context, synchronize


def profile_model(
    model: torch.nn.Module,
    device: torch.device,
    output_path: Path,
    batch_size: int = 1,
    image_size: int = 224,
    precision: str = "fp32",
    warmup: int = 5,
    steps: int = 10,
) -> dict[str, Any]:
    """Export a Chrome trace from torch.profiler for later inspection."""
    if steps < 1 or warmup < 0:
        raise ValueError("steps must be >= 1 and warmup must be >= 0")
    model = model.to(device).eval()
    batch = torch.randn(batch_size, 3, image_size, image_size, device=device)
    activities = [torch.profiler.ProfilerActivity.CPU]
    if device.type == "cuda":
        activities.append(torch.profiler.ProfilerActivity.CUDA)

    with torch.inference_mode():
        for _ in range(warmup):
            with autocast_context(device, precision):
                _ = model(batch)
        synchronize(device)

        with torch.profiler.profile(
            activities=activities,
            record_shapes=True,
            profile_memory=True,
            with_stack=False,
        ) as prof:
            for _ in range(steps):
                with autocast_context(device, precision):
                    _ = model(batch)
                prof.step()
        synchronize(device)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    prof.export_chrome_trace(str(output_path))
    table = prof.key_averages().table(sort_by="self_cuda_time_total" if device.type == "cuda" else "self_cpu_time_total", row_limit=15)
    return {"trace_path": str(output_path), "summary_table": table}
