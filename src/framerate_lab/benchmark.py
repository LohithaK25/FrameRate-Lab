from __future__ import annotations

import time
from typing import Any

import torch

from .metrics import latency_summary
from .runtime import autocast_context, synchronize


def run_benchmark(
    model: torch.nn.Module,
    device: torch.device,
    batch_sizes: list[int],
    precisions: list[str],
    warmup: int,
    iterations: int,
    image_size: int = 224,
    seed: int = 7,
) -> list[dict[str, Any]]:
    """Measure transfer, forward-pass, and combined latency for each config."""
    if warmup < 0 or iterations < 1:
        raise ValueError("warmup must be >= 0 and iterations must be >= 1")
    if any(size < 1 for size in batch_sizes):
        raise ValueError("batch sizes must all be positive")

    model = model.to(device).eval()
    results: list[dict[str, Any]] = []
    generator = torch.Generator(device="cpu").manual_seed(seed)

    for precision in precisions:
        if precision != "fp32" and device.type != "cuda":
            results.append(
                {
                    "batch_size": None,
                    "precision": precision,
                    "status": "skipped",
                    "reason": "CUDA autocast is not available on CPU in this benchmark.",
                }
            )
            continue

        try:
            # Validate precision support before starting a sweep.
            autocast_context(device, precision)
        except (ValueError, RuntimeError) as exc:
            results.append(
                {
                    "batch_size": None,
                    "precision": precision,
                    "status": "skipped",
                    "reason": str(exc),
                }
            )
            continue

        for batch_size in batch_sizes:
            cpu_batch = torch.randn(
                batch_size,
                3,
                image_size,
                image_size,
                generator=generator,
                dtype=torch.float32,
                pin_memory=(device.type == "cuda"),
            )

            try:
                with torch.inference_mode():
                    # Warm-up includes both transfer and execution so lazy CUDA
                    # initialization and algorithm selection are outside samples.
                    for _ in range(warmup):
                        device_batch = cpu_batch.to(device, non_blocking=(device.type == "cuda"))
                        with autocast_context(device, precision):
                            _ = model(device_batch)
                    synchronize(device)

                    if device.type == "cuda":
                        torch.cuda.reset_peak_memory_stats(device)

                    transfer_samples: list[float] = []
                    inference_samples: list[float] = []
                    total_samples: list[float] = []

                    for _ in range(iterations):
                        synchronize(device)
                        total_start = time.perf_counter_ns()

                        transfer_start = time.perf_counter_ns()
                        device_batch = cpu_batch.to(device, non_blocking=(device.type == "cuda"))
                        synchronize(device)
                        transfer_end = time.perf_counter_ns()

                        inference_start = time.perf_counter_ns()
                        with autocast_context(device, precision):
                            _ = model(device_batch)
                        synchronize(device)
                        inference_end = time.perf_counter_ns()

                        transfer_samples.append((transfer_end - transfer_start) / 1e6)
                        inference_samples.append((inference_end - inference_start) / 1e6)
                        total_samples.append((inference_end - total_start) / 1e6)

                total_seconds = sum(total_samples) / 1000.0
                record: dict[str, Any] = {
                    "batch_size": batch_size,
                    "precision": precision,
                    "status": "ok",
                    "iterations": iterations,
                    "warmup": warmup,
                    "image_size": image_size,
                    "transfer": latency_summary(transfer_samples),
                    "inference": latency_summary(inference_samples),
                    "end_to_end": latency_summary(total_samples),
                    "throughput_fps": float(batch_size * iterations / total_seconds),
                    "peak_memory_allocated_mb": None,
                    "peak_memory_reserved_mb": None,
                }
                if device.type == "cuda":
                    record["peak_memory_allocated_mb"] = (
                        torch.cuda.max_memory_allocated(device) / (1024**2)
                    )
                    record["peak_memory_reserved_mb"] = (
                        torch.cuda.max_memory_reserved(device) / (1024**2)
                    )
                results.append(record)

            except torch.cuda.OutOfMemoryError as exc:
                if device.type == "cuda":
                    torch.cuda.empty_cache()
                results.append(
                    {
                        "batch_size": batch_size,
                        "precision": precision,
                        "status": "out_of_memory",
                        "reason": str(exc).splitlines()[0],
                    }
                )
            except RuntimeError as exc:
                # Keep an informative result for unsupported precision/operator
                # combinations, but do not hide unrelated programming errors.
                message = str(exc)
                if "not implemented" in message.lower() or "not supported" in message.lower():
                    results.append(
                        {
                            "batch_size": batch_size,
                            "precision": precision,
                            "status": "unsupported",
                            "reason": message.splitlines()[0],
                        }
                    )
                else:
                    raise

    return results
