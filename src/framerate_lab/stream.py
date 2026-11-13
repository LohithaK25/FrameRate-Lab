from __future__ import annotations

import queue
import threading
import time
from typing import Any

import torch

from .metrics import latency_summary
from .runtime import autocast_context, synchronize


def run_stream_simulation(
    model: torch.nn.Module,
    device: torch.device,
    target_fps: float = 30.0,
    duration_seconds: float = 10.0,
    queue_size: int = 4,
    precision: str = "fp32",
    image_size: int = 224,
    seed: int = 11,
    warmup_iterations: int = 10,
) -> dict[str, Any]:
    """Simulate a real-time frame source with a bounded latest-frame queue.

    Frames are synthetic tensors. This is a scheduler/queue experiment, not a
    measurement of a physical camera, game engine, or live graphics workload.
    ``elapsed_seconds`` covers the measured simulation and queue drain, but not
    the warm-up phase.
    """
    if target_fps <= 0 or duration_seconds <= 0 or queue_size < 1:
        raise ValueError("target_fps/duration_seconds must be > 0 and queue_size >= 1")
    if warmup_iterations < 0:
        raise ValueError("warmup_iterations must be >= 0")
    autocast_context(device, precision)  # validate config up front

    model = model.to(device).eval()
    cpu_generator = torch.Generator(device="cpu").manual_seed(seed)
    frame_pool = [
        torch.rand(3, image_size, image_size, generator=cpu_generator, dtype=torch.float32)
        for _ in range(max(4, queue_size + 2))
    ]

    warmup_start = time.perf_counter()
    first_inference_ms: float | None = None
    with torch.inference_mode():
        for iteration in range(warmup_iterations):
            inference_start = time.perf_counter()
            device_frame = frame_pool[0].unsqueeze(0).to(
                device, non_blocking=(device.type == "cuda")
            )
            with autocast_context(device, precision):
                _ = model(device_frame)
            synchronize(device)
            if iteration == 0:
                first_inference_ms = (time.perf_counter() - inference_start) * 1000.0
    warmup_duration_ms = (time.perf_counter() - warmup_start) * 1000.0

    frames: queue.Queue[tuple[int, int, torch.Tensor]] = queue.Queue(maxsize=queue_size)
    producer_done = threading.Event()
    counters = {"produced": 0, "dropped": 0, "processed": 0, "deadline_misses": 0}
    counter_lock = threading.Lock()
    age_samples_ms: list[float] = []
    frame_interval = 1.0 / target_fps

    def produce() -> None:
        frame_id = 0
        next_frame_at = time.perf_counter()
        while time.perf_counter() < stop_at:
            now = time.perf_counter()
            if now < next_frame_at:
                time.sleep(next_frame_at - now)
            captured_ns = time.perf_counter_ns()
            frame = frame_pool[frame_id % len(frame_pool)]
            item = (frame_id, captured_ns, frame)

            try:
                frames.put_nowait(item)
            except queue.Full:
                # Keep the freshest frame. Old queued frames are less useful for
                # real-time perception once their deadlines have passed.
                try:
                    frames.get_nowait()
                    frames.task_done()
                    with counter_lock:
                        counters["dropped"] += 1
                except queue.Empty:
                    pass
                try:
                    frames.put_nowait(item)
                except queue.Full:
                    with counter_lock:
                        counters["dropped"] += 1

            with counter_lock:
                counters["produced"] += 1
            frame_id += 1
            next_frame_at += frame_interval
        producer_done.set()

    def consume() -> None:
        with torch.inference_mode():
            while not producer_done.is_set() or not frames.empty():
                try:
                    _, captured_ns, cpu_frame = frames.get(timeout=0.05)
                except queue.Empty:
                    continue
                try:
                    device_frame = cpu_frame.unsqueeze(0).to(
                        device, non_blocking=(device.type == "cuda")
                    )
                    with autocast_context(device, precision):
                        _ = model(device_frame)
                    synchronize(device)
                    age_ms = (time.perf_counter_ns() - captured_ns) / 1e6
                    age_samples_ms.append(age_ms)
                    with counter_lock:
                        counters["processed"] += 1
                        if age_ms > (1000.0 / target_fps):
                            counters["deadline_misses"] += 1
                finally:
                    frames.task_done()

    start = time.perf_counter()
    stop_at = start + duration_seconds
    producer = threading.Thread(target=produce, name="frame-producer", daemon=True)
    consumer = threading.Thread(target=consume, name="inference-consumer", daemon=True)
    consumer.start()
    producer.start()
    producer.join()
    consumer.join()
    elapsed = time.perf_counter() - start

    summary = dict(counters)
    summary.update(
        {
            "target_fps": target_fps,
            "duration_seconds": duration_seconds,
            "elapsed_seconds": elapsed,
            "warmup_iterations": warmup_iterations,
            "first_inference_ms": first_inference_ms,
            "warmup_duration_ms": warmup_duration_ms,
            "queue_size": queue_size,
            "precision": precision,
            "device": str(device),
            "processed_fps": counters["processed"] / elapsed if elapsed else 0.0,
            "drop_rate": counters["dropped"] / counters["produced"] if counters["produced"] else 0.0,
            "deadline_miss_rate": (
                counters["deadline_misses"] / counters["processed"]
                if counters["processed"]
                else 0.0
            ),
            "frame_age": latency_summary(age_samples_ms) if age_samples_ms else None,
        }
    )
    return summary
