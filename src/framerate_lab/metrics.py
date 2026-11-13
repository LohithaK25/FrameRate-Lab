from __future__ import annotations

from typing import Iterable

import numpy as np


def latency_summary(samples_ms: Iterable[float]) -> dict[str, float]:
    values = np.asarray(list(samples_ms), dtype=np.float64)
    if values.size == 0:
        raise ValueError("At least one latency sample is required")
    if not np.isfinite(values).all():
        raise ValueError("Latency samples must be finite")
    return {
        "mean_ms": float(np.mean(values)),
        "median_ms": float(np.percentile(values, 50)),
        "p95_ms": float(np.percentile(values, 95)),
        "p99_ms": float(np.percentile(values, 99)),
        "min_ms": float(np.min(values)),
        "max_ms": float(np.max(values)),
    }
