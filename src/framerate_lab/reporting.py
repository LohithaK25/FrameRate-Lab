from __future__ import annotations

import csv
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


def save_json(payload: dict[str, Any], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    payload["created_at_utc"] = datetime.now(timezone.utc).isoformat()
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")


def save_benchmark_csv(results: list[dict[str, Any]], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    columns = [
        "batch_size",
        "precision",
        "status",
        "throughput_fps",
        "inference_mean_ms",
        "inference_p95_ms",
        "end_to_end_mean_ms",
        "end_to_end_p95_ms",
        "transfer_mean_ms",
        "peak_memory_allocated_mb",
        "peak_memory_reserved_mb",
        "reason",
    ]
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=columns)
        writer.writeheader()
        for result in results:
            inference = result.get("inference") or {}
            total = result.get("end_to_end") or {}
            transfer = result.get("transfer") or {}
            writer.writerow(
                {
                    "batch_size": result.get("batch_size"),
                    "precision": result.get("precision"),
                    "status": result.get("status"),
                    "throughput_fps": result.get("throughput_fps"),
                    "inference_mean_ms": inference.get("mean_ms"),
                    "inference_p95_ms": inference.get("p95_ms"),
                    "end_to_end_mean_ms": total.get("mean_ms"),
                    "end_to_end_p95_ms": total.get("p95_ms"),
                    "transfer_mean_ms": transfer.get("mean_ms"),
                    "peak_memory_allocated_mb": result.get("peak_memory_allocated_mb"),
                    "peak_memory_reserved_mb": result.get("peak_memory_reserved_mb"),
                    "reason": result.get("reason"),
                }
            )
