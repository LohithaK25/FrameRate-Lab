#!/usr/bin/env python3
"""Plot benchmark CSV output. Requires the optional matplotlib dependency."""
from __future__ import annotations

import argparse
import csv
from pathlib import Path


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=Path, default=Path("results/raw/benchmark.csv"))
    parser.add_argument("--output-dir", type=Path, default=Path("results/figures"))
    args = parser.parse_args()

    try:
        import matplotlib.pyplot as plt
    except ImportError as exc:
        raise SystemExit("Install plotting support with: pip install -e '.[plots]'") from exc

    with args.input.open(newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))

    valid = [row for row in rows if row.get("status") == "ok" and row.get("throughput_fps")]
    if not valid:
        raise SystemExit(f"No successful benchmark rows found in {args.input}")

    args.output_dir.mkdir(parents=True, exist_ok=True)
    groups: dict[str, list[dict[str, str]]] = {}
    for row in valid:
        groups.setdefault(row["precision"], []).append(row)

    fig, ax = plt.subplots(figsize=(8, 5))
    for precision, group in groups.items():
        group.sort(key=lambda item: int(item["batch_size"]))
        x = [int(item["batch_size"]) for item in group]
        y = [float(item["throughput_fps"]) for item in group]
        ax.plot(x, y, marker="o", label=precision.upper())
    ax.set_title("Throughput by batch size")
    ax.set_xlabel("Batch size")
    ax.set_ylabel("Images per second")
    ax.grid(True, alpha=0.3)
    ax.legend()
    fig.tight_layout()
    fig.savefig(args.output_dir / "throughput_by_batch_size.png", dpi=160)
    plt.close(fig)

    for metric, title, filename in [
        ("inference_p95_ms", "P95 inference latency by batch size", "p95_inference_latency.png"),
        ("end_to_end_p95_ms", "P95 end-to-end latency by batch size", "p95_end_to_end_latency.png"),
    ]:
        fig, ax = plt.subplots(figsize=(8, 5))
        for precision, group in groups.items():
            group.sort(key=lambda item: int(item["batch_size"]))
            x = [int(item["batch_size"]) for item in group]
            key = "inference_p95_ms" if metric == "inference_p95_ms" else "end_to_end_p95_ms"
            y = [float(item[key]) for item in group if item.get(key)]
            x = x[: len(y)]
            if y:
                ax.plot(x, y, marker="o", label=precision.upper())
        ax.set_title(title)
        ax.set_xlabel("Batch size")
        ax.set_ylabel("Latency (ms)")
        ax.grid(True, alpha=0.3)
        ax.legend()
        fig.tight_layout()
        fig.savefig(args.output_dir / filename, dpi=160)
        plt.close(fig)

    print(f"Saved plots to {args.output_dir}")


if __name__ == "__main__":
    main()
