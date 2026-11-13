from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch

from . import __version__
from .benchmark import run_benchmark
from .models import build_model
from .profiling import profile_model
from .reporting import save_benchmark_csv, save_json
from .runtime import environment_info, resolve_device, seed_everything
from .stream import run_stream_simulation


def _csv_ints(value: str) -> list[int]:
    try:
        values = [int(part.strip()) for part in value.split(",") if part.strip()]
    except ValueError as exc:
        raise argparse.ArgumentTypeError("Expected comma-separated integers") from exc
    if not values or any(item < 1 for item in values):
        raise argparse.ArgumentTypeError("Values must be positive integers")
    return values


def _csv_strings(value: str) -> list[str]:
    values = [part.strip().lower() for part in value.split(",") if part.strip()]
    if not values:
        raise argparse.ArgumentTypeError("Provide at least one value")
    allowed = {"fp32", "fp16", "bf16"}
    if any(item not in allowed for item in values):
        raise argparse.ArgumentTypeError(f"Values must be selected from {sorted(allowed)}")
    return values


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="framerate-lab",
        description="Benchmark and profile real-time computer-vision inference workloads.",
    )
    parser.add_argument("--version", action="version", version=f"FrameRate Lab {__version__}")
    subparsers = parser.add_subparsers(dest="command", required=True)

    benchmark = subparsers.add_parser("benchmark", help="Sweep batch sizes and precision modes")
    benchmark.add_argument("--device", choices=["auto", "cpu", "cuda"], default="auto")
    benchmark.add_argument("--weights", choices=["DEFAULT", "none"], default="DEFAULT")
    benchmark.add_argument("--batch-sizes", type=_csv_ints, default=[1, 2, 4, 8])
    benchmark.add_argument("--precisions", type=_csv_strings, default=["fp32", "fp16"])
    benchmark.add_argument("--warmup", type=int, default=20)
    benchmark.add_argument("--iterations", type=int, default=100)
    benchmark.add_argument("--image-size", type=int, default=224)
    benchmark.add_argument("--seed", type=int, default=7)
    benchmark.add_argument("--output", type=Path, default=Path("results/raw/benchmark.json"))
    benchmark.add_argument("--csv", type=Path, default=Path("results/raw/benchmark.csv"))

    stream = subparsers.add_parser("stream", help="Simulate a real-time frame source and bounded queue")
    stream.add_argument("--device", choices=["auto", "cpu", "cuda"], default="auto")
    stream.add_argument("--weights", choices=["DEFAULT", "none"], default="DEFAULT")
    stream.add_argument("--target-fps", type=float, default=30.0)
    stream.add_argument("--duration", type=float, default=10.0)
    stream.add_argument("--queue-size", type=int, default=4)
    stream.add_argument("--precision", choices=["fp32", "fp16", "bf16"], default="fp32")
    stream.add_argument("--image-size", type=int, default=224)
    stream.add_argument("--seed", type=int, default=11)
    stream.add_argument("--output", type=Path, default=Path("results/raw/stream.json"))

    profile = subparsers.add_parser("profile", help="Export a PyTorch Profiler Chrome trace")
    profile.add_argument("--device", choices=["auto", "cpu", "cuda"], default="auto")
    profile.add_argument("--weights", choices=["DEFAULT", "none"], default="DEFAULT")
    profile.add_argument("--batch-size", type=int, default=1)
    profile.add_argument("--precision", choices=["fp32", "fp16", "bf16"], default="fp32")
    profile.add_argument("--image-size", type=int, default=224)
    profile.add_argument("--warmup", type=int, default=5)
    profile.add_argument("--steps", type=int, default=10)
    profile.add_argument("--output", type=Path, default=Path("results/traces/resnet18_trace.json"))

    return parser


def main() -> None:
    args = build_parser().parse_args()
    seed_everything(getattr(args, "seed", 7))
    device = resolve_device(args.device)
    print(f"Device: {device}")

    if args.command == "benchmark":
        model = build_model(args.weights)
        results = run_benchmark(
            model=model,
            device=device,
            batch_sizes=args.batch_sizes,
            precisions=args.precisions,
            warmup=args.warmup,
            iterations=args.iterations,
            image_size=args.image_size,
            seed=args.seed,
        )
        payload = {
            "project": "FrameRate Lab",
            "model": "torchvision ResNet-18",
            "weights": args.weights,
            "environment": environment_info(device),
            "method": {
                "timing": "perf_counter_ns with CUDA synchronization around measured regions",
                "input": "synthetic random RGB tensors; no dataset or video decode included",
                "transfer": "CPU-to-device transfer measured separately",
                "inference": "model forward pass measured separately",
                "end_to_end": "transfer plus model forward pass",
            },
            "results": results,
        }
        save_json(payload, args.output)
        save_benchmark_csv(results, args.csv)
        print(json.dumps({"json": str(args.output), "csv": str(args.csv), "results": results}, indent=2))

    elif args.command == "stream":
        model = build_model(args.weights)
        result = run_stream_simulation(
            model=model,
            device=device,
            target_fps=args.target_fps,
            duration_seconds=args.duration,
            queue_size=args.queue_size,
            precision=args.precision,
            image_size=args.image_size,
            seed=args.seed,
        )
        payload = {
            "project": "FrameRate Lab",
            "model": "torchvision ResNet-18",
            "weights": args.weights,
            "environment": environment_info(device),
            "method": "Synthetic frame-arrival simulation with a bounded latest-frame queue; not a physical camera or game-engine benchmark.",
            "result": result,
        }
        save_json(payload, args.output)
        print(json.dumps(payload, indent=2))

    elif args.command == "profile":
        model = build_model(args.weights)
        result = profile_model(
            model=model,
            device=device,
            output_path=args.output,
            batch_size=args.batch_size,
            image_size=args.image_size,
            precision=args.precision,
            warmup=args.warmup,
            steps=args.steps,
        )
        print(result["summary_table"])
        print(f"Chrome trace saved to: {result['trace_path']}")


if __name__ == "__main__":
    main()
