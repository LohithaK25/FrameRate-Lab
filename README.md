# FrameRate Lab

**A reproducible performance lab for real-time computer-vision inference.**

FrameRate Lab investigates a practical question: when frames arrive continuously, how do model choice, batch size, numeric precision, host-to-device transfer, and queue policy affect latency, throughput, memory use, and deadline misses?

The first workload is **torchvision ResNet-18**, the spatial backbone used in the TRIGUARD-X capstone project. This repository is a new, standalone benchmark harness. It does **not** reproduce the full TRIGUARD-X three-stream deepfake classifier, train a model, or require the Celeb-DF dataset.

## Why this exists

Real-time inference is not just a model-forward-pass problem. A useful performance investigation must distinguish:

- data transfer from model execution;
- per-request latency from aggregate throughput;
- average latency from tail latency;
- a healthy inference process from a pipeline that is dropping or serving stale frames;
- larger batches that improve throughput from the extra queueing delay they may introduce.

This project measures those concerns with synthetic RGB tensors so experiments are reproducible without downloading video datasets.

## Current scope

- ResNet-18 inference with optional pretrained ImageNet weights.
- CPU or CUDA execution, selected automatically or explicitly.
- FP32 and CUDA autocast FP16/BF16 where the device supports them.
- Batch-size sweeps.
- Separate host-to-device transfer, model-forward, and combined latency measurements.
- Mean, median, P95, P99, minimum, and maximum latency summaries.
- CUDA peak allocated and reserved memory reporting.
- A real-time frame-arrival simulation with a bounded latest-frame queue, explicit drop accounting, frame age, and deadline-miss measurements.
- PyTorch Profiler Chrome trace export.
- JSON and CSV benchmark outputs.

## What this project does not claim

- It is not a game-engine benchmark and does not measure Unreal Engine, Unity, DirectX, Vulkan, or graphics rendering.
- It does not measure a physical camera or a real video decoder. Frames are synthetic tensors.
- It does not implement CUDA kernels or claim kernel-level optimization.
- It does not claim a performance improvement until before/after measurements are recorded on the same hardware and software environment.
- CPU results must not be compared directly with GPU results from a different machine as if they were controlled experiments.

These boundaries are intentional. The aim is to produce honest, repeatable measurements and explain the bottlenecks they reveal.

## Quick start (free GPU first)

### Option A: Google Colab free runtime

1. Open a new Colab notebook and select **Runtime → Change runtime type → T4 GPU** if a free GPU is offered for the session. Availability is not guaranteed.
2. Clone this repository or upload the project ZIP.
3. Install the package:

```bash
pip install -e .[test,plots]
```

4. Check the runtime:

```bash
nvidia-smi
python -c "import torch; print(torch.__version__, torch.cuda.is_available()); print(torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'CPU runtime')"
```

5. Run a short smoke benchmark first:

```bash
framerate-lab benchmark --device auto --batch-sizes 1,2 --precisions fp32,fp16 --warmup 5 --iterations 20
```

6. Run a fuller sweep after the smoke test succeeds:

```bash
framerate-lab benchmark --device cuda --batch-sizes 1,2,4,8 --precisions fp32,fp16 --warmup 20 --iterations 100
```

Colab free GPU access, GPU type, session duration, and quotas can vary. Do not purchase compute for the first version. If no GPU is available, run the CPU smoke test and wait for a free GPU session to perform CUDA measurements.

### Option B: Local CPU smoke test

Python 3.10+ is recommended. Create a virtual environment, then install:

```bash
python -m venv .venv
# Windows PowerShell
.venv\Scripts\Activate.ps1
# macOS/Linux
source .venv/bin/activate
pip install -e .
```

Run a small test without downloading pretrained weights:

```bash
framerate-lab benchmark --device cpu --weights none --batch-sizes 1,2 --precisions fp32 --warmup 2 --iterations 5 --image-size 64
```

For a pretrained ResNet-18, use `--weights DEFAULT`. Torchvision downloads the model weights on the first run. The dataset is not downloaded.

## Experiments

### 1. Batch-size and precision sweep

```bash
framerate-lab benchmark \
  --device auto \
  --weights DEFAULT \
  --batch-sizes 1,2,4,8,16 \
  --precisions fp32,fp16 \
  --warmup 20 \
  --iterations 100
```

The command writes:

- `results/raw/benchmark.json` — full environment, method, and measurements;
- `results/raw/benchmark.csv` — flat table for analysis.

If a batch size exceeds available GPU memory, the result is recorded as `out_of_memory` and the sweep continues. Start with smaller batches on memory-limited GPUs.

**Timing method:** CUDA is synchronized around timed regions so asynchronous kernel launches are not mistaken for completed inference. Warm-up iterations are excluded. Synthetic CPU input generation is outside the timed regions. Host-to-device transfer, forward execution, and the combined path are recorded separately.

### 2. Real-time queue simulation

```bash
framerate-lab stream \
  --device auto \
  --target-fps 30 \
  --duration 10 \
  --queue-size 4 \
  --precision fp32 \
  --warmup 10
```

The producer simulates a frame source. The bounded queue drops the oldest queued frame when full, favoring fresh frames over stale work. The consumer runs inference and records frame age, deadline misses, processed FPS, and drop rate.

The configured duration starts after warm-up. Warm-up runs the model on a representative synthetic frame before the producer and measured consumer start; warm-up frames do not enter the queue or measured counters. `first_inference_ms` reports the first warm-up inference, including host-to-device transfer and device synchronization, while `warmup_duration_ms` reports the whole warm-up phase. Set `--warmup 0` to disable warm-up; in that case `first_inference_ms` is `null`.

`frame_age` measures synthetic frame creation to inference completion only during the measured queue simulation; it is not the startup inference time. `elapsed_seconds` covers the measured simulation plus any queue drain after the producer stops, and excludes warm-up. Keep startup measurements visible when investigating latency outliers: warm-up is intended to separate initialization from steady-state measurements, but repeated experiments are needed to determine whether it explains any particular frame-age outlier.

This is a controlled queue/scheduling simulation, not a measurement of an actual game or camera pipeline.

### 3. PyTorch Profiler trace

```bash
framerate-lab profile \
  --device auto \
  --batch-size 1 \
  --precision fp32 \
  --warmup 5 \
  --steps 10
```

The trace is written to `results/traces/resnet18_trace.json`. Open it with a compatible Chrome trace viewer or inspect it in Perfetto. CUDA activity is included when running on a CUDA device.

### 4. Plot benchmark results

Install the optional plotting dependency and generate throughput/P95 latency charts:

```bash
pip install -e ".[plots]"
python scripts/plot_results.py --input results/raw/benchmark.csv
```

## How to read the results

- **Inference latency:** time spent in the model forward pass, with device synchronization.
- **Transfer latency:** time for the synthetic CPU batch to move to the selected device.
- **End-to-end latency:** transfer plus forward execution as measured by this harness. It excludes camera capture, real decode, rendering, and application networking.
- **Throughput:** processed images per second across all measured iterations.
- **Peak allocated memory:** peak tensor memory allocated by PyTorch during the measured run; it is not total device memory use.
- **Peak reserved memory:** peak memory reserved by PyTorch's caching allocator; it is not equivalent to allocated memory.
- **Frame age:** elapsed time from synthetic frame creation to inference completion in the queue simulation.
- **Startup inference:** `first_inference_ms` is the first warm-up inference including transfer and synchronization; it is reported separately from steady-state frame age. `warmup_duration_ms` is the total warm-up phase duration.
- **Deadline miss:** a processed frame whose age exceeds one target frame interval.

A higher FPS number alone does not prove a better real-time experience. Compare throughput together with P95/P99 latency, memory, dropped frames, and deadline misses.

## Reproducibility checklist

For every result shared publicly, record:

- GPU model and available memory;
- driver, CUDA runtime, PyTorch, and Torchvision versions;
- operating system and Python version;
- model weights (`DEFAULT` or `none`);
- image size, batch size, precision, warm-up, and iteration count;
- whether measurements include transfer;
- exact command used;
- any OOM, unsupported operation, or skipped configuration.

Run each important configuration more than once before drawing conclusions. Avoid comparing measurements collected under different GPU power states, concurrent workloads, or runtime versions without noting the difference.

## Project roadmap

- [x] ResNet-18 synthetic-input benchmark harness.
- [x] CPU/CUDA device selection and precision validation.
- [x] Batch-size sweep and transfer/forward timing separation.
- [x] Bounded latest-frame queue simulation.
- [x] PyTorch Profiler Chrome trace export.
- [x] Add plot generation from benchmark CSV.
- [ ] Run controlled free-GPU experiments and publish raw outputs.
- [ ] Profile one workload with Nsight Systems where a compatible NVIDIA environment is available.
- [ ] Add an optimization only after profiling identifies a specific bottleneck; report before/after results.
- [ ] Explore a game/graphics frame source if time and hardware permit.

## Relation to TRIGUARD-X

TRIGUARD-X is an academic multi-stream video deepfake detection prototype whose documented spatial branch uses ResNet-18. FrameRate Lab borrows that model-family connection as a starting point for inference performance experiments. The new benchmark uses a standalone ResNet-18 image workload and synthetic inputs; it does not reuse the full temporal Transformer/ViT fusion model or claim to reproduce TRIGUARD-X accuracy.

## License

A license has not yet been selected. Add one only after confirming that all included code and assets can be licensed appropriately.
