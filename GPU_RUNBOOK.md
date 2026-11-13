# Free-GPU runbook

Use a free accelerator first. Do not buy a GPU instance for this project.

## Recommended: Kaggle Notebooks

Kaggle's official notebook documentation currently describes free NVIDIA GPU accelerators, including Tesla P100 and, depending on the environment, T4 options. Availability can be limited and queues may occur. Check the accelerator shown in your own session rather than assuming a particular model.

1. Create a new Kaggle Notebook.
2. Open **Settings → Accelerator → GPU**.
3. Enable Internet if you plan to clone the public GitHub repository or install packages from PyPI.
4. In the first cell, verify the actual hardware:

```python
!nvidia-smi
import torch
print("PyTorch:", torch.__version__)
print("CUDA available:", torch.cuda.is_available())
if torch.cuda.is_available():
    print("GPU:", torch.cuda.get_device_name(0))
    print("BF16 supported:", torch.cuda.is_bf16_supported())
```

5. After pushing this project to a public GitHub repository, clone it:

```bash
!git clone https://github.com/LohithaK25/FrameRate-Lab.git
%cd FrameRate-Lab
!pip install -e ".[plots,test]"
```

6. Run a short smoke test before a longer sweep:

```bash
!framerate-lab benchmark --device cuda --batch-sizes 1,2 --precisions fp32,fp16 --warmup 5 --iterations 20
```

7. If the smoke test succeeds, run the full sweep:

```bash
!framerate-lab benchmark --device cuda --batch-sizes 1,2,4,8 --precisions fp32,fp16 --warmup 20 --iterations 100
```

8. Run the queue experiment and profiler:

```bash
!framerate-lab stream --device cuda --target-fps 30 --duration 15 --queue-size 4
!framerate-lab profile --device cuda --batch-size 1 --precision fp32 --warmup 5 --steps 10
```

9. Generate charts:

```bash
!python scripts/plot_results.py
```

10. Download the `results/raw` and `results/figures` folders before ending the session. Stop the notebook session when finished so it does not consume quota unnecessarily.

## Backup: Google Colab free runtime

1. Open a new Colab notebook.
2. Select **Runtime → Change runtime type → T4 GPU**, if offered.
3. Clone the public repository and install it using the same commands above.
4. Confirm GPU availability with `nvidia-smi` and `torch.cuda.is_available()`.

Colab's free GPU access is not guaranteed. Hardware type, session duration, and usage limits vary. If no free GPU is available, save the code and run it when an accelerator becomes available rather than paying for compute.

## Cost and experiment discipline

- No training is performed.
- No video dataset is downloaded.
- Pretrained ResNet-18 weights are downloaded once (or use `--weights none` for a quick architecture-only test).
- Start with batch sizes 1 and 2; only increase after checking memory.
- Use 5 warm-up and 20 timed iterations for a smoke run; use 20 warm-up and 100 timed iterations for a reportable sweep.
- Record exact GPU/driver/software details with every result.
- Never compare results from different GPU models as if they were a controlled before/after optimization.
