import torch

from framerate_lab.models import build_model


def test_resnet18_forward_shape_without_downloading_weights():
    model = build_model("none")
    model.eval()
    with torch.inference_mode():
        output = model(torch.randn(1, 3, 224, 224))
    assert output.shape == (1, 1000)


def test_benchmark_runs_with_a_small_cpu_identity_model():
    from framerate_lab.benchmark import run_benchmark

    results = run_benchmark(
        model=torch.nn.Identity(),
        device=torch.device("cpu"),
        batch_sizes=[1],
        precisions=["fp32"],
        warmup=0,
        iterations=1,
        image_size=8,
    )
    assert results[0]["status"] == "ok"
    assert results[0]["throughput_fps"] > 0
    assert results[0]["end_to_end"]["mean_ms"] >= 0
