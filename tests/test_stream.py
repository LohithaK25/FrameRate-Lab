import pytest
import torch

from framerate_lab.stream import run_stream_simulation


def test_stream_simulation_reports_warmup_metadata():
    result = run_stream_simulation(
        model=torch.nn.Identity(),
        device=torch.device("cpu"),
        target_fps=10,
        duration_seconds=0.1,
        queue_size=2,
        image_size=8,
        warmup_iterations=3,
    )

    assert result["warmup_iterations"] == 3
    assert result["first_inference_ms"] is not None
    assert result["first_inference_ms"] >= 0
    assert result["warmup_duration_ms"] >= result["first_inference_ms"]
    assert result["produced"] >= 1
    assert result["processed"] >= 1
    assert result["processed"] <= result["produced"]


def test_stream_simulation_supports_zero_warmup():
    result = run_stream_simulation(
        model=torch.nn.Identity(),
        device=torch.device("cpu"),
        target_fps=10,
        duration_seconds=0.1,
        queue_size=2,
        image_size=8,
        warmup_iterations=0,
    )

    assert result["warmup_iterations"] == 0
    assert result["first_inference_ms"] is None
    assert result["warmup_duration_ms"] >= 0


def test_stream_simulation_rejects_negative_warmup():
    with pytest.raises(ValueError, match="warmup_iterations"):
        run_stream_simulation(
            model=torch.nn.Identity(),
            device=torch.device("cpu"),
            warmup_iterations=-1,
        )
