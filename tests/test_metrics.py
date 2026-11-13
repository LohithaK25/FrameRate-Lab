import pytest

from framerate_lab.metrics import latency_summary


def test_latency_summary_reports_percentiles():
    result = latency_summary([1.0, 2.0, 3.0, 4.0, 5.0])
    assert result["mean_ms"] == pytest.approx(3.0)
    assert result["median_ms"] == pytest.approx(3.0)
    assert result["min_ms"] == pytest.approx(1.0)
    assert result["max_ms"] == pytest.approx(5.0)
    assert result["p95_ms"] >= result["median_ms"]


def test_latency_summary_rejects_empty_values():
    with pytest.raises(ValueError):
        latency_summary([])


def test_latency_summary_rejects_nan():
    with pytest.raises(ValueError):
        latency_summary([1.0, float("nan")])


def test_stream_simulation_processes_synthetic_frames():
    import torch

    from framerate_lab.stream import run_stream_simulation

    result = run_stream_simulation(
        model=torch.nn.Identity(),
        device=torch.device("cpu"),
        target_fps=10,
        duration_seconds=0.2,
        queue_size=2,
        precision="fp32",
        image_size=8,
    )
    assert result["produced"] >= 1
    assert result["processed"] >= 1
    assert result["processed"] <= result["produced"]
