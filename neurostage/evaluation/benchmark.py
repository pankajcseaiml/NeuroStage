"""
Efficiency benchmarking: parameter count, model size, CPU inference latency.

These metrics are first-class in NeuroStage — not an afterthought.
The accuracy-vs-efficiency trade-off curve is a core research output.
"""

from __future__ import annotations

import os
import time
from typing import Any

import torch
import torch.nn as nn


def count_parameters(model: nn.Module) -> dict[str, int]:
    """Count total and trainable parameters."""
    total = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    return {"total_params": total, "trainable_params": trainable}


def model_size_mb(model: nn.Module, tmp_path: str = "/tmp/neurostage_bench.pt") -> float:
    """Measure model disk size (FP32) in MB."""
    torch.save(model.state_dict(), tmp_path)
    size_bytes = os.path.getsize(tmp_path)
    os.remove(tmp_path)
    return size_bytes / (1024 ** 2)


def benchmark_latency(
    model: nn.Module,
    input_shape: tuple = (1, 20, 1, 3000),
    n_warmup: int = 50,
    n_runs: int = 200,
    device: str = "cpu",
) -> dict[str, float]:
    """
    Measure CPU (or GPU) inference latency in milliseconds.

    Returns mean, p50, p95, p99 latency across n_runs.
    All measurements are wall-clock time for a single forward pass.
    """
    model.eval()
    model.to(device)
    dummy = torch.zeros(*input_shape, device=device)

    # Warm up
    with torch.no_grad():
        for _ in range(n_warmup):
            _ = model(dummy)

    latencies_ms = []
    with torch.no_grad():
        for _ in range(n_runs):
            t0 = time.perf_counter()
            _ = model(dummy)
            t1 = time.perf_counter()
            latencies_ms.append((t1 - t0) * 1000.0)

    arr = sorted(latencies_ms)
    return {
        "latency_mean_ms": float(sum(arr) / len(arr)),
        "latency_p50_ms": float(arr[len(arr) // 2]),
        "latency_p95_ms": float(arr[int(len(arr) * 0.95)]),
        "latency_p99_ms": float(arr[int(len(arr) * 0.99)]),
        "n_runs": n_runs,
        "device": device,
        "input_shape": list(input_shape),
    }


def full_efficiency_report(
    model: nn.Module,
    cfg: Any,
    device: str = "cpu",
) -> dict:
    """Combine parameter count, model size, and latency into one report."""
    context_epochs = cfg.data.context_epochs
    n_samples = int(cfg.data.sfreq * cfg.data.epoch_sec)
    input_shape = (1, context_epochs, cfg.model.cnn.in_channels, n_samples)

    params = count_parameters(model)
    size = model_size_mb(model)
    latency = benchmark_latency(
        model,
        input_shape=input_shape,
        n_warmup=cfg.evaluation.benchmark_n_warmup,
        n_runs=cfg.evaluation.benchmark_n_runs,
        device=device,
    )
    return {**params, "model_size_mb_fp32": size, **latency}
