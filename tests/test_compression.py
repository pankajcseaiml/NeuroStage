"""Unit tests for model compression (pruning & quantization)."""
from __future__ import annotations

import torch
import torch.nn as nn

from neurostage.compression.pruning import (
    _compute_global_sparsity,
    apply_global_pruning,
    get_sparsity_report,
    remove_pruning_masks,
)
from neurostage.compression.quantization import (
    apply_dynamic_quantization,
    measure_quantization_reduction,
)


class SimpleModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv = nn.Conv1d(1, 16, kernel_size=5, padding=2)
        self.fc1 = nn.Linear(16 * 100, 64)
        self.fc2 = nn.Linear(64, 5)

    def forward(self, x):
        h = self.conv(x)
        h = h.flatten(1)
        h = self.fc1(h)
        return self.fc2(h)


def test_global_pruning():
    model = SimpleModel()
    initial_sparsity = _compute_global_sparsity(model)
    assert initial_sparsity < 0.01

    # Apply 40% pruning
    pruned_model = apply_global_pruning(model, amount=0.4)
    sparsity_after = _compute_global_sparsity(pruned_model)
    assert 0.35 <= sparsity_after <= 0.45

    report = get_sparsity_report(pruned_model)
    assert "conv" in report
    assert "fc1" in report
    assert report["fc1"]["total_params"] > 0

    # Remove masks and verify model still forwards
    clean_model = remove_pruning_masks(pruned_model)
    x = torch.randn(2, 1, 100)
    out = clean_model(x)
    assert out.shape == (2, 5)


def test_dynamic_quantization():
    model = SimpleModel()
    model.eval()

    quantized = apply_dynamic_quantization(model)
    assert quantized is not None

    reduction = measure_quantization_reduction(model, quantized)
    assert "fp32_size_mb" in reduction
    assert "int8_size_mb" in reduction
    assert reduction["fp32_size_mb"] > 0
