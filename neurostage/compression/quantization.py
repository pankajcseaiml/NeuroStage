"""
INT8 Post-Training Quantization (PTQ).

Uses PyTorch's native quantization API (torch.quantization).
Backend: 'qnnpack' for ARM/mobile devices, 'fbgemm' for x86.

Quantization pipeline:
  1. Set backend + fuse Conv-BN-ReLU/GELU patterns (where possible)
  2. Insert observer hooks (calibration quantization stubs)
  3. Run calibration data through the model to collect activation statistics
  4. Convert to INT8 model

Note on GELU:
  PyTorch's static quantization does not natively support GELU fusion.
  For a deployable INT8 model targeting ARM, replacing GELU with ReLU
  during quantization export is a recommended simplification.
  This implementation uses dynamic quantization (per-op, no calibration)
  as a safe fallback that works with any activation function while
  still giving meaningful latency and size reductions.
"""

from __future__ import annotations

import logging
import os

import torch
import torch.nn as nn

log = logging.getLogger(__name__)


import tempfile


def apply_dynamic_quantization(model: nn.Module, backend: str = "fbgemm") -> nn.Module:
    """
    Apply dynamic INT8 quantization to Linear and LSTM layers.

    Dynamic quantization computes scale/zero-point at runtime per-tensor.
    No calibration dataset required — makes it the simplest PTQ variant.
    Works with GELU activations without modification.

    Returns a quantized model (new object, original model unchanged).
    """
    supported = getattr(torch.backends.quantized, "supported_engines", [])
    chosen_backend = backend if backend in supported else ("fbgemm" if "fbgemm" in supported else ("qnnpack" if "qnnpack" in supported else None))

    if chosen_backend:
        try:
            torch.backends.quantized.engine = chosen_backend
        except Exception as e:
            log.warning(f"Could not set quantized engine {chosen_backend}: {e}")

    quantized = torch.quantization.quantize_dynamic(
        model,
        qconfig_spec={nn.Linear},
        dtype=torch.qint8,
        inplace=False,
    )
    log.info(f"Applied dynamic INT8 quantization (backend={chosen_backend or 'default'})")
    return quantized


def measure_quantization_reduction(
    original_model: nn.Module, quantized_model: nn.Module, tmp_dir: str | None = None
) -> dict:
    """Compare FP32 vs INT8 model sizes on disk."""
    base_dir = tmp_dir or tempfile.gettempdir()
    fp32_path = os.path.join(base_dir, "fp32_model.pt")
    int8_path = os.path.join(base_dir, "int8_model.pt")

    torch.save(original_model.state_dict(), fp32_path)
    torch.save(quantized_model.state_dict(), int8_path)

    fp32_mb = os.path.getsize(fp32_path) / (1024 ** 2)
    int8_mb = os.path.getsize(int8_path) / (1024 ** 2)

    try:
        os.remove(fp32_path)
        os.remove(int8_path)
    except Exception:
        pass

    reduction_pct = (1 - int8_mb / fp32_mb) * 100 if fp32_mb > 0 else 0

    return {
        "fp32_size_mb": round(fp32_mb, 3),
        "int8_size_mb": round(int8_mb, 3),
        "size_reduction_pct": round(reduction_pct, 1),
    }
