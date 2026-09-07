"""
Magnitude-based weight pruning.

Type: Unstructured L1-norm magnitude pruning applied globally across all
      Conv1d and Linear weight tensors.

Why unstructured:
  Achieves a good sparsity ratio without requiring hardware-specific
  structured-pruning support. For actual edge deployment, structured pruning
  or channel pruning would give better speedups on bare-metal hardware.
  The current implementation prioritizes parameter-count reduction and
  demonstrability for the research prototype.

Pruning workflow:
  1. apply_pruning()     — mark weights to zero, model still has original shape
  2. fine_tune()         — optional: re-train for n epochs on train set
  3. remove_pruning()    — make zeroing permanent, reduce effective param count

The before/after parameter count and accuracy are logged to MLflow by compress.py.
"""

from __future__ import annotations

import logging

import torch.nn as nn
import torch.nn.utils.prune as prune

log = logging.getLogger(__name__)


def _get_prunable_modules(model: nn.Module) -> list[tuple[nn.Module, str]]:
    """Return (module, param_name) pairs for all Conv1d and Linear layers."""
    modules_to_prune = []
    for name, module in model.named_modules():
        if isinstance(module, (nn.Conv1d, nn.Linear)):
            modules_to_prune.append((module, "weight"))
    return modules_to_prune


def apply_global_pruning(model: nn.Module, amount: float = 0.3) -> nn.Module:
    """
    Apply global unstructured L1-magnitude pruning.

    amount: fraction of weights to zero globally (e.g. 0.3 = 30% zeroed).
    Returns the model with pruning masks applied (weights not yet permanently removed).
    """
    modules = _get_prunable_modules(model)
    prune.global_unstructured(
        modules,
        pruning_method=prune.L1Unstructured,
        amount=amount,
    )
    sparsity = _compute_global_sparsity(model)
    log.info(f"Applied global pruning (amount={amount}). Global sparsity: {sparsity:.2%}")
    return model


def remove_pruning_masks(model: nn.Module) -> nn.Module:
    """Make pruning permanent (zero out weights and remove masks)."""
    for module in model.modules():
        if isinstance(module, (nn.Conv1d, nn.Linear)):
            try:
                prune.remove(module, "weight")
            except ValueError:
                pass  # Not pruned — skip
    log.info("Pruning masks removed — weights permanently zeroed.")
    return model


def _compute_global_sparsity(model: nn.Module) -> float:
    """Compute fraction of zero-valued parameters across the model."""
    total = 0
    zeros = 0
    for module in model.modules():
        if isinstance(module, (nn.Conv1d, nn.Linear)):
            w = module.weight
            total += w.numel()
            zeros += int((w == 0).sum().item())
    return zeros / total if total > 0 else 0.0


def get_sparsity_report(model: nn.Module) -> dict:
    """Per-layer sparsity report."""
    report = {}
    for name, module in model.named_modules():
        if isinstance(module, (nn.Conv1d, nn.Linear)):
            w = module.weight.data
            sparsity = (w == 0).sum().item() / w.numel()
            report[name] = {
                "shape": list(w.shape),
                "sparsity": round(sparsity, 4),
                "nonzero_params": int((w != 0).sum().item()),
                "total_params": w.numel(),
            }
    return report
