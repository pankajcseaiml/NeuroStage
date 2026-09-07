#!/usr/bin/env python
"""
compress.py — Apply and benchmark model compression techniques.

Usage:
    python compress.py --checkpoint runs/checkpoints/neurostage_base_best.pt --method pruning
    python compress.py --checkpoint runs/checkpoints/neurostage_base_best.pt --method int8
    python compress.py --checkpoint runs/checkpoints/neurostage_base_best.pt --method all
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
if hasattr(sys.stderr, "reconfigure"):
    sys.stderr.reconfigure(encoding="utf-8", errors="replace")

import torch
from torch.utils.data import DataLoader

from neurostage.compression.pruning import (
    apply_global_pruning,
    get_sparsity_report,
    remove_pruning_masks,
)
from neurostage.compression.quantization import (
    apply_dynamic_quantization,
    measure_quantization_reduction,
)
from neurostage.config import load_config
from neurostage.data.dataset import SleepDataset
from neurostage.data.splits import subject_kfold_splits
from neurostage.evaluation.benchmark import benchmark_latency, count_parameters, model_size_mb
from neurostage.evaluation.metrics import compute_metrics
from neurostage.models.neurostage import build_model

logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)s | %(message)s")
log = logging.getLogger("compress")


def _load_model_and_data(checkpoint_path: str, cfg_path: str):
    cfg = load_config(cfg_path)
    checkpoint = torch.load(checkpoint_path, map_location="cpu")
    model = build_model(cfg)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()

    splits = subject_kfold_splits(cfg.data.manifest_path, cfg.experiment.n_folds)
    test_subjects = splits[-1]["test_subjects"]
    val_ds = SleepDataset(
        cfg.data.manifest_path,
        test_subjects,
        context_epochs=cfg.data.context_epochs,
        stride=cfg.data.context_epochs,
        augment=False,
    )
    loader = DataLoader(val_ds, batch_size=8, shuffle=False)
    return model, cfg, loader


def _quick_eval(model, loader) -> dict:
    all_preds, all_labels = [], []
    with torch.no_grad():
        for batch_x, batch_y in loader:
            logits, _ = model(batch_x)
            B, S, C = logits.shape
            all_preds.append(logits.reshape(B * S, C).argmax(dim=-1).cpu())
            all_labels.append(batch_y.reshape(B * S).cpu())
    return compute_metrics(
        torch.cat(all_labels).numpy(),
        torch.cat(all_preds).numpy(),
    )


def run_pruning(checkpoint_path: str, cfg_path: str, amount: float = 0.3) -> dict:
    log.info(f"=== PRUNING (amount={amount}) ===")
    model, cfg, loader = _load_model_and_data(checkpoint_path, cfg_path)

    params_before = count_parameters(model)
    size_before = model_size_mb(model)
    latency_before = benchmark_latency(model, n_warmup=10, n_runs=50)

    apply_global_pruning(model, amount=amount)
    remove_pruning_masks(model)

    params_after = count_parameters(model)
    size_after = model_size_mb(model)
    latency_after = benchmark_latency(model, n_warmup=10, n_runs=50)
    metrics = _quick_eval(model, loader)

    report = {
        "method": "pruning",
        "amount": amount,
        "sparsity_report": get_sparsity_report(model),
        "params_before": params_before,
        "params_after": params_after,
        "size_before_mb": size_before,
        "size_after_mb": size_after,
        "latency_before_ms": latency_before["latency_mean_ms"],
        "latency_after_ms": latency_after["latency_mean_ms"],
        "val_accuracy": metrics["accuracy"],
        "val_macro_f1": metrics["macro_f1"],
        "val_kappa": metrics["kappa"],
        "val_per_class_f1": metrics["per_class_f1"],
    }

    log.info(f"Accuracy after pruning: {metrics['accuracy']:.4f} | Macro-F1: {metrics['macro_f1']:.4f}")
    log.info(f"Latency before: {latency_before['latency_mean_ms']:.2f}ms → after: {latency_after['latency_mean_ms']:.2f}ms")
    return report


def run_int8(checkpoint_path: str, cfg_path: str) -> dict:
    log.info("=== INT8 QUANTIZATION ===")
    model, cfg, loader = _load_model_and_data(checkpoint_path, cfg_path)

    backend = cfg.compression.quantization.backend
    latency_before = benchmark_latency(model, n_warmup=10, n_runs=50)

    quantized_model = apply_dynamic_quantization(model, backend=backend)
    reduction = measure_quantization_reduction(model, quantized_model)
    latency_after = benchmark_latency(quantized_model, n_warmup=10, n_runs=50)
    metrics = _quick_eval(quantized_model, loader)

    report = {
        "method": "int8_dynamic_quantization",
        "backend": backend,
        **reduction,
        "latency_before_ms": latency_before["latency_mean_ms"],
        "latency_after_ms": latency_after["latency_mean_ms"],
        "val_accuracy": metrics["accuracy"],
        "val_macro_f1": metrics["macro_f1"],
        "val_kappa": metrics["kappa"],
        "val_per_class_f1": metrics["per_class_f1"],
    }

    log.info(f"Accuracy after INT8: {metrics['accuracy']:.4f} | Size: {reduction['fp32_size_mb']:.2f}MB → {reduction['int8_size_mb']:.2f}MB ({reduction['size_reduction_pct']:.1f}% reduction)")
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description="NeuroStage Model Compression")
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--config", default="configs/base.yaml")
    parser.add_argument("--method", default="all", choices=["pruning", "int8", "all"])
    parser.add_argument("--prune_amount", type=float, default=0.3)
    args = parser.parse_args()

    results = {}
    if args.method in ("pruning", "all"):
        results["pruning"] = run_pruning(args.checkpoint, args.config, args.prune_amount)
    if args.method in ("int8", "all"):
        results["int8"] = run_int8(args.checkpoint, args.config)

    out_path = Path(args.checkpoint).with_suffix(".compression.json")
    with open(out_path, "w") as f:
        json.dump(results, f, indent=2)
    log.info(f"Compression results saved to {out_path}")


if __name__ == "__main__":
    main()
