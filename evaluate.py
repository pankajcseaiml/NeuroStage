#!/usr/bin/env python
"""
evaluate.py — Evaluate a trained NeuroStage checkpoint.

Usage:
    python evaluate.py --checkpoint runs/checkpoints/neurostage_base_best.pt
    python evaluate.py --checkpoint runs/checkpoints/neurostage_base_best.pt --benchmark
"""

from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path

import torch
from torch.utils.data import DataLoader

from neurostage.config import load_config
from neurostage.data.dataset import SleepDataset
from neurostage.data.splits import subject_kfold_splits
from neurostage.evaluation.benchmark import full_efficiency_report
from neurostage.evaluation.metrics import compute_metrics
from neurostage.models.neurostage import build_model

logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)s | %(message)s")
log = logging.getLogger("evaluate")


def evaluate_checkpoint(
    checkpoint_path: str,
    cfg_path: str = "configs/base.yaml",
    run_benchmark: bool = False,
) -> dict:
    checkpoint = torch.load(checkpoint_path, map_location="cpu")
    cfg = load_config(cfg_path)
    model = build_model(cfg)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()

    log.info(f"Loaded checkpoint from epoch {checkpoint.get('epoch', '?')}")
    log.info(f"Checkpoint val metrics: {checkpoint.get('val_metrics', {})}")

    # Load test split (use last fold as default evaluation split)
    splits = subject_kfold_splits(cfg.data.manifest_path, cfg.experiment.n_folds)
    test_subjects = splits[-1]["test_subjects"]
    log.info(f"Evaluating on {len(test_subjects)} subjects: {test_subjects}")

    test_ds = SleepDataset(
        cfg.data.manifest_path,
        test_subjects,
        context_epochs=cfg.data.context_epochs,
        stride=cfg.data.context_epochs,
        augment=False,
    )
    loader = DataLoader(test_ds, batch_size=cfg.training.batch_size, shuffle=False)

    all_preds, all_labels = [], []
    with torch.no_grad():
        for batch_x, batch_y in loader:
            logits, _ = model(batch_x)
            B, S, C = logits.shape
            all_preds.append(logits.reshape(B * S, C).argmax(dim=-1).cpu())
            all_labels.append(batch_y.reshape(B * S).cpu())

    preds = torch.cat(all_preds).numpy()
    labels = torch.cat(all_labels).numpy()
    metrics = compute_metrics(labels, preds)

    log.info("\n" + "=" * 50)
    log.info("EVALUATION RESULTS")
    log.info(f"  Accuracy : {metrics['accuracy']:.4f}")
    log.info(f"  Macro-F1 : {metrics['macro_f1']:.4f}")
    log.info(f"  Kappa    : {metrics['kappa']:.4f}")
    for i, name in enumerate(metrics["class_names"]):
        log.info(f"  F1 ({name:4s}) : {metrics['per_class_f1'][i]:.4f}")

    result = {"classification": metrics}

    if run_benchmark:
        log.info("\nRunning efficiency benchmark...")
        eff = full_efficiency_report(model, cfg)
        log.info(f"  Parameters     : {eff['total_params']:,}")
        log.info(f"  Model size (FP32): {eff['model_size_mb_fp32']:.2f} MB")
        log.info(f"  Latency (mean) : {eff['latency_mean_ms']:.2f} ms")
        log.info(f"  Latency (p95)  : {eff['latency_p95_ms']:.2f} ms")
        result["efficiency"] = eff

    # Save result
    out_path = Path(checkpoint_path).with_suffix(".eval.json")
    with open(out_path, "w") as f:
        json.dump(result, f, indent=2)
    log.info(f"\nResults saved to {out_path}")
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate NeuroStage checkpoint")
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--config", default="configs/base.yaml")
    parser.add_argument("--benchmark", action="store_true")
    args = parser.parse_args()
    evaluate_checkpoint(args.checkpoint, args.config, args.benchmark)


if __name__ == "__main__":
    main()
