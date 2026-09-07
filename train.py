#!/usr/bin/env python
"""
train.py — NeuroStage training CLI.

Usage:
    # Single run with base config:
    python train.py

    # With config override:
    python train.py configs/ablation_cnn_only.yaml

    # With dotted CLI overrides:
    python train.py training.lr=5e-4 model.transformer.n_layers=4

    # Full ablation suite (runs all configs sequentially):
    python train.py --ablation

All runs are logged to MLflow (default: http://localhost:5000).
Start MLflow UI: mlflow ui --port 5000
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path

# Fix Windows cp1252 encoding crash when MLflow prints emojis
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
if hasattr(sys.stderr, "reconfigure"):
    sys.stderr.reconfigure(encoding="utf-8", errors="replace")

from torch.utils.data import DataLoader

from neurostage.config import load_config
from neurostage.data.dataset import SleepDataset
from neurostage.data.splits import subject_kfold_splits
from neurostage.evaluation.metrics import aggregate_fold_metrics
from neurostage.models.neurostage import build_model
from neurostage.training.trainer import Trainer, set_seed

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
)
log = logging.getLogger("train")

ABLATION_CONFIGS = [
    "configs/base.yaml",
    "configs/ablation_cnn_only.yaml",
    "configs/ablation_transformer_1l.yaml",
    "configs/ablation_transformer_4l.yaml",
]


def run_single(cfg_path: str = "configs/base.yaml", extra_overrides: list[str] | None = None) -> dict:
    """Run one full subject-independent k-fold training experiment."""
    overrides = extra_overrides or []
    cfg = load_config(cfg_path, *overrides)
    set_seed(cfg.experiment.seed)

    manifest_path = cfg.data.manifest_path
    if not Path(manifest_path).exists():
        log.error(
            f"Dataset manifest not found at {manifest_path}. "
            "Run `python scripts/prepare_data.py` first."
        )
        sys.exit(1)

    splits = subject_kfold_splits(manifest_path, cfg.experiment.n_folds, cfg.experiment.seed)
    fold_metrics = []

    for split in splits:
        fold = split["fold"]
        log.info(f"\n{'='*60}")
        log.info(f"FOLD {fold + 1}/{cfg.experiment.n_folds}")
        log.info(f"  Train subjects: {split['train_subjects']}")
        log.info(f"  Test  subjects: {split['test_subjects']}")

        train_ds = SleepDataset(
            manifest_path,
            split["train_subjects"],
            context_epochs=cfg.data.context_epochs,
            augment=True,
        )
        val_ds = SleepDataset(
            manifest_path,
            split["test_subjects"],
            context_epochs=cfg.data.context_epochs,
            stride=cfg.data.context_epochs,  # Non-overlapping for evaluation
            augment=False,
        )

        if len(train_ds) == 0 or len(val_ds) == 0:
            log.warning(f"Fold {fold} has empty split — skipping.")
            continue

        class_weights = (
            SleepDataset.compute_class_weights(manifest_path, split["train_subjects"])
            if cfg.training.use_class_weights
            else None
        )

        train_loader = DataLoader(train_ds, batch_size=cfg.training.batch_size, shuffle=True, num_workers=0)
        val_loader = DataLoader(val_ds, batch_size=cfg.training.batch_size, shuffle=False, num_workers=0)

        model = build_model(cfg)
        param_info = model.count_parameters()
        log.info(f"  Model parameters: {param_info['total']:,}")

        trainer = Trainer(
            model=model,
            cfg=cfg,
            train_loader=train_loader,
            val_loader=val_loader,
            class_weights=class_weights,
            run_name=f"{cfg.experiment.name}_fold{fold}",
        )
        best_metrics = trainer.train()
        fold_metrics.append(best_metrics)

    if not fold_metrics:
        log.error("No folds completed successfully.")
        return {}

    agg = aggregate_fold_metrics(fold_metrics)
    log.info("\n" + "=" * 60)
    log.info("CROSS-VALIDATION RESULTS (mean ± std)")
    log.info(f"  Accuracy  : {agg['accuracy_mean']:.4f} ± {agg['accuracy_std']:.4f}")
    log.info(f"  Macro-F1  : {agg['macro_f1_mean']:.4f} ± {agg['macro_f1_std']:.4f}")
    log.info(f"  Kappa     : {agg['kappa_mean']:.4f} ± {agg['kappa_std']:.4f}")
    log.info(f"  F1 (N1)   : {agg['f1_n1_mean']:.4f} ± {agg['f1_n1_std']:.4f}  ← hardest class")

    results_dir = Path(cfg.evaluation.output_dir)
    results_dir.mkdir(parents=True, exist_ok=True)
    out_path = results_dir / f"{cfg.experiment.name}_cv_results.json"
    with open(out_path, "w") as f:
        json.dump({"config": cfg_path, "aggregated": agg, "per_fold": fold_metrics}, f, indent=2)
    log.info(f"Results saved to {out_path}")

    return agg


def run_ablation() -> None:
    """Run all ablation configs in sequence."""
    log.info("Running full ablation matrix...")
    for cfg_path in ABLATION_CONFIGS:
        log.info(f"\n{'#'*60}")
        log.info(f"# ABLATION: {cfg_path}")
        run_single(cfg_path)


def main() -> None:
    parser = argparse.ArgumentParser(description="NeuroStage Training CLI")
    parser.add_argument("overrides", nargs="*", help="Config file or key=value overrides")
    parser.add_argument("--ablation", action="store_true", help="Run full ablation suite")
    args = parser.parse_args()

    if args.ablation:
        run_ablation()
        return

    # Separate YAML files from key=value overrides
    cfg_path = "configs/base.yaml"
    extra = []
    for o in args.overrides:
        if o.endswith(".yaml") and Path(o).exists():
            cfg_path = o
        else:
            extra.append(o)

    run_single(cfg_path, extra)


if __name__ == "__main__":
    main()
