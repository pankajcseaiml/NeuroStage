"""
Training loop for NeuroStage with MLflow experiment tracking.

Key research-rigor requirements implemented here:
  1. Fixed random seed for reproducibility
  2. Class-weighted CrossEntropyLoss for N1 imbalance
  3. MLflow logging of all hyperparameters, per-epoch metrics, and artifacts
  4. Early stopping to prevent overfitting
  5. Learning rate warm-up + cosine decay
"""

from __future__ import annotations

import logging
import random
import sys
import time
from pathlib import Path
from typing import Any

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
if hasattr(sys.stderr, "reconfigure"):
    sys.stderr.reconfigure(encoding="utf-8", errors="replace")

import mlflow
import numpy as np
import torch
import torch.nn as nn
from torch.optim import Adam
from torch.optim.lr_scheduler import CosineAnnealingLR, LinearLR, SequentialLR
from torch.utils.data import DataLoader

from neurostage.evaluation.metrics import compute_metrics

log = logging.getLogger(__name__)


def set_seed(seed: int) -> None:
    """Fix all random seeds for full reproducibility."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


class Trainer:
    """Manages the training loop, MLflow logging, and model checkpointing."""

    def __init__(
        self,
        model: nn.Module,
        cfg: Any,
        train_loader: DataLoader,
        val_loader: DataLoader,
        class_weights: torch.Tensor | None = None,
        run_name: str | None = None,
        checkpoint_dir: str = "runs/checkpoints",
    ):
        self.model = model
        self.cfg = cfg
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.checkpoint_dir = Path(checkpoint_dir)
        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model.to(self.device)

        if class_weights is not None:
            self.criterion = nn.CrossEntropyLoss(weight=class_weights.to(self.device))
        else:
            self.criterion = nn.CrossEntropyLoss()

        self.optimizer = Adam(
            model.parameters(),
            lr=cfg.training.lr,
            weight_decay=cfg.training.weight_decay,
        )

        total_steps = cfg.training.epochs
        warmup_steps = cfg.training.lr_warmup_epochs

        warmup_scheduler = LinearLR(
            self.optimizer, start_factor=0.01, end_factor=1.0, total_iters=warmup_steps
        )
        cosine_scheduler = CosineAnnealingLR(
            self.optimizer, T_max=total_steps - warmup_steps, eta_min=1e-6
        )
        self.scheduler = SequentialLR(
            self.optimizer,
            schedulers=[warmup_scheduler, cosine_scheduler],
            milestones=[warmup_steps],
        )

        self.run_name = run_name or cfg.experiment.name
        self.best_val_f1 = -float("inf")
        self.patience_counter = 0
        self.best_ckpt_path: Path | None = None

    def _epoch_loop(
        self, loader: DataLoader, train: bool = True
    ) -> tuple[float, dict]:
        """Run one epoch of training or validation. Returns (loss, metrics_dict)."""
        self.model.train(train)
        total_loss = 0.0
        all_preds, all_labels = [], []

        with torch.set_grad_enabled(train):
            for batch_x, batch_y in loader:
                batch_x = batch_x.to(self.device)      # (B, S, 1, T)
                batch_y = batch_y.to(self.device)      # (B, S)

                logits, _ = self.model(batch_x)         # (B, S, C)
                B, S, C = logits.shape
                loss = self.criterion(
                    logits.reshape(B * S, C), batch_y.reshape(B * S)
                )

                if train:
                    self.optimizer.zero_grad()
                    loss.backward()
                    nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
                    self.optimizer.step()

                total_loss += loss.item() * B * S
                all_preds.append(logits.reshape(B * S, C).argmax(dim=-1).cpu())
                all_labels.append(batch_y.reshape(B * S).cpu())

        n_samples = max(1, len(loader.dataset) * getattr(loader.dataset, "context_epochs", 1))
        avg_loss = total_loss / n_samples
        if len(all_preds) == 0:
            return 0.0, {
                "accuracy": 0.0,
                "macro_f1": 0.0,
                "kappa": 0.0,
                "per_class_f1": [0.0] * 5,
                "confusion_matrix": [[0]*5 for _ in range(5)],
                "class_names": ["Wake", "N1", "N2", "N3", "REM"],
            }
        preds = torch.cat(all_preds).numpy()
        labels = torch.cat(all_labels).numpy()
        metrics = compute_metrics(labels, preds)
        return avg_loss, metrics

    def train(self) -> dict:
        """Full training run. Returns best validation metrics dict."""
        mlflow.set_tracking_uri(self.cfg.experiment.mlflow_tracking_uri)
        mlflow.set_experiment(self.cfg.experiment.name)

        with mlflow.start_run(run_name=self.run_name) as run:
            # Log all config as hyperparameters
            mlflow.log_params(self._flatten_config(self.cfg))

            best_metrics = {}
            for epoch in range(1, self.cfg.training.epochs + 1):
                t0 = time.time()
                train_loss, train_m = self._epoch_loop(self.train_loader, train=True)
                val_loss, val_m = self._epoch_loop(self.val_loader, train=False)
                self.scheduler.step()

                elapsed = time.time() - t0
                log.info(
                    f"Epoch {epoch:3d} | "
                    f"train_loss={train_loss:.4f}  val_loss={val_loss:.4f}  "
                    f"val_acc={val_m['accuracy']:.4f}  val_macro_f1={val_m['macro_f1']:.4f}  "
                    f"val_kappa={val_m['kappa']:.4f}  [{elapsed:.1f}s]"
                )

                # MLflow logging
                mlflow.log_metrics({
                    "train_loss": train_loss,
                    "val_loss": val_loss,
                    "val_accuracy": val_m["accuracy"],
                    "val_macro_f1": val_m["macro_f1"],
                    "val_kappa": val_m["kappa"],
                    "val_f1_wake": val_m["per_class_f1"][0],
                    "val_f1_n1": val_m["per_class_f1"][1],
                    "val_f1_n2": val_m["per_class_f1"][2],
                    "val_f1_n3": val_m["per_class_f1"][3],
                    "val_f1_rem": val_m["per_class_f1"][4],
                    "lr": self.scheduler.get_last_lr()[0],
                }, step=epoch)

                # Checkpoint best model by macro F1
                val_f1 = val_m["macro_f1"]
                if val_f1 > self.best_val_f1:
                    self.best_val_f1 = val_f1
                    self.patience_counter = 0
                    best_metrics = val_m
                    ckpt_path = self.checkpoint_dir / f"{self.run_name}_best.pt"
                    torch.save({
                        "epoch": epoch,
                        "model_state_dict": self.model.state_dict(),
                        "optimizer_state_dict": self.optimizer.state_dict(),
                        "val_metrics": val_m,
                        "mlflow_run_id": run.info.run_id,
                    }, ckpt_path)
                    self.best_ckpt_path = ckpt_path
                    mlflow.log_artifact(str(ckpt_path), artifact_path="checkpoints")
                else:
                    self.patience_counter += 1

                # Early stopping
                if self.patience_counter >= self.cfg.training.patience:
                    log.info(f"Early stopping at epoch {epoch}.")
                    break

            # Log final best metrics as summary
            if best_metrics:
                mlflow.log_metrics({f"best_{k}": v for k, v in best_metrics.items()
                                    if isinstance(v, float)})

            log.info(f"Training complete. Best val macro-F1: {self.best_val_f1:.4f}")
            log.info(f"MLflow run ID: {run.info.run_id}")
            log.info(f"Best checkpoint: {self.best_ckpt_path}")

        return best_metrics

    @staticmethod
    def _flatten_config(cfg: Any, prefix: str = "") -> dict:
        """Flatten nested config dict to key=value pairs for MLflow param logging."""
        flat = {}
        for key, val in cfg.items():
            full_key = f"{prefix}.{key}" if prefix else key
            if isinstance(val, dict):
                flat.update(Trainer._flatten_config(val, full_key))
            else:
                flat[full_key] = str(val)
        return flat
