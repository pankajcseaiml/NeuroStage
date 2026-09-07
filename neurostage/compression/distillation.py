"""
Knowledge Distillation training loop.

A larger "teacher" model guides a smaller "student" model.

Loss function:
    L = α × KL(student_soft || teacher_soft) + (1 - α) × CE(student_logits, true_labels)

where soft targets are obtained by dividing logits by temperature T before softmax.
Higher temperature produces a softer distribution, giving the student more information
about the teacher's "dark knowledge" (inter-class similarity structure).

Typical values: T = 4.0, α = 0.7 (per Hinton et al., 2015)
"""

from __future__ import annotations

import logging
from typing import Any

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader

from neurostage.evaluation.metrics import compute_metrics

log = logging.getLogger(__name__)


def distillation_loss(
    student_logits: torch.Tensor,
    teacher_logits: torch.Tensor,
    true_labels: torch.Tensor,
    temperature: float = 4.0,
    alpha: float = 0.7,
) -> torch.Tensor:
    """
    Combined knowledge distillation + cross-entropy loss.

    student_logits : (N, n_classes) — from student model
    teacher_logits : (N, n_classes) — from teacher model (no grad)
    true_labels    : (N,) — hard labels
    """
    # Soft targets
    soft_student = F.log_softmax(student_logits / temperature, dim=-1)
    soft_teacher = F.softmax(teacher_logits / temperature, dim=-1)

    kd_loss = F.kl_div(soft_student, soft_teacher, reduction="batchmean") * (temperature ** 2)
    ce_loss = F.cross_entropy(student_logits, true_labels)

    return alpha * kd_loss + (1.0 - alpha) * ce_loss


def train_with_distillation(
    student: nn.Module,
    teacher: nn.Module,
    train_loader: DataLoader,
    val_loader: DataLoader,
    cfg: Any,
    device: str = "cpu",
) -> dict:
    """
    Train student model with teacher guidance.

    Parameters
    ----------
    student  : smaller NeuroStage model (to train)
    teacher  : pre-trained larger model (frozen)
    cfg      : config object

    Returns best validation metrics dict.
    """
    kd_cfg = cfg.compression.distillation
    teacher.eval()
    for p in teacher.parameters():
        p.requires_grad_(False)

    student.to(device)
    teacher.to(device)

    optimizer = torch.optim.Adam(student.parameters(), lr=1e-3, weight_decay=1e-4)

    best_val_f1 = -float("inf")
    best_metrics = {}

    for epoch in range(1, kd_cfg.n_epochs + 1):
        student.train()
        for batch_x, batch_y in train_loader:
            batch_x = batch_x.to(device)
            batch_y = batch_y.to(device)
            B, S, C, T = batch_x.shape

            student_logits, _ = student(batch_x)
            with torch.no_grad():
                teacher_logits, _ = teacher(batch_x)

            s_flat = student_logits.reshape(B * S, -1)
            t_flat = teacher_logits.reshape(B * S, -1)
            y_flat = batch_y.reshape(B * S)

            loss = distillation_loss(
                s_flat, t_flat, y_flat,
                temperature=kd_cfg.temperature,
                alpha=kd_cfg.alpha,
            )
            optimizer.zero_grad()
            loss.backward()
            nn.utils.clip_grad_norm_(student.parameters(), 1.0)
            optimizer.step()

        # Validate
        student.eval()
        all_preds, all_labels = [], []
        with torch.no_grad():
            for batch_x, batch_y in val_loader:
                batch_x = batch_x.to(device)
                logits, _ = student(batch_x)
                B, S, C_out = logits.shape
                preds = logits.reshape(B * S, C_out).argmax(dim=-1).cpu()
                labels = batch_y.reshape(B * S).cpu()
                all_preds.append(preds)
                all_labels.append(labels)

        preds_arr = torch.cat(all_preds).numpy()
        labels_arr = torch.cat(all_labels).numpy()
        val_m = compute_metrics(labels_arr, preds_arr)

        log.info(
            f"KD Epoch {epoch:3d} | val_acc={val_m['accuracy']:.4f} "
            f"val_macro_f1={val_m['macro_f1']:.4f}"
        )

        if val_m["macro_f1"] > best_val_f1:
            best_val_f1 = val_m["macro_f1"]
            best_metrics = val_m

    log.info(f"Distillation complete. Best val macro-F1: {best_val_f1:.4f}")
    return best_metrics
