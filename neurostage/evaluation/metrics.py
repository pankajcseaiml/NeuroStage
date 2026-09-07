"""Evaluation metrics: accuracy, macro-F1, per-class F1, Cohen's Kappa."""

from __future__ import annotations

import numpy as np
from sklearn.metrics import (
    accuracy_score,
    cohen_kappa_score,
    confusion_matrix,
    f1_score,
)

STAGE_NAMES = ["Wake", "N1", "N2", "N3", "REM"]


def compute_metrics(
    y_true: np.ndarray, y_pred: np.ndarray
) -> dict:
    """
    Compute the full NeuroStage evaluation metric suite.

    Parameters
    ----------
    y_true : Ground-truth integer labels (0-4)
    y_pred : Predicted integer labels (0-4)

    Returns
    -------
    dict with keys:
      accuracy, macro_f1, kappa, per_class_f1 (list of 5 floats),
      confusion_matrix (5×5 ndarray), class_names
    """
    accuracy = float(accuracy_score(y_true, y_pred))
    macro_f1 = float(f1_score(y_true, y_pred, average="macro", zero_division=0, labels=list(range(5))))
    try:
        raw_kappa = cohen_kappa_score(y_true, y_pred, labels=list(range(5)))
        kappa = 0.0 if np.isnan(raw_kappa) else float(raw_kappa)
    except Exception:
        kappa = 0.0
    per_class_f1 = f1_score(y_true, y_pred, average=None, zero_division=0, labels=list(range(5)))
    cm = confusion_matrix(y_true, y_pred, labels=list(range(5)))

    return {
        "accuracy": accuracy,
        "macro_f1": macro_f1,
        "kappa": kappa,
        "per_class_f1": per_class_f1.tolist(),
        "confusion_matrix": cm.tolist(),
        "class_names": STAGE_NAMES,
    }


def aggregate_fold_metrics(fold_metrics: list[dict]) -> dict:
    """
    Aggregate per-fold metrics across a k-fold run.

    Returns mean ± std for each scalar metric (accuracy, macro_f1, kappa)
    and per-class F1.
    """
    scalar_keys = ["accuracy", "macro_f1", "kappa"]
    aggregated = {}

    for key in scalar_keys:
        values = [m[key] for m in fold_metrics]
        aggregated[f"{key}_mean"] = float(np.mean(values))
        aggregated[f"{key}_std"] = float(np.std(values))

    # Per-class F1 mean ± std
    per_class = np.array([m["per_class_f1"] for m in fold_metrics])
    for i, name in enumerate(STAGE_NAMES):
        aggregated[f"f1_{name.lower()}_mean"] = float(per_class[:, i].mean())
        aggregated[f"f1_{name.lower()}_std"] = float(per_class[:, i].std())

    aggregated["n_folds"] = len(fold_metrics)
    return aggregated
