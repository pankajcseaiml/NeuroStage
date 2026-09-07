"""
Subject-independent k-fold splits.

CRITICAL: No subject's epochs may appear in both train and test sets.
Epoch-level shuffling is explicitly avoided here.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np


def subject_kfold_splits(
    manifest_path: str | Path, n_folds: int = 5, seed: int = 42
) -> list[dict]:
    """
    Generate subject-independent k-fold splits.

    Parameters
    ----------
    manifest_path : path to the dataset manifest JSON.
    n_folds : number of folds.
    seed : for reproducible shuffling of subject order.

    Returns
    -------
    List of dicts, each with 'train' and 'test' keys containing subject_ids.
    """
    with open(manifest_path) as f:
        manifest = json.load(f)

    subjects = [s["subject_id"] for s in manifest["subjects"]]
    if len(subjects) == 0:
        return []
    n_folds = min(n_folds, len(subjects))
    rng = np.random.default_rng(seed)
    shuffled = rng.permutation(subjects).tolist()

    fold_size = max(1, len(shuffled) // n_folds)
    splits = []
    for fold_idx in range(n_folds):
        start = fold_idx * fold_size
        end = start + fold_size if fold_idx < n_folds - 1 else len(shuffled)
        test_subjects = shuffled[start:end]
        train_subjects = [s for s in shuffled if s not in test_subjects]
        splits.append(
            {
                "fold": fold_idx,
                "train_subjects": train_subjects,
                "test_subjects": test_subjects,
            }
        )

    return splits
