"""
PyTorch Dataset for NeuroStage.

Loads pre-processed HDF5 epoch files and serves sequences of epochs (context windows)
to the model. Each item is a sequence of `context_epochs` consecutive epochs + labels.
"""

from __future__ import annotations

import json
from pathlib import Path

import h5py
import numpy as np
import torch
from torch.utils.data import Dataset

STAGE_NAMES = {0: "Wake", 1: "N1", 2: "N2", 3: "N3", 4: "REM"}


class SleepDataset(Dataset):
    """
    Returns (sequence, label_sequence) where:
      sequence       : FloatTensor of shape (context_epochs, 1, n_samples)
      label_sequence : LongTensor of shape (context_epochs,)

    Sequences are built by sliding a context window over each subject's night.
    """

    def __init__(
        self,
        manifest_path: str | Path,
        subject_ids: list[str],
        context_epochs: int = 20,
        stride: int = 1,
        augment: bool = False,
    ):
        self.context_epochs = context_epochs
        self.stride = stride
        self.augment = augment

        with open(manifest_path) as f:
            manifest = json.load(f)

        # Build index of (h5_path, start_epoch_index) for every window
        self._index: list[tuple[str, int]] = []
        self._h5_cache: dict[str, tuple[np.ndarray, np.ndarray]] = {}

        subject_meta = {s["subject_id"]: s for s in manifest["subjects"]}

        for sid in subject_ids:
            if sid not in subject_meta:
                continue
            h5_path = subject_meta[sid]["h5_path"]
            n_epochs = subject_meta[sid]["n_epochs"]
            if n_epochs < context_epochs:
                self._index.append((h5_path, 0))
            else:
                for start in range(0, n_epochs - context_epochs + 1, stride):
                    self._index.append((h5_path, start))

    def _load_subject(self, h5_path: str) -> tuple[np.ndarray, np.ndarray]:
        if h5_path not in self._h5_cache:
            with h5py.File(h5_path, "r") as f:
                epochs = f["epochs"][:]  # (N, 1, T)
                labels = f["labels"][:]  # (N,)
            self._h5_cache[h5_path] = (epochs, labels)
        return self._h5_cache[h5_path]

    def __len__(self) -> int:
        return len(self._index)

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, torch.Tensor]:
        h5_path, start = self._index[idx]
        epochs, labels = self._load_subject(h5_path)
        end = min(start + self.context_epochs, len(epochs))
        seq = epochs[start:end].astype(np.float32)   # (S, 1, T)
        lbl = labels[start:end].astype(np.int64)      # (S,)

        # Pad if shorter than context_epochs
        if len(seq) < self.context_epochs:
            pad_len = self.context_epochs - len(seq)
            pad_seq = np.zeros((pad_len, *seq.shape[1:]), dtype=np.float32)
            pad_lbl = np.zeros(pad_len, dtype=np.int64)
            seq = np.concatenate([seq, pad_seq], axis=0)
            lbl = np.concatenate([lbl, pad_lbl], axis=0)

        if self.augment:
            # Simple augmentations: Gaussian noise, random amplitude scaling
            seq = self._augment(seq)

        return torch.from_numpy(seq), torch.from_numpy(lbl)

    @staticmethod
    def _augment(seq: np.ndarray) -> np.ndarray:
        # Gaussian noise (~1% of signal std)
        noise = np.random.randn(*seq.shape).astype(np.float32) * 0.01
        seq = seq + noise
        # Random amplitude scaling [0.9, 1.1]
        scale = np.random.uniform(0.9, 1.1)
        seq = seq * scale
        return seq

    @staticmethod
    def compute_class_weights(
        manifest_path: str | Path, subject_ids: list[str]
    ) -> torch.Tensor:
        """Compute inverse-frequency class weights for subjects in the split."""
        with open(manifest_path) as f:
            manifest = json.load(f)
        subject_meta = {s["subject_id"]: s for s in manifest["subjects"]}
        counts = np.zeros(5, dtype=np.float64)
        for sid in subject_ids:
            if sid in subject_meta:
                for stage_name, cnt in subject_meta[sid]["class_counts"].items():
                    for k, v in STAGE_NAMES.items():
                        if v == stage_name:
                            counts[k] += cnt
        # Inverse frequency, normalized
        weights = 1.0 / (counts + 1e-6)
        weights = weights / weights.sum() * len(counts)
        return torch.tensor(weights, dtype=torch.float32)
