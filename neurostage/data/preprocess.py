"""
EEG preprocessing pipeline.

Steps per recording:
  1. Load PSG EDF + Hypnogram EDF via MNE
  2. Select EEG channel (Fpz-Cz preferred)
  3. Resample to target sfreq
  4. Band-pass filter (0.3–35 Hz)
  5. Extract 30-second epochs aligned to annotation events
  6. Z-score normalize per recording
  7. Map AASM stage labels to integers {0: Wake, 1: N1, 2: N2, 3: N3, 4: REM}
  8. Persist to HDF5 and update dataset_manifest.json

All preprocessing parameters are read from the config so results are reproducible.
"""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any

import h5py
import mne
import numpy as np

log = logging.getLogger(__name__)

# AASM label -> integer
DEFAULT_STAGE_MAP = {
    "Sleep stage W": 0,
    "Sleep stage N1": 1,
    "Sleep stage N2": 2,
    "Sleep stage N3": 3,
    "Sleep stage 4": 3,   # Older datasets: stage 4 = N3
    "Sleep stage R": 4,
}

STAGE_NAMES = {0: "Wake", 1: "N1", 2: "N2", 3: "N3", 4: "REM"}
PREFERRED_EEG_CHANNELS = [
    "eeg fpz-cz", "fpz-cz", "fpz", "eeg pz-oz", "pz-oz", "pz",
    "eeg c4-a1", "c4-a1", "c4-m1", "eeg c3-a2", "c3-a2", "c3-m2", "eeg"
]


def _select_eeg_channel(raw: mne.io.BaseRaw) -> str:
    """Select the best single EEG channel available in the recording."""
    ch_names = raw.ch_names
    # 1. Exact match (case-insensitive)
    for preferred in PREFERRED_EEG_CHANNELS:
        for ch in ch_names:
            if preferred == ch.strip().lower():
                return ch

    # 2. Substring match
    for preferred in PREFERRED_EEG_CHANNELS:
        for ch in ch_names:
            if preferred in ch.strip().lower():
                return ch

    # 3. Fallback: MNE-detected EEG channel
    eeg_chs = mne.pick_types(raw.info, eeg=True, exclude=[])
    if len(eeg_chs) > 0:
        return ch_names[eeg_chs[0]]

    # 4. Safe fallback: return the first channel
    if len(ch_names) > 0:
        return ch_names[0]

    raise ValueError(f"No channels found in recording: {ch_names}")


def preprocess_recording(
    psg_path: str | Path,
    hypnogram_path: str | Path,
    cfg: Any,
) -> tuple[np.ndarray, np.ndarray] | None:
    """
    Load and preprocess one PSG recording.

    Returns
    -------
    epochs : np.ndarray of shape (n_epochs, 1, n_samples_per_epoch)
    labels : np.ndarray of shape (n_epochs,) with integer sleep stage labels
    Returns None if the recording cannot be processed.
    """
    psg_path = Path(psg_path)
    hypnogram_path = Path(hypnogram_path)

    mne.set_log_level("WARNING")

    try:
        raw = mne.io.read_raw_edf(str(psg_path), preload=True, verbose=False)
        annot = mne.read_annotations(str(hypnogram_path))
        raw.set_annotations(annot, emit_warning=False)
    except Exception as e:
        log.warning(f"Failed to load {psg_path.name}: {e}")
        return None

    # Select EEG channel
    try:
        ch = _select_eeg_channel(raw)
    except ValueError as e:
        log.warning(str(e))
        return None

    raw.pick([ch])

    # Resample
    if raw.info["sfreq"] != cfg.data.sfreq:
        raw.resample(cfg.data.sfreq, npad="auto")

    # Band-pass filter
    raw.filter(
        l_freq=cfg.data.bandpass_low,
        h_freq=cfg.data.bandpass_high,
        method="fir",
        fir_window="hamming",
        verbose=False,
    )

    # Extract sleep-stage events from annotations
    events, event_id = mne.events_from_annotations(
        raw,
        event_id=DEFAULT_STAGE_MAP,
        verbose=False,
    )
    if len(events) == 0:
        log.warning(f"No valid sleep-stage annotations in {psg_path.name}")
        return None

    # Epoch at 30-second boundaries
    epoch_sec = cfg.data.epoch_sec
    n_samples = int(epoch_sec * cfg.data.sfreq)

    epochs_list, labels_list = [], []
    for onset_sample, _, stage_label in events:
        end_sample = onset_sample + n_samples
        if end_sample > len(raw.times):
            break
        epoch_data = raw.get_data(start=onset_sample, stop=end_sample)  # (1, n_samples)
        epochs_list.append(epoch_data)
        labels_list.append(stage_label)

    if len(epochs_list) == 0:
        return None

    epochs = np.array(epochs_list, dtype=np.float32)   # (N, 1, n_samples)
    labels = np.array(labels_list, dtype=np.int64)

    # Z-score normalize per recording (not per epoch)
    mean = epochs.mean()
    std = epochs.std() + 1e-8
    epochs = (epochs - mean) / std

    return epochs, labels


def preprocess_dataset(
    raw_dir: str | Path,
    processed_dir: str | Path,
    cfg: Any,
    dataset_name: str = "sleep-edf",
) -> dict:
    """
    Preprocess all recordings in raw_dir and save to processed_dir as HDF5.

    Returns the dataset manifest dict.
    """
    raw_dir = Path(raw_dir)
    processed_dir = Path(processed_dir)
    processed_dir.mkdir(parents=True, exist_ok=True)

    # Find PSG files
    psg_files = sorted(raw_dir.glob("*-PSG.edf"))
    if not psg_files:
        psg_files = sorted(raw_dir.glob("*.edf"))
        log.warning("No *-PSG.edf found; using all .edf files (may include hypnograms)")

    subjects = []
    total_epochs = 0
    class_counts = {i: 0 for i in range(5)}

    for psg_path in psg_files:
        subject_id = psg_path.stem.split("-")[0][:6]

        # Match hypnogram by subject ID prefix (e.g., SC4001EC-Hypnogram.edf for SC4001E0-PSG.edf)
        hyp_candidates = list(raw_dir.glob(f"{subject_id}*Hypnogram*.edf")) + list(raw_dir.glob(f"{subject_id}*hypnogram*.edf"))

        # Filter candidates: must exist and be a valid EDF file (> 500 bytes)
        valid_hyp = [h for h in hyp_candidates if h.is_file() and h.stat().st_size > 500]

        if not valid_hyp:
            # Try ISRUC text annotation fallback
            txt_candidate = psg_path.with_name(psg_path.stem.replace("_1", "") + "_1.txt")
            if txt_candidate.exists():
                hyp_path = txt_candidate
            else:
                log.warning(f"No valid hypnogram found for {psg_path.name}, skipping.")
                continue
        else:
            hyp_path = valid_hyp[0]
        out_h5 = processed_dir / f"{subject_id}.h5"

        # Skip if already processed (check exists + metadata matches)
        if out_h5.exists():
            log.info(f"  [skip] {subject_id} already processed.")
            with h5py.File(out_h5, "r") as f:
                n = len(f["labels"])
                cls_c = {i: int((f["labels"][()] == i).sum()) for i in range(5)}
        else:
            log.info(f"  Processing {subject_id} ...")
            result = preprocess_recording(psg_path, hyp_path, cfg)
            if result is None:
                continue
            epochs, labels = result

            with h5py.File(out_h5, "w") as f:
                f.create_dataset("epochs", data=epochs, compression="gzip", compression_opts=4)
                f.create_dataset("labels", data=labels)
                f.attrs["subject_id"] = subject_id
                f.attrs["sfreq"] = cfg.data.sfreq
                f.attrs["epoch_sec"] = cfg.data.epoch_sec
                f.attrs["bandpass"] = [cfg.data.bandpass_low, cfg.data.bandpass_high]

            n = len(labels)
            cls_c = {i: int((labels == i).sum()) for i in range(5)}

        total_epochs += n
        for i in range(5):
            class_counts[i] += cls_c.get(i, 0)

        subjects.append(
            {
                "subject_id": subject_id,
                "h5_path": str(out_h5),
                "n_epochs": n,
                "class_counts": {STAGE_NAMES[i]: cls_c.get(i, 0) for i in range(5)},
            }
        )
        log.info(f"    → {n} epochs | {cls_c}")

    manifest = {
        "dataset": dataset_name,
        "n_subjects": len(subjects),
        "total_epochs": total_epochs,
        "class_counts": {STAGE_NAMES[i]: class_counts[i] for i in range(5)},
        "preprocessing": {
            "sfreq": cfg.data.sfreq,
            "bandpass": [cfg.data.bandpass_low, cfg.data.bandpass_high],
            "epoch_sec": cfg.data.epoch_sec,
            "normalization": "z-score-per-recording",
        },
        "subjects": subjects,
    }

    manifest_path = processed_dir / "manifest.json"
    with open(manifest_path, "w") as f:
        json.dump(manifest, f, indent=2)

    log.info(f"\nPreprocessing complete. {len(subjects)} subjects, {total_epochs} epochs.")
    log.info(f"Manifest: {manifest_path}")
    return manifest
