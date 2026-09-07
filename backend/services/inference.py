"""
Inference service: loads the model once at application startup and serves predictions.

The model is loaded into module-level state during the FastAPI lifespan event.
This prevents per-request model loading overhead.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any

import numpy as np
import torch

from neurostage.config import load_config
from neurostage.models.neurostage import build_model

log = logging.getLogger(__name__)

_model: Any = None
_cfg: Any = None
_device = "cpu"

STAGE_NAMES = {0: "Wake", 1: "N1", 2: "N2", 3: "N3", 4: "REM"}


def load_inference_model(checkpoint_path: str, cfg_path: str) -> None:
    """Load model into module-level state. Call once at app startup."""
    global _model, _cfg
    _cfg = load_config(cfg_path)
    if not Path(checkpoint_path).exists():
        log.warning(
            f"Model checkpoint not found at {checkpoint_path}. "
            "Inference endpoints will return 503 until a checkpoint is available."
        )
        return
    ckpt = torch.load(checkpoint_path, map_location=_device)
    _model = build_model(_cfg)
    _model.load_state_dict(ckpt["model_state_dict"])
    _model.eval()
    log.info(f"Inference model loaded from {checkpoint_path}")


def is_model_loaded() -> bool:
    return _model is not None


def predict_epochs(epochs: np.ndarray) -> dict:
    """
    Run inference on a sequence of EEG epochs.

    Parameters
    ----------
    epochs : np.ndarray of shape (n_epochs, 1, n_samples) — pre-processed EEG

    Returns
    -------
    dict with per-epoch predictions, confidence, and attention weights.
    """
    if not is_model_loaded():
        raise RuntimeError("Model not loaded.")

    context = _cfg.data.context_epochs
    n_epochs = len(epochs)

    all_predicted = []
    all_confidence = []
    all_attention = []

    with torch.no_grad():
        for start in range(0, n_epochs, context):
            window = epochs[start: start + context]
            pad = context - len(window)
            if pad > 0:
                window = np.concatenate([window, np.zeros((pad, *window.shape[1:]), dtype=np.float32)])

            x = torch.from_numpy(window).unsqueeze(0).float()  # (1, S, 1, T)
            result = _model.predict(x)

            predicted = result["predicted"][0].numpy().tolist()     # (S,)
            confidence = result["confidence"][0].numpy().tolist()   # (S,)
            attn = result["attention_weights"]
            attn_list = attn[0].numpy().tolist() if attn is not None else None

            actual_len = min(context, n_epochs - start)
            all_predicted.extend(predicted[:actual_len])
            all_confidence.extend(confidence[:actual_len])
            if attn_list:
                all_attention.append(attn_list)

    return {
        "n_epochs": n_epochs,
        "predictions": [
            {
                "epoch_index": i,
                "predicted_class": all_predicted[i],
                "stage_name": STAGE_NAMES[all_predicted[i]],
                "confidence": round(all_confidence[i], 4),
            }
            for i in range(n_epochs)
        ],
        "attention_weights": all_attention if all_attention else None,
    }


def preprocess_edf_for_inference(edf_path: str) -> np.ndarray | None:
    """Preprocess an uploaded EDF file for inference using the stored config."""
    if _cfg is None:
        return None
    try:
        import mne

        from neurostage.data.preprocess import (
            _select_eeg_channel,
        )

        mne.set_log_level("WARNING")
        raw = mne.io.read_raw_edf(edf_path, preload=True, verbose=False)
        ch = _select_eeg_channel(raw)
        raw.pick([ch])
        if raw.info["sfreq"] != _cfg.data.sfreq:
            raw.resample(_cfg.data.sfreq, npad="auto")
        raw.filter(
            l_freq=_cfg.data.bandpass_low,
            h_freq=_cfg.data.bandpass_high,
            method="fir",
            verbose=False,
        )

        n_samples = int(_cfg.data.epoch_sec * _cfg.data.sfreq)
        total_samples = raw.n_times
        n_epochs = total_samples // n_samples

        epochs_list = []
        for i in range(n_epochs):
            start = i * n_samples
            stop = start + n_samples
            epoch = raw.get_data(start=start, stop=stop)  # (1, n_samples)
            epochs_list.append(epoch)

        epochs = np.array(epochs_list, dtype=np.float32)  # (N, 1, T)
        mean, std = epochs.mean(), epochs.std() + 1e-8
        epochs = (epochs - mean) / std
        return epochs
    except Exception as e:
        log.error(f"Failed to preprocess EDF: {e}")
        return None
