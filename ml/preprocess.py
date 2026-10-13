"""Module 1 - Data pipeline: band-pass filter, z-score, cut into 30 s epochs."""
import numpy as np
from scipy.signal import butter, sosfiltfilt
from config import FS, EPOCH_LEN


def preprocess(signal, fs=FS):
    sos = butter(4, [0.3, 35.0], btype="band", fs=fs, output="sos")
    x = sosfiltfilt(sos, np.asarray(signal, dtype=np.float64))
    x = (x - x.mean()) / (x.std() + 1e-8)
    return np.clip(x, -8, 8).astype(np.float32)          # clip artifacts


def to_epochs(x):
    n = len(x) // EPOCH_LEN
    return x[: n * EPOCH_LEN].reshape(n, EPOCH_LEN)
