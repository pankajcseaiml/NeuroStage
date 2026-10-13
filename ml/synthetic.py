"""
Synthetic EEG generator.

Why it exists: it lets the WHOLE project (training, compression, benchmark,
web app) run on a laptop with no dataset download. Each sleep stage is built from
the same EEG ingredients real clinicians look for:
  Wake -> alpha (8-12 Hz) + beta      N2 -> spindles (12-15 Hz bursts) + K-complexes
  N1   -> theta (4-8 Hz)              N3 -> big slow delta waves (0.5-4 Hz)
  REM  -> low-amplitude mixed theta/beta
Per-subject and per-epoch random jitter makes the stages overlap, so N1 stays hard,
just like in real data. Numbers on synthetic data are NOT Sleep-EDF results.
"""
import numpy as np
from config import FS, EPOCH_LEN, EPOCH_SEC

W, N1, N2, N3, REM = range(5)
# weights of: delta, theta, alpha, sigma, beta, pink-noise
MIX = {
    W:   [0.30, 0.30, 1.00, 0.10, 0.60, 0.50],
    N1:  [0.40, 0.70, 0.40, 0.10, 0.25, 0.50],
    N2:  [0.50, 0.50, 0.15, 0.15, 0.15, 0.50],
    N3:  [1.60, 0.30, 0.10, 0.05, 0.10, 0.40],
    REM: [0.35, 0.60, 0.30, 0.05, 0.30, 0.50],
}
BANDS = [(0.5, 4), (4, 8), (8, 12), (12, 15), (15, 30)]


def _band_noise(n, lo, hi, rng):
    f = np.fft.rfftfreq(n, 1 / FS)
    spec = np.fft.rfft(rng.standard_normal(n))
    spec[(f < lo) | (f > hi)] = 0
    x = np.fft.irfft(spec, n)
    return x / (x.std() + 1e-9)


def _pink(n, rng):
    f = np.fft.rfftfreq(n, 1 / FS)
    spec = np.fft.rfft(rng.standard_normal(n))
    spec[1:] /= np.sqrt(np.maximum(f[1:], 0.3))
    x = np.fft.irfft(spec, n)
    return x / (x.std() + 1e-9)


def synth_epoch(stage, rng, subject_gain):
    w = np.array(MIX[stage]) * subject_gain * rng.lognormal(0, 0.2, 6)
    x = sum(w[i] * _band_noise(EPOCH_LEN, *BANDS[i], rng) for i in range(5))
    x = x + w[5] * _pink(EPOCH_LEN, rng)
    t = np.arange(EPOCH_LEN) / FS
    if stage == N2:  # sleep spindles
        for _ in range(rng.integers(1, 4)):
            c, d = rng.uniform(2, EPOCH_SEC - 2), rng.uniform(0.5, 1.5)
            env = np.exp(-0.5 * ((t - c) / (d / 2.5)) ** 2)
            x += rng.uniform(1.5, 2.5) * env * np.sin(2 * np.pi * rng.uniform(12, 15) * t)
        if rng.random() < 0.3:  # K-complex: sharp negative then positive wave
            c = rng.uniform(3, EPOCH_SEC - 3)
            x += 3 * (-np.exp(-0.5 * ((t - c) / 0.25) ** 2) + 0.6 * np.exp(-0.5 * ((t - c - 0.6) / 0.35) ** 2))
    if stage == W and rng.random() < 0.1:  # movement artifact
        a = rng.integers(0, EPOCH_LEN - 300)
        x[a:a + 300] += rng.standard_normal(300) * 4
    return x


def synth_hypnogram(n_epochs, rng):
    seq = [W] * int(rng.integers(6, 20))
    cycle = 0
    while len(seq) < n_epochs:                       # ~90 minute sleep cycles
        seq += [N1] * int(rng.integers(2, 6))
        seq += [N2] * int(rng.integers(10, 30))
        seq += [N3] * max(0, int(rng.normal(40 - cycle * 10, 6)))   # deep sleep fades
        seq += [N2] * int(rng.integers(6, 16))
        seq += [REM] * int(np.clip(rng.normal(10 + cycle * 10, 4), 3, 60))  # REM grows
        if rng.random() < 0.4:
            seq += [W] * int(rng.integers(1, 5))
        cycle += 1
    return np.array(seq[:n_epochs], dtype=np.int64)


def generate_night(hours=2.0, seed=0):
    """Returns (signal float32 [n*3000], labels int64 [n])."""
    rng = np.random.default_rng(seed)
    n = int(hours * 3600 / EPOCH_SEC)
    labels = synth_hypnogram(n, rng)
    gain = rng.lognormal(0, 0.25, 6)               # this "person's" EEG personality
    sig = np.concatenate([synth_epoch(s, rng, gain) for s in labels])
    return sig.astype(np.float32), labels


def generate_cohort(n_subjects=10, hours=2.0, seed=0):
    return [generate_night(hours, seed * 1000 + i) for i in range(n_subjects)]
