"""
Loader for the real Sleep-EDF Expanded dataset (single channel EEG Fpz-Cz).
Needs `pip install mne`. First call downloads files from PhysioNet (~50 MB per night).
Returns the same format as synthetic.generate_cohort: list of (signal, labels).
"""
import numpy as np
from config import EPOCH_SEC

ANN_MAP = {"Sleep stage W": 0, "Sleep stage 1": 1, "Sleep stage 2": 2,
           "Sleep stage 3": 3, "Sleep stage 4": 3, "Sleep stage R": 4}  # N3 = stages 3+4 (AASM)


def load_sleepedf(n_subjects=20, trim_wake_min=30):
    import mne
    from mne.datasets import sleep_physionet
    mne.set_log_level("WARNING")
    files = sleep_physionet.age.fetch_data(subjects=list(range(n_subjects)), recording=[1])
    cohort = []
    for psg, hyp in files:
        raw = mne.io.read_raw_edf(psg, preload=True)
        raw.set_annotations(mne.read_annotations(hyp))
        events, _ = mne.events_from_annotations(raw, event_id=ANN_MAP, chunk_duration=float(EPOCH_SEC))
        epochs = mne.Epochs(raw, events, tmin=0.0, tmax=EPOCH_SEC - 1 / raw.info["sfreq"],
                            baseline=None, picks=["EEG Fpz-Cz"], preload=True)
        x = epochs.get_data()[:, 0, :] * 1e6              # volts -> microvolts, shape (n, 3000)
        y = epochs.events[:, 2].astype(np.int64)
        asleep = np.where(y != 0)[0]                      # keep only 30 min of wake before/after sleep
        pad = trim_wake_min * 60 // EPOCH_SEC
        lo, hi = max(0, asleep[0] - pad), min(len(y), asleep[-1] + pad + 1)
        cohort.append((x[lo:hi].reshape(-1).astype(np.float32), y[lo:hi]))
    return cohort
