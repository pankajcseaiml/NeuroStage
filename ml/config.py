"""Shared constants for NeuroStage. Change them here, everything else follows."""
from pathlib import Path

FS = 100                    # sampling rate (Hz) used by Sleep-EDF Fpz-Cz
EPOCH_SEC = 30              # AASM epoch length
EPOCH_LEN = FS * EPOCH_SEC  # 3000 samples per epoch
SEQ_LEN = 10                # how many consecutive epochs the Transformer sees
STAGES = ["Wake", "N1", "N2", "N3", "REM"]
N_CLASSES = len(STAGES)

ROOT = Path(__file__).parent
ARTIFACTS = ROOT / "artifacts"
RECORDINGS = ROOT / "recordings"
ARTIFACTS.mkdir(exist_ok=True)
RECORDINGS.mkdir(exist_ok=True)
