"""
Download Sleep-EDF (Cassette subset) from PhysioNet.

Supports:
1. Direct MNE Fetcher (Robust, official PhysioNet downloader with automatic retry & resumption)
2. Python Streaming Downloader with requests/urllib (fallback)
3. Synthetic / Fast mode (--mock) for instant pipeline testing without 1.5GB download

Usage:
    python scripts/download_data.py --n_subjects 5
    python scripts/download_data.py --mock --n_subjects 5
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import shutil
import sys
import urllib.request
from pathlib import Path

logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)s | %(message)s")
log = logging.getLogger("download_data")

PHYSIONET_BASE_URL = "https://physionet.org/files/sleep-edfx/1.0.0/sleep-cassette/"

SUBJECT_FILE_PAIRS = [
    ("SC4001E0-PSG.edf", "SC4001EC-Hypnogram.edf"),
    ("SC4002E0-PSG.edf", "SC4002EC-Hypnogram.edf"),
    ("SC4011E0-PSG.edf", "SC4011EC-Hypnogram.edf"),
    ("SC4012E0-PSG.edf", "SC4012EC-Hypnogram.edf"),
    ("SC4021E0-PSG.edf", "SC4021EC-Hypnogram.edf"),
    ("SC4022E0-PSG.edf", "SC4022EC-Hypnogram.edf"),
    ("SC4031E0-PSG.edf", "SC4031EC-Hypnogram.edf"),
    ("SC4032E0-PSG.edf", "SC4032EC-Hypnogram.edf"),
    ("SC4041E0-PSG.edf", "SC4041EC-Hypnogram.edf"),
    ("SC4042E0-PSG.edf", "SC4042EC-Hypnogram.edf"),
]


def download_with_progress(url: str, dest: Path) -> bool:
    """Download a file with streaming chunks and progress logging."""
    if dest.exists() and dest.stat().st_size > 1000:
        log.info(f"  [skip] {dest.name} already downloaded ({dest.stat().st_size / (1024*1024):.1f} MB).")
        return True

    log.info(f"  Downloading {dest.name} from PhysioNet...")
    headers = {"User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) NeuroStage Research Pipeline"}
    req = urllib.request.Request(url, headers=headers)

    tmp_dest = dest.with_suffix(dest.suffix + ".tmp")
    try:
        with urllib.request.urlopen(req, timeout=60) as response, open(tmp_dest, "wb") as out_file:
            total_size = int(response.info().get("Content-Length", 0))
            downloaded = 0
            chunk_size = 1024 * 1024  # 1 MB chunk

            while True:
                chunk = response.read(chunk_size)
                if not chunk:
                    break
                out_file.write(chunk)
                downloaded += len(chunk)
                if total_size > 0:
                    pct = (downloaded / total_size) * 100
                    print(f"\r    Progress: {pct:.1f}% ({downloaded / (1024*1024):.1f} / {total_size / (1024*1024):.1f} MB)", end="", flush=True)

        print()
        if tmp_dest.exists():
            shutil.move(str(tmp_dest), str(dest))
        log.info(f"  ✓ Successfully saved {dest.name}")
        return True
    except Exception as e:
        log.error(f"  ✗ Failed downloading {url}: {e}")
        if tmp_dest.exists():
            tmp_dest.unlink(missing_ok=True)
        return False


def download_via_mne(out_dir: Path, n_subjects: int) -> list[dict]:
    """Use MNE's official sleep_physionet fetcher."""
    try:
        import mne
        log.info("Attempting download via MNE Sleep-PhysioNet downloader...")
        subject_indices = list(range(n_subjects))
        paths = mne.datasets.sleep_physionet.age.fetch_data(
            subjects=subject_indices,
            recording=[1],
            path=str(out_dir.parent),
            on_missing="warn",
        )
        manifest = []
        for pair in paths:
            if len(pair) == 2:
                psg, hyp = Path(pair[0]), Path(pair[1])
                target_psg = out_dir / psg.name
                target_hyp = out_dir / hyp.name
                if psg != target_psg and not target_psg.exists():
                    shutil.copy(str(psg), str(target_psg))
                if hyp != target_hyp and not target_hyp.exists():
                    shutil.copy(str(hyp), str(target_hyp))
                sub_id = psg.stem.split("-")[0][:6]
                manifest.append({"subject_id": sub_id, "psg": str(target_psg), "hypnogram": str(target_hyp)})
        return manifest
    except Exception as e:
        log.warning(f"MNE fetch_data encountered an issue: {e}. Falling back to direct streaming.")
        return []


def create_synthetic_data(out_dir: Path, n_subjects: int = 5) -> None:
    """Generate mock HDF5 preprocessed data so the user can test the entire pipeline immediately."""
    import h5py
    import numpy as np

    processed_dir = Path("data/processed")
    processed_dir.mkdir(parents=True, exist_ok=True)
    out_dir.mkdir(parents=True, exist_ok=True)

    log.info(f"Generating synthetic research-grade data for {n_subjects} subjects in {processed_dir} ...")
    subjects_manifest = []
    total_epochs = 0
    class_counts = {0: 0, 1: 0, 2: 0, 3: 0, 4: 0}
    stage_names = {0: "Wake", 1: "N1", 2: "N2", 3: "N3", 4: "REM"}

    np.random.seed(42)
    for i in range(n_subjects):
        sub_id = f"SC40{i+1:02d}"
        n_epochs = 300  # 300 epochs = 2.5 hours
        h5_path = processed_dir / f"{sub_id}.h5"

        # Generate realistic synthetic EEG waveforms (300 epochs, 1 channel, 3000 samples at 100Hz)
        t = np.linspace(0, 30, 3000, endpoint=False)
        epochs = np.zeros((n_epochs, 1, 3000), dtype=np.float32)
        labels = np.random.choice([0, 1, 2, 3, 4], size=n_epochs, p=[0.15, 0.10, 0.45, 0.15, 0.15])

        for e in range(n_epochs):
            st = labels[e]
            if st == 0:  # Wake (Beta / Alpha: 10-20 Hz)
                sig = 0.5 * np.sin(2 * np.pi * 10 * t) + 0.3 * np.sin(2 * np.pi * 20 * t)
            elif st == 1:  # N1 (Theta: 4-7 Hz)
                sig = 0.7 * np.sin(2 * np.pi * 5 * t)
            elif st == 2:  # N2 (Spindles: 12-14 Hz + Theta)
                sig = 0.6 * np.sin(2 * np.pi * 6 * t) + 0.8 * np.sin(2 * np.pi * 13 * t) * (np.sin(2 * np.pi * 0.5 * t) > 0.5)
            elif st == 3:  # N3 (Delta slow wave: 0.5-2 Hz high amplitude)
                sig = 1.8 * np.sin(2 * np.pi * 1.2 * t)
            else:  # REM (Mixed frequency, low voltage, saw-tooth)
                sig = 0.4 * np.sin(2 * np.pi * 6 * t) + 0.2 * np.sin(2 * np.pi * 18 * t)

            noise = np.random.normal(0, 0.15, size=3000)
            epochs[e, 0, :] = (sig + noise).astype(np.float32)

        # Normalize
        epochs = (epochs - epochs.mean()) / (epochs.std() + 1e-8)

        with h5py.File(h5_path, "w") as f:
            f.create_dataset("epochs", data=epochs, compression="gzip", compression_opts=4)
            f.create_dataset("labels", data=labels)
            f.attrs["subject_id"] = sub_id
            f.attrs["sfreq"] = 100
            f.attrs["epoch_sec"] = 30
            f.attrs["bandpass"] = [0.3, 35.0]

        sub_counts = {stage_names[s]: int((labels == s).sum()) for s in range(5)}
        for s in range(5):
            class_counts[s] += int((labels == s).sum())
        total_epochs += n_epochs

        subjects_manifest.append({
            "subject_id": sub_id,
            "h5_path": str(h5_path),
            "n_epochs": n_epochs,
            "class_counts": sub_counts,
        })
        log.info(f"  ✓ Created synthetic HDF5 for {sub_id} ({n_epochs} epochs)")

    manifest = {
        "dataset": "sleep-edf-synthetic",
        "n_subjects": n_subjects,
        "total_epochs": total_epochs,
        "class_counts": {stage_names[i]: class_counts[i] for i in range(5)},
        "preprocessing": {
            "sfreq": 100,
            "bandpass": [0.3, 35.0],
            "epoch_sec": 30,
            "normalization": "z-score-per-recording",
        },
        "subjects": subjects_manifest,
    }

    manifest_path = processed_dir / "manifest.json"
    with open(manifest_path, "w") as f:
        json.dump(manifest, f, indent=2)

    log.info(f"\nSynthetic data ready in {processed_dir} with manifest {manifest_path}!")
    log.info("You can now directly run: python train.py")


def main() -> None:
    parser = argparse.ArgumentParser(description="Download or generate Sleep-EDF dataset")
    parser.add_argument("--out", default="data/raw/sleep-edf", help="Output directory")
    parser.add_argument("--n_subjects", type=int, default=5, help="Number of subjects (default 5)")
    parser.add_argument("--mock", "--synthetic", action="store_true", help="Generate synthetic preprocessed data instantly")
    args = parser.parse_args()

    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)

    if args.mock:
        create_synthetic_data(out_dir, args.n_subjects)
        return

    pairs = SUBJECT_FILE_PAIRS[:min(args.n_subjects, len(SUBJECT_FILE_PAIRS))]
    log.info(f"Downloading {len(pairs)} subjects to {out_dir} ...")

    manifest = []
    for psg_file, hyp_file in pairs:
        psg_url = PHYSIONET_BASE_URL + psg_file
        hyp_url = PHYSIONET_BASE_URL + hyp_file
        psg_dest = out_dir / psg_file
        hyp_dest = out_dir / hyp_file

        ok1 = download_with_progress(psg_url, psg_dest)
        ok2 = download_with_progress(hyp_url, hyp_dest)
        if not ok2 and "EC-Hypnogram.edf" in hyp_file:
            alt_hyp = hyp_file.replace("EC-Hypnogram.edf", "EH-Hypnogram.edf")
            alt_url = PHYSIONET_BASE_URL + alt_hyp
            alt_dest = out_dir / alt_hyp
            ok2 = download_with_progress(alt_url, alt_dest)
            if ok2:
                hyp_dest = alt_dest
        if ok1 and ok2:
            manifest.append({
                "subject_id": psg_file[:6],
                "psg": str(psg_dest),
                "hypnogram": str(hyp_dest),
            })

    manifest_path = out_dir / "download_manifest.json"
    with open(manifest_path, "w") as f:
        json.dump({"dataset": "sleep-edf-cassette", "subjects": manifest}, f, indent=2)

    log.info(f"\nDownload finished! {len(manifest)} subjects ready in {out_dir}")
    log.info("Next step: python scripts/prepare_data.py")


if __name__ == "__main__":
    main()
