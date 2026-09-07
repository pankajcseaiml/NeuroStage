#!/usr/bin/env python
"""
prepare_data.py — Preprocess Sleep-EDF raw EDF files into HDF5 epoch files.

Usage:
    python scripts/prepare_data.py
    python scripts/prepare_data.py --raw data/raw/sleep-edf --out data/processed --config configs/base.yaml
"""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)s | %(message)s")

# Make sure the project root is in sys.path
sys.path.insert(0, str(Path(__file__).parent.parent))

from neurostage.config import load_config
from neurostage.data.preprocess import preprocess_dataset


def main() -> None:
    parser = argparse.ArgumentParser(description="Preprocess EEG data to HDF5")
    parser.add_argument("--raw", default="data/raw/sleep-edf")
    parser.add_argument("--out", default="data/processed")
    parser.add_argument("--config", default="configs/base.yaml")
    args = parser.parse_args()

    cfg = load_config(args.config)
    manifest = preprocess_dataset(
        raw_dir=args.raw,
        processed_dir=args.out,
        cfg=cfg,
        dataset_name="sleep-edf",
    )

    print(f"\nDone! {manifest['n_subjects']} subjects, {manifest['total_epochs']} total epochs.")
    print(f"Class distribution:")
    for stage, count in manifest["class_counts"].items():
        pct = 100 * count / manifest["total_epochs"] if manifest["total_epochs"] > 0 else 0
        print(f"  {stage:4s}: {count:6d} epochs ({pct:.1f}%)")


if __name__ == "__main__":
    main()
