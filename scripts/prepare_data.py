"""Download CIFAR-10 / CIFAR-100 with torchvision and store them as uint8 .npy arrays.

    python scripts/prepare_data.py                      # both datasets -> data/
    python scripts/prepare_data.py --datasets cifar10

Writes data/<dataset>_{train,test}_{x,y}.npy (x: [N,32,32,3] uint8, y: int64) and
data/<dataset>_stats.json (per-channel mean/std of the training set).
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np

from src.config import DATA_DIR
from src.data import compute_stats


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--datasets", nargs="+", default=["cifar10", "cifar100"], choices=["cifar10", "cifar100"])
    ap.add_argument("--raw-dir", default=None, help="where torchvision downloads the archives (default data/raw)")
    args = ap.parse_args()

    import torchvision  # only needed here

    raw = Path(args.raw_dir) if args.raw_dir else DATA_DIR / "raw"
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    for name in args.datasets:
        cls = {"cifar10": torchvision.datasets.CIFAR10, "cifar100": torchvision.datasets.CIFAR100}[name]
        for split, train in (("train", True), ("test", False)):
            ds = cls(root=str(raw), train=train, download=True)
            x = np.asarray(ds.data, dtype=np.uint8)        # [N,32,32,3]
            y = np.asarray(ds.targets, dtype=np.int64)
            np.save(DATA_DIR / f"{name}_{split}_x.npy", x)
            np.save(DATA_DIR / f"{name}_{split}_y.npy", y)
            counts = np.bincount(y)
            print(f"{name} {split}: x {x.shape}, {len(counts)} classes, {counts.min()}-{counts.max()} per class")
        mean, std = compute_stats(np.load(DATA_DIR / f"{name}_train_x.npy"))
        (DATA_DIR / f"{name}_stats.json").write_text(json.dumps({"mean": mean, "std": std}, indent=2))
        print(f"{name} train mean {np.round(mean, 4)} std {np.round(std, 4)}")


if __name__ == "__main__":
    main()
