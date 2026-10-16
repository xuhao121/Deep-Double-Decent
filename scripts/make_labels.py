"""Pre-generate every noisy-label file and subset permutation needed by runs.csv.

    python scripts/make_labels.py
    python scripts/make_labels.py --runs-csv runs_test.csv

Files go to labels/ - commit them. Existing files are never overwritten, so every run that
shares (dataset, p, noise_seed) uses exactly the same labels on both machines.
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np

from src.config import NUM_CLASSES, RUNS_CSV, TRAIN_SIZE, load_runs
from src.data import load_split
from src.noise import create_label_file, create_subset_file, load_label_file
from src.utils import sha256_file


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--runs-csv", default=str(RUNS_CSV))
    args = ap.parse_args()

    runs = load_runs(args.runs_csv)
    label_keys = sorted({(r.dataset, r.noise, r.noise_seed) for r in runs})
    subset_keys = sorted({(r.dataset, r.subset_seed) for r in runs if r.n < TRAIN_SIZE})

    y_true = {}
    for dataset, p, ns in label_keys:
        if dataset not in y_true:
            y_true[dataset] = load_split(dataset, "train")[1]
        path = create_label_file(dataset, p, ns, y_true[dataset], NUM_CLASSES[dataset])
        y_noisy, flip, y_saved, _ = load_label_file(dataset, p, ns)
        assert np.array_equal(y_saved, y_true[dataset]), f"{path}: stored true labels differ from data/"
        print(f"{path.name:32s} p={p:<5g} realized={flip.mean():.4f}  sha256={sha256_file(path)[:12]}")
    for dataset, ss in subset_keys:
        path = create_subset_file(dataset, ss)
        print(f"{path.name:32s} sha256={sha256_file(path)[:12]}")


if __name__ == "__main__":
    main()
