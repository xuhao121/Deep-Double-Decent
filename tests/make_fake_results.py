"""Write SYNTHETIC results for every run in runs.csv, to dry-run the analysis scripts early.

    export DD_RESULTS_DIR=/tmp/fake/results DD_OUTPUTS_DIR=/tmp/fake/outputs
    python tests/make_fake_results.py
    python -m analysis.make_figures all        # figures/ and tables/ are then built from fake data

The numbers only have a double-descent-like shape; they are NOT results. The script refuses to
run unless DD_RESULTS_DIR points to a directory whose path contains 'fake', so it can never
overwrite real results. Delete figures/ and tables/ afterwards.
"""
from __future__ import annotations

import math
import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np

from src.config import DATA_DIR, RESULTS_DIR, load_runs, out_dir, results_csv
from src.evaluate import CSV_COLUMNS, ResultsLogger, eval_schedule


def final_values(cfg, rng):
    """(train_err, test_err) at the end of training: decreasing curve + bump at a threshold width."""
    p = cfg.noise
    if cfg.model == "resnet18k":
        k_thr = 7 + 40 * p
        base = 0.07 + 0.55 * math.exp(-cfg.k / 3)
    else:
        k_thr = (4 + 30 * p) * (cfg.n / 50000) ** 0.6 * (1.6 if cfg.aug else 1.0) * (2.5 if cfg.dataset == "cifar100" else 1)
        base = (0.10 if cfg.aug else 0.25) + 0.5 * math.exp(-cfg.k / 4) + 0.1 * (1 - cfg.n / 50000)
        if cfg.dataset == "cifar100":
            base += 0.3
    shape = 1 / (1 + math.exp(-6 * (math.log(cfg.k) - math.log(k_thr))))       # 0 -> 1 around k_thr
    train = max(0.0, (p + 0.3 * math.exp(-cfg.k / 2)) * (1 - shape) + rng.normal(0, 0.002))
    bump = (0.9 * p + 0.04) * math.exp(-(math.log(cfg.k / k_thr)) ** 2 / 0.12) * (0.4 if cfg.wd > 0 else 1)
    test = min(0.95, base + bump + rng.normal(0, 0.008))
    return train, test, k_thr


def main() -> None:
    if "fake" not in str(RESULTS_DIR):
        sys.exit("set DD_RESULTS_DIR (and DD_OUTPUTS_DIR) to a directory containing 'fake' first")
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    y_path = DATA_DIR / "cifar10_test_y.npy"
    if not y_path.exists():                       # A7 needs test labels; fake ones if CIFAR is absent
        DATA_DIR.mkdir(parents=True, exist_ok=True)
        np.save(DATA_DIR / "cifar10_test_x.npy", np.zeros((10000, 32, 32, 3), np.uint8))
        np.save(y_path, np.random.default_rng(0).integers(0, 10, 10000))
    y_test = np.load(y_path)
    runs = load_runs()
    r6_widths = {r.k for r in runs if r.exp == "R6"}
    for cfg in runs:
        rng = np.random.default_rng(abs(hash(cfg.run_id)) % 2**32)
        train_f, test_f, k_thr = final_values(cfg, rng)
        ticks = eval_schedule(cfg.budget_unit, cfg.budget)
        T = ticks[-1]
        logger = ResultsLogger(results_csv(cfg.run_id))
        t_mem = T * 0.05 * (k_thr / cfg.k) ** 1.5                 # when noisy labels get memorised
        for t in ticks:
            prog = 1 - math.exp(-t / max(t_mem, 1))
            train = train_f + (0.9 - train_f) * math.exp(-t / max(t_mem / 3, 1))
            epoch_bump = (cfg.noise * 0.8) * math.exp(-(math.log(t / max(t_mem, 1))) ** 2 / 0.5) if cfg.k > k_thr else 0
            test = test_f + (0.9 - test_f) * math.exp(-t / max(t_mem / 10, 1)) + epoch_bump * (1 - t / T)
            row = {"epoch": t if cfg.unit == "epoch" else round(t / 391, 3), "step": t * 391 if cfg.unit == "epoch" else t,
                   "time_s": t * 2.0, "lr": cfg.lr, "train_loss_running": train * 2, "train_err": min(train, 0.9),
                   "train_loss": train * 2, "clean_part_err": train * 0.5,
                   "noisy_memorized": prog if cfg.noise > 0 else float("nan"),
                   "noisy_true": (1 - prog) * 0.6 if cfg.noise > 0 else float("nan"),
                   "test_err": min(test + rng.normal(0, 0.004), 0.95), "test_loss": test * 3}
            logger.log({c: row[c] for c in CSV_COLUMNS})
        if cfg.exp in ("R1", "R6") and cfg.k in r6_widths and cfg.noise == 0.15:
            d = out_dir(cfg.run_id)
            d.mkdir(parents=True, exist_ok=True)
            preds = np.full((len(ticks), 10000), 255, np.uint8)
            wrong = rng.random(10000) < test_f
            preds[-1] = np.where(wrong, (y_test + rng.integers(1, 10, 10000)) % 10, y_test)
            np.save(d / "test_preds.npy", preds)
            np.save(d / "test_preds_ticks.npy", np.asarray(ticks))
    print(f"fake results for {len(runs)} runs in {RESULTS_DIR}")


if __name__ == "__main__":
    main()
