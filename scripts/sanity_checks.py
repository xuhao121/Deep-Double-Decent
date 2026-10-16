"""Automated versions of the sanity checks in step 2 of the plan.

    python scripts/sanity_checks.py params        # check 1: parameter counts (CNN k=64 must be 1,558,026)
    python scripts/sanity_checks.py noise         # check 2: label-noise statistics
    python scripts/sanity_checks.py aug           # check 3: augmentation grid -> notes/aug_grid.png
    python scripts/sanity_checks.py overfit       # check 4: ResNet18 k=64 fits 1000 images to 0 train error
    python scripts/sanity_checks.py test-runs     # writes runs_test.csv with two tiny runs (checks 6-8)
    python scripts/sanity_checks.py csv --run <run_id> [--runs-csv runs_test.csv]   # checks 6-8 on a CSV

Check 5 (CNN k=64, 0 % noise, augmentation reaches > 90 % test accuracy) is simply the R10 run
cnn_c10_p00_ns0_k064_s0_n50000_aug1_wd0 - start it first and look at its final test_err.

Checks 6-8 (resume, error definition, complete logs):
    python scripts/sanity_checks.py test-runs
    python scripts/make_labels.py --runs-csv runs_test.csv
    python -m src.train --run test_r18_c10_p15_ns0_k004_s0 --runs-csv runs_test.csv   # press Ctrl-C once mid-way
    python -m src.train --run test_r18_c10_p15_ns0_k004_s0 --runs-csv runs_test.csv   # resumes
    python scripts/sanity_checks.py csv --run test_r18_c10_p15_ns0_k004_s0 --runs-csv runs_test.csv
"""
from __future__ import annotations

import argparse
import csv
import math
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np
import torch

from src.config import ROOT, TRAIN_SIZE, RunConfig, get_run, results_csv, save_runs
from src.data import CIFAR10_MEAN, CIFAR10_STD, GPUData, load_split
from src.engine import train_step
from src.evaluate import CSV_COLUMNS, evaluate
from src.models import build_model, count_params
from src.noise import label_path, load_label_file, make_noisy_labels
from src.schedules import build_optimizer, build_scheduler, invsqrt_factor

CNN_K64_PARAMS = 1_558_026   # paper, Appendix B.1


def report(name: str, ok: bool, detail: str = "") -> bool:
    print(f"[{'PASS' if ok else 'FAIL'}] {name}" + (f"  - {detail}" if detail else ""))
    return ok


def check_params() -> bool:
    ok = True
    for model in ("mcnn", "resnet18k"):
        for k in (1, 16, 64):
            print(f"{model:10s} k={k:<3d} {count_params(build_model(model, k, 10)):>12,d} parameters")
    n = count_params(build_model("mcnn", 64, 10))
    ok &= report("CNN k=64 parameter count", n == CNN_K64_PARAMS, f"{n:,} (paper: {CNN_K64_PARAMS:,})")
    n = count_params(build_model("resnet18k", 64, 10))
    ok &= report("ResNet18 k=64 is about 11.2M parameters", 10.5e6 < n < 11.8e6, f"{n:,}")
    return ok


def check_noise() -> bool:
    from scipy.stats import chisquare
    y_true = load_split("cifar10", "train")[1]
    ok = True
    for p in (0.10, 0.15, 0.20):
        y_noisy, flip = make_noisy_labels(y_true, p, 10, noise_seed=0)
        rate = flip.mean()
        ok &= report(f"p={p}: realized flip rate", abs(rate - p) < 0.005, f"{rate:.4f}")
        ok &= report(f"p={p}: flipped labels are never the true class", bool((y_noisy[flip] != y_true[flip]).all()))
        ok &= report(f"p={p}: unflipped labels unchanged", bool((y_noisy[~flip] == y_true[~flip]).all()))
        offsets = (y_noisy[flip] - y_true[flip]) % 10           # 1..9, should be uniform
        pval = chisquare(np.bincount(offsets, minlength=10)[1:]).pvalue
        ok &= report(f"p={p}: wrong class uniform (chi-square)", pval > 1e-3, f"p-value {pval:.3f}")
        if label_path("cifar10", p, 0).exists():
            y_file, flip_file, _, _ = load_label_file("cifar10", p, 0)
            ok &= report(f"p={p}: labels/ file equals a fresh draw", bool((y_file == y_noisy).all() and (flip_file == flip).all()))
    _, f15 = make_noisy_labels(y_true, 0.15, 10, 0)
    _, f20 = make_noisy_labels(y_true, 0.20, 10, 0)
    ok &= report("noise levels nested (15 % flips subset of 20 % flips)", bool((~f15 | f20).all()))
    return ok


def check_aug() -> bool:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    x, y = load_split("cifar10", "train")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    data = GPUData(x[:8], y[:8], x[:8], y[:8], CIFAR10_MEAN, CIFAR10_STD, aug=True, batch_size=8, device=device)
    gen = torch.Generator(device=device)
    gen.manual_seed(0)
    from src.data import random_crop_flip
    idx = torch.arange(8, device=device)
    a = random_crop_flip(data.x_train_pad, idx, gen).cpu().numpy()
    b = random_crop_flip(data.x_train_pad, idx, gen).cpu().numpy()
    fig, axes = plt.subplots(3, 8, figsize=(12, 4.8))
    for j in range(8):
        for i, img in enumerate((x[j], a[j].transpose(1, 2, 0), b[j].transpose(1, 2, 0))):
            axes[i, j].imshow(img)
            axes[i, j].axis("off")
    for i, t in enumerate(("original", "augmentation 1", "augmentation 2")):
        axes[i, 0].set_title(t, fontsize=9, loc="left")
    out = ROOT / "notes" / "aug_grid.png"
    out.parent.mkdir(exist_ok=True)
    fig.tight_layout()
    fig.savefig(out, dpi=120)
    ok = report("two augmentation passes differ", bool((a != b).any()))
    ok &= report("augmented batch shape 3x32x32", a.shape == (8, 3, 32, 32))
    print(f"inspect {out}: crops shifted by up to 4 px with black borders, about half flipped")
    return ok


def check_overfit(max_epochs: int = 200) -> bool:
    """ResNet18 k=64 on 1000 images without augmentation must reach 0 training error."""
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    x, y = load_split("cifar10", "train")
    data = GPUData(x[:1000], y[:1000], x[:1000], y[:1000], CIFAR10_MEAN, CIFAR10_STD, aug=False,
                   batch_size=128, device=device)
    cfg = RunConfig("overfit", "-", "-", "-", "resnet18k", "cifar10", 64, 0.0, 0, 0, 0, 1000, 0,
                    "adam", 1e-4, 0.0, 0.0, "const", "epochs", max_epochs)
    torch.manual_seed(0)
    model = build_model("resnet18k", 64, 10).to(device).to(memory_format=torch.channels_last)
    opt = build_optimizer(model, cfg, device)
    gen = torch.Generator(device=device)
    gen.manual_seed(0)
    amp = device.type == "cuda"
    flip = torch.zeros(1000, dtype=torch.bool, device=device)
    for epoch in range(1, max_epochs + 1):
        model.train()
        for xb, yb in data.train_batches(gen):
            train_step(model, opt, None, xb, yb, amp)
        if epoch % 10 == 0:
            m, _, _ = evaluate(model, data, data.y_train, flip, amp)
            print(f"epoch {epoch:3d}: train_err {m['train_err']:.4f}")
            if m["train_err"] == 0.0:
                return report("ResNet18 k=64 fits 1000 images", True, f"0 train error after {epoch} epochs")
    return report("ResNet18 k=64 fits 1000 images", False, f"train error > 0 after {max_epochs} epochs")


def write_test_runs() -> None:
    """Two tiny runs with distinct ids for the resume / logging checks."""
    r18 = RunConfig("test_r18_c10_p15_ns0_k004_s0", "TEST", "m1", "P0", "resnet18k", "cifar10", 4, 0.15, 0, 0, 0, TRAIN_SIZE, 1,
                    "adam", 1e-4, 0.0, 0.0, "const", "epochs", 20)
    cnn = RunConfig("test_cnn_c10_p20_ns0_k008_s0_n6250_aug1_wd0", "TEST", "m2", "P0", "mcnn", "cifar10", 8,
                    0.20, 0, 0, 0, 6250, 1, "sgd", 0.1, 0.0, 0.0, "invsqrt", "steps", 3000)
    path = ROOT / "runs_test.csv"
    save_runs([r18, cnn], path)
    print(f"wrote {path}:\n  {r18.run_id}  (20 epochs)\n  {cnn.run_id}  (3000 steps)")


def check_csv(run_id: str, runs_csv: str) -> bool:
    """Checks 6-8 on a finished (or running) run: complete columns, monotone steps, valid errors, lr."""
    cfg = get_run(run_id, runs_csv)
    path = results_csv(run_id)
    with open(path, newline="") as f:
        rows = list(csv.DictReader(f))
    ok = report("CSV has all columns", list(rows[0].keys()) == CSV_COLUMNS if rows else False)
    steps = [int(float(r["step"])) for r in rows]
    ok &= report("steps strictly increasing (no duplicates after resume)", all(a < b for a, b in zip(steps, steps[1:])))
    times = [float(r["time_s"]) for r in rows]
    ok &= report("elapsed time increasing", all(a <= b for a, b in zip(times, times[1:])))
    errs = [float(r[c]) for r in rows for c in ("train_err", "test_err")]
    ok &= report("train_err / test_err are 0-1 errors", all(0.0 <= e <= 1.0 for e in errs))
    missing = [c for r in rows for c in ("train_err", "test_err", "train_loss", "test_loss") if r[c] in ("", "nan")]
    ok &= report("no missing values in the main columns", not missing)
    if cfg.sched == "invsqrt":
        bad = [s for s, r in zip(steps, rows) if not math.isclose(float(r["lr"]), cfg.lr * invsqrt_factor(s), rel_tol=1e-4)]
        ok &= report("lr follows lr0 / sqrt(1 + floor(t/512))", not bad, f"{len(bad)} mismatches")
    else:
        ok &= report("lr constant", all(math.isclose(float(r["lr"]), cfg.lr) for r in rows))
    print(f"{len(rows)} rows, last: step {steps[-1] if steps else '-'}, test_err {rows[-1]['test_err'] if rows else '-'}")
    return ok


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("check", choices=["params", "noise", "aug", "overfit", "test-runs", "csv"])
    ap.add_argument("--run", default=None)
    ap.add_argument("--runs-csv", default=str(ROOT / "runs.csv"))
    args = ap.parse_args()
    if args.check == "params":
        ok = check_params()
    elif args.check == "noise":
        ok = check_noise()
    elif args.check == "aug":
        ok = check_aug()
    elif args.check == "overfit":
        ok = check_overfit()
    elif args.check == "test-runs":
        write_test_runs()
        ok = True
    else:
        if not args.run:
            sys.exit("--run is required for the csv check")
        ok = check_csv(args.run, args.runs_csv)
    sys.exit(0 if ok else 1)


if __name__ == "__main__":
    main()
