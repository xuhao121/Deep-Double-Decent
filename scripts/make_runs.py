"""Create / extend runs.csv with the experiments R1-R13 of the plan.

    python scripts/make_runs.py                            # base runs (everything except R3 and R6)
    python scripts/make_runs.py --r3 7 9 11 13 18          # after P0: refinement widths around the peak
    python scripts/make_runs.py --r6 4 6 8 10 12 2 32 64   # widths of the 5-model ensembles (Fig. 28)
    python scripts/make_runs.py --update-estimates         # recompute est_hours from timing.json

Existing rows are never modified (except est_hours with --update-estimates), so re-running is safe.

Experiment blocks (owner m1 = ResNet18 line, m2 = 5-layer CNN line):
  R1  P0  ResNet18, 15 % noise, W16, seed 0                       -> Fig. 1, 2
  R2  P0  ResNet18, 20 % noise, W16, seed 0                       -> Fig. 9, 4b
  R3  P0  ResNet18, 15 % and 20 %, extra widths near the peak     -> denser peak
  R4  P1  ResNet18, 0 % and 10 % noise, W11                       -> Fig. 4b (0 % = negative control)
  R5  P1  ResNet18, 15 %, seeds 1-4, SAME noisy labels (ns0)      -> Fig. 1 mean +- std over 5 inits
  R6  P2  ResNet18, 15 %, seeds 1-4, INDEPENDENT noise (ns=seed)  -> Fig. 28 ensembles (with R1)
  R7  P2  ResNet18 k=64, 20 %, SGD + momentum 0.9, const lr 0.01  -> App. E.1 optimizer robustness
  R8  P0  CNN, 20 %, aug, n in {6250, 12500, 25000, 50000}, C14  -> Fig. 11a, 12
  R9  P0  CNN, 20 %, no aug, n = 50000                            -> Fig. 5
  R10 P1  CNN, {0 %, 10 %} x {aug, no aug}, n = 50000             -> Fig. 5
  R11 P1  CNN, 10 %, aug, weight decay 5e-4                       -> Fig. 21
  R12 P1  as R8, seeds 1-4 (new init, subset and noise per seed)  -> Fig. 11a error bars
  R13 P2  CNN, CIFAR-100, 0 %, no aug, 1M steps, seeds 0-4        -> Fig. 7 / 20
"""
from __future__ import annotations

import argparse
import math
import sys
from collections import defaultdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.config import (RUNS_CSV, TRAIN_SIZE, RunConfig, estimate_hours, load_runs, load_timing,
                        make_run_id, save_runs)

W16 = [1, 2, 3, 4, 5, 6, 8, 10, 12, 14, 16, 20, 24, 32, 48, 64]
W11 = [1, 2, 4, 6, 8, 10, 12, 16, 24, 32, 64]
C14 = [1, 2, 3, 4, 6, 8, 10, 12, 16, 20, 24, 32, 48, 64]
SUBSETS = [6250, 12500, 25000, 50000]


def resnet(exp, prio, k, noise, init_seed=0, noise_seed=0, opt="adam", lr=1e-4, momentum=0.0,
           sched="const", tag="") -> RunConfig:
    """ResNet18 on CIFAR-10: Adam 1e-4, batch 128, augmentation, 4000 epochs (paper defaults)."""
    return RunConfig(
        run_id=make_run_id("resnet18k", "cifar10", noise, noise_seed, k, init_seed, tag=tag),
        exp=exp, owner="m1", priority=prio, model="resnet18k", dataset="cifar10", k=k, noise=noise,
        noise_seed=noise_seed, init_seed=init_seed, subset_seed=0, n=TRAIN_SIZE, aug=1, opt=opt, lr=lr,
        momentum=momentum, wd=0.0, sched=sched, budget_unit="epochs", budget=4000)


def cnn(exp, prio, k, noise, n=TRAIN_SIZE, aug=1, wd=0.0, init_seed=0, noise_seed=0, subset_seed=0,
        dataset="cifar10", steps=500_000) -> RunConfig:
    """5-layer CNN: SGD lr 0.1 with inverse-sqrt schedule, no momentum, batch 128, 500k steps."""
    return RunConfig(
        run_id=make_run_id("mcnn", dataset, noise, noise_seed, k, init_seed, n=n, aug=aug, wd=wd),
        exp=exp, owner="m2", priority=prio, model="mcnn", dataset=dataset, k=k, noise=noise,
        noise_seed=noise_seed, init_seed=init_seed, subset_seed=subset_seed if n < TRAIN_SIZE else 0,
        n=n, aug=aug, opt="sgd", lr=0.1, momentum=0.0, wd=wd, sched="invsqrt", budget_unit="steps", budget=steps)


def base_runs(r5_widths: list[int]) -> list[RunConfig]:
    runs: list[RunConfig] = []
    runs += [resnet("R1", "P0", k, 0.15) for k in W16]
    runs += [resnet("R2", "P0", k, 0.20) for k in W16]
    runs += [resnet("R4", "P1", k, p) for p in (0.0, 0.10) for k in W11]
    runs += [resnet("R5", "P1", k, 0.15, init_seed=s) for s in range(1, 5) for k in r5_widths]
    runs.append(resnet("R7", "P2", 64, 0.20, opt="sgd", lr=0.01, momentum=0.9, sched="const", tag="sgdm"))
    runs += [cnn("R8", "P0", k, 0.20, n=n) for n in SUBSETS for k in C14]
    runs += [cnn("R9", "P0", k, 0.20, aug=0) for k in C14]
    runs += [cnn("R10", "P1", k, p, aug=a) for p in (0.0, 0.10) for a in (1, 0) for k in C14]
    runs += [cnn("R11", "P1", k, 0.10, aug=1, wd=5e-4) for k in C14]
    runs += [cnn("R12", "P1", k, 0.20, n=n, init_seed=s, noise_seed=s, subset_seed=s)
             for s in range(1, 5) for n in SUBSETS for k in C14]
    runs += [cnn("R13", "P2", k, 0.0, aug=0, init_seed=s, dataset="cifar100", steps=1_000_000)
             for s in range(5) for k in C14]
    return runs


def r3_runs(widths: list[int]) -> list[RunConfig]:
    return [resnet("R3", "P0", k, p) for p in (0.15, 0.20) for k in widths]


def r6_runs(widths: list[int]) -> list[RunConfig]:
    return [resnet("R6", "P2", k, 0.15, init_seed=s, noise_seed=s) for s in range(1, 5) for k in widths]


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--runs-csv", default=str(RUNS_CSV))
    ap.add_argument("--r3", nargs="+", type=int, default=None, help="extra ResNet widths near the peak (R3)")
    ap.add_argument("--r6", nargs="+", type=int, default=None, help="ResNet widths for the ensembles (R6)")
    ap.add_argument("--r5-widths", nargs="+", type=int, default=W16,
                    help="widths for the 4 extra seeds of R5 (default W16; e.g. 10 widths to halve the cost)")
    ap.add_argument("--update-estimates", action="store_true", help="recompute est_hours of all rows")
    args = ap.parse_args()

    path = Path(args.runs_csv)
    timing = load_timing()
    runs = load_runs(path) if path.exists() else []
    known = {r.run_id for r in runs}

    new = base_runs(args.r5_widths)
    if args.r3:
        new += r3_runs(args.r3)
    if args.r6:
        new += r6_runs(args.r6)
    added = 0
    for r in new:
        if r.run_id in known:
            continue                       # never touch existing rows (also skips R3 widths already in W16)
        r.est_hours = estimate_hours(r, timing)
        runs.append(r)
        known.add(r.run_id)
        added += 1
    if args.update_estimates:
        for r in runs:
            r.est_hours = estimate_hours(r, timing)
    save_runs(runs, path)

    print(f"{path}: {len(runs)} runs ({added} added){'' if timing else '  [estimates: default assumptions]'}")
    by_exp = defaultdict(lambda: [0, 0.0])
    for r in runs:
        by_exp[(r.owner, r.priority, r.exp)][0] += 1
        by_exp[(r.owner, r.priority, r.exp)][1] += 0 if math.isnan(r.est_hours) else r.est_hours
    print(f"{'owner':6}{'prio':6}{'exp':6}{'runs':>6}{'slice-h':>10}")
    for (owner, prio, exp), (count, hours) in sorted(by_exp.items(), key=lambda x: (x[0][0], x[0][1], int(x[0][2][1:]))):
        print(f"{owner:6}{prio:6}{exp:6}{count:6d}{hours:10.1f}")
    print(f"total slice-hours: {sum(v[1] for v in by_exp.values()):.1f}")


if __name__ == "__main__":
    main()
