"""Progress of the runs (reads runs.csv, results/*.csv and the markers in outputs/).

    python scripts/status.py                    # summary per experiment block
    python scripts/status.py --owner m1 --detail
    python scripts/status.py --exp R1 R2 --detail

Status: done (DONE marker) | running (fresh RUNNING marker) | paused (checkpoint, no process)
        | crashed (results exist, no checkpoint, no process) | todo.
"""
from __future__ import annotations

import argparse
import csv
import math
import sys
from collections import defaultdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.config import RUNS_CSV, load_runs, out_dir, results_csv
from src.utils import marker_is_live


def last_row(path: Path) -> dict | None:
    if not path.exists():
        return None
    with open(path, newline="") as f:
        rows = list(csv.DictReader(f))
    return rows[-1] if rows else None


def run_status(run_id: str, stale_minutes: float) -> str:
    odir = out_dir(run_id)
    if (odir / "DONE").exists():
        return "done"
    if marker_is_live(odir / "RUNNING", stale_minutes):
        return "running"
    if (odir / "ckpt.pt").exists():
        return "paused"
    if results_csv(run_id).exists():
        return "crashed"
    return "todo"


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--runs-csv", default=str(RUNS_CSV))
    ap.add_argument("--owner", default=None)
    ap.add_argument("--exp", nargs="+", default=None)
    ap.add_argument("--detail", action="store_true", help="one line per started run")
    ap.add_argument("--stale-minutes", type=float, default=120.0)
    args = ap.parse_args()

    summary = defaultdict(lambda: defaultdict(int))
    remaining = defaultdict(float)
    lines = []
    for cfg in load_runs(args.runs_csv):
        if (args.owner and cfg.owner != args.owner) or (args.exp and cfg.exp not in args.exp):
            continue
        st = run_status(cfg.run_id, args.stale_minutes)
        summary[cfg.exp][st] += 1
        row = last_row(results_csv(cfg.run_id))
        est = 0.0 if math.isnan(cfg.est_hours) else cfg.est_hours
        frac = 0.0
        eta = float("nan")
        if row:
            tick = float(row["epoch"] if cfg.unit == "epoch" else row["step"])
            frac = min(tick / cfg.budget, 1.0)
            t = float(row["time_s"])
            eta = t * (1 / frac - 1) / 3600 if frac > 0 else float("nan")
        if st != "done":
            remaining[cfg.exp] += est * (1 - frac)
        if args.detail and st != "todo":
            lines.append(f"{cfg.run_id:46s} {st:8s} {100 * frac:6.1f}%  "
                         f"train_err {row['train_err'] if row else '-':>8}  test_err {row['test_err'] if row else '-':>8}  "
                         f"eta {eta:6.2f} h")
    if lines:
        print("\n".join(lines) + "\n")
    states = ["done", "running", "paused", "crashed", "todo"]
    print(f"{'exp':5}" + "".join(f"{s:>9}" for s in states) + f"{'left (h)':>11}")
    for exp in sorted(summary, key=lambda e: int(e[1:]) if e[1:].isdigit() else 99):
        print(f"{exp:5}" + "".join(f"{summary[exp][s]:9d}" for s in states) + f"{remaining[exp]:11.1f}")
    print(f"{'all':5}" + "".join(f"{sum(summary[e][s] for e in summary):9d}" for s in states)
          + f"{sum(remaining.values()):11.1f}")
    print("(left = estimated single-process slice-hours still to run)")


if __name__ == "__main__":
    main()
