"""Move not-yet-started runs from one owner to another (e.g. when m2's slices become free).

    python scripts/rebalance.py --from m1 --to m2 --exp R5 R6 R7 --fraction 0.5
    python scripts/rebalance.py --from m1 --to m2 --exp R5 --fraction 0.5 --dry-run

Run it on the machine of the *from* owner: it only moves runs that have not been started
there (no outputs/<run_id> directory and no results CSV). Then commit + push runs.csv.
Workers on the from-machine skip the moved runs automatically (train.py --owner check);
the other member pulls runs.csv, runs make_labels.py, assign_jobs.py and launch_tmux.sh.
"""
from __future__ import annotations

import argparse
import math
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.config import RUNS_CSV, load_runs, out_dir, results_csv, save_runs

PRIO_ORDER = {"P0": 0, "P1": 1, "P2": 2}


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--from", dest="src", required=True)
    ap.add_argument("--to", dest="dst", required=True)
    ap.add_argument("--exp", nargs="+", default=None, help="only these experiment blocks")
    ap.add_argument("--fraction", type=float, default=0.5, help="share of the movable estimated hours to move")
    ap.add_argument("--runs-csv", default=str(RUNS_CSV))
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args()

    runs = load_runs(args.runs_csv)
    movable = [r for r in runs if r.owner == args.src and (not args.exp or r.exp in args.exp)
               and not out_dir(r.run_id).exists() and not results_csv(r.run_id).exists()]
    hours = lambda r: 0.0 if math.isnan(r.est_hours) else r.est_hours
    total = sum(hours(r) for r in movable)
    # Move the *last* runs in execution order (low priority, short) so the from-owner keeps its queue head.
    movable.sort(key=lambda r: (PRIO_ORDER.get(r.priority, 9), -hours(r)), reverse=True)
    moved, acc = [], 0.0
    for r in movable:
        if acc >= args.fraction * total:
            break
        moved.append(r)
        acc += hours(r)
    for r in moved:
        r.owner = args.dst
    print(f"{len(moved)} of {len(movable)} movable runs ({acc:.1f} of {total:.1f} est. hours): {args.src} -> {args.dst}")
    for r in moved[:10]:
        print("  ", r.run_id)
    if len(moved) > 10:
        print(f"   ... and {len(moved) - 10} more")
    if not args.dry_run and moved:
        save_runs(runs, args.runs_csv)
        print(f"{args.runs_csv} updated - commit and push it")


if __name__ == "__main__":
    main()
