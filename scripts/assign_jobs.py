"""Distribute not-yet-finished runs over MIG slices ("lanes"), longest job first.

    python scripts/assign_jobs.py --owner m1                    # all unfinished runs of m1
    python scripts/assign_jobs.py --owner m1 --priority P0 P1   # only some priorities
    python scripts/assign_jobs.py --owner m2 --light-lanes 3 --heavy-slices 1
then start the workers:
    bash scripts/launch_tmux.sh jobs/m1/lanes.tsv m1

Heavy runs (ResNet k>=32, CNN k>=48) get a whole slice (one lane per slice); light runs share
a slice (--light-lanes processes per slice). Within each lane P0 runs come first, then P1, P2;
within a priority the longest runs come first. Runs with a DONE marker or a fresh RUNNING
marker are skipped. Slices are read from `nvidia-smi -L` unless --slices is given.
"""
from __future__ import annotations

import argparse
import re
import subprocess
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.config import ROOT, RUNS_CSV, estimate_hours, is_heavy, load_runs, load_timing, out_dir
from src.utils import marker_is_live

DEFAULT_SLOWDOWN = {1: 1.0, 2: 1.25, 3: 1.5}   # per-process slow-down when sharing a slice (pilot_timing)
PRIORITIES = ["P0", "P1", "P2"]


def detect_slices() -> list[str]:
    """MIG device UUIDs from `nvidia-smi -L` (falls back to full-GPU UUIDs if MIG is off)."""
    out = subprocess.run(["nvidia-smi", "-L"], capture_output=True, text=True, check=True).stdout
    mig = re.findall(r"UUID:\s*(MIG-[^\s)]+)", out)
    return mig if mig else re.findall(r"UUID:\s*(GPU-[^\s)]+)", out)


def lpt(jobs_by_prio: dict, n_lanes: int, slowdown: float):
    """Longest-processing-time-first, priority by priority; returns (lanes, loads in hours)."""
    lanes = [[] for _ in range(n_lanes)]
    loads = [0.0] * n_lanes
    if n_lanes == 0:
        return lanes, loads
    for prio in PRIORITIES:
        for hours, cfg in sorted(jobs_by_prio.get(prio, []), key=lambda x: -x[0]):
            i = min(range(n_lanes), key=lambda j: loads[j])
            lanes[i].append(cfg)
            loads[i] += hours * slowdown
    return lanes, loads


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--owner", required=True, help="m1 or m2: only runs of this owner")
    ap.add_argument("--runs-csv", default=str(RUNS_CSV))
    ap.add_argument("--priority", nargs="+", default=PRIORITIES)
    ap.add_argument("--exp", nargs="+", default=None, help="restrict to experiment blocks, e.g. R1 R2")
    ap.add_argument("--slices", nargs="+", default=None, help="MIG UUIDs (default: from nvidia-smi -L)")
    ap.add_argument("--heavy-slices", type=int, default=None, help="slices reserved for heavy runs (default: auto)")
    ap.add_argument("--light-lanes", type=int, default=3, help="concurrent light runs per slice")
    ap.add_argument("--stale-minutes", type=float, default=120.0)
    ap.add_argument("--out", default=None, help="output directory (default jobs/<owner>)")
    args = ap.parse_args()

    timing = load_timing()
    slowdown_tab = {int(k): v for k, v in (timing or {}).get("slowdown", {}).items()} or DEFAULT_SLOWDOWN
    light_slow = slowdown_tab.get(args.light_lanes, DEFAULT_SLOWDOWN.get(args.light_lanes, 1.5))
    slices = args.slices or detect_slices()
    if not slices:
        sys.exit("no GPU / MIG devices found")

    heavy_jobs, light_jobs = {}, {}
    skipped = 0
    for cfg in load_runs(args.runs_csv):
        if cfg.owner != args.owner or cfg.priority not in args.priority:
            continue
        if args.exp and cfg.exp not in args.exp:
            continue
        odir = out_dir(cfg.run_id)
        if (odir / "DONE").exists() or marker_is_live(odir / "RUNNING", args.stale_minutes):
            skipped += 1
            continue
        hours = estimate_hours(cfg, timing)
        target = heavy_jobs if is_heavy(cfg) else light_jobs
        target.setdefault(cfg.priority, []).append((hours, cfg))

    n_heavy_runs = sum(len(v) for v in heavy_jobs.values())
    n_light_runs = sum(len(v) for v in light_jobs.values())
    if n_heavy_runs + n_light_runs == 0:
        sys.exit(f"nothing to assign for {args.owner} ({skipped} runs done or running)")

    # Choose how many slices serve heavy runs: minimise the estimated makespan.
    S = len(slices)
    candidates = [args.heavy_slices] if args.heavy_slices is not None else range(S + 1)
    best = None
    for h in candidates:
        if (n_heavy_runs and h == 0) or (n_light_runs and h == S):
            continue
        hl, hloads = lpt(heavy_jobs, h, 1.0)
        ll, lloads = lpt(light_jobs, (S - h) * args.light_lanes, light_slow)
        makespan = max(hloads + lloads + [0.0])
        if best is None or makespan < best[0]:
            best = (makespan, h, hl, hloads, ll, lloads)
    if best is None:
        sys.exit("cannot place the runs: use more slices or change --heavy-slices")
    makespan, h, hl, hloads, ll, lloads = best

    out = Path(args.out) if args.out else ROOT / "jobs" / args.owner
    out.mkdir(parents=True, exist_ok=True)
    for old in out.glob("lane_*.txt"):
        old.unlink()
    rows = []
    for i, (lane, load) in enumerate(zip(hl, hloads)):
        rows.append((f"lane_s{i}_heavy", slices[i], lane, load))
    for j, (lane, load) in enumerate(zip(ll, lloads)):
        s = h + j // args.light_lanes
        rows.append((f"lane_s{s}_light{j % args.light_lanes}", slices[s], lane, load))
    with open(out / "lanes.tsv", "w") as f:
        f.write("lane\tdevice\tjob_list\n")
        for name, dev, lane, load in rows:
            if not lane:
                continue
            path = out / f"{name}.txt"
            path.write_text("".join(c.run_id + "\n" for c in lane))
            f.write(f"{name}\t{dev}\t{path}\n")

    print(f"{args.owner}: {n_heavy_runs} heavy + {n_light_runs} light runs on {S} slices "
          f"({h} heavy, {S - h} x {args.light_lanes} light lanes); {skipped} skipped (done/running)")
    for name, dev, lane, load in rows:
        if lane:
            print(f"  {name:22s} {dev[:22]:22s} {len(lane):4d} runs  ~{load:6.1f} h   first: {lane[0].run_id}")
    print(f"estimated makespan ~{makespan:.1f} h  ->  bash scripts/launch_tmux.sh {out / 'lanes.tsv'} {args.owner}")


if __name__ == "__main__":
    main()
