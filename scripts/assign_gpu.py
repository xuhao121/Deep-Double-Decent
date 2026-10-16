"""Single full GPU (no MIG): spread the unfinished runs of one owner over N concurrent lanes.

    python scripts/assign_gpu.py --owner m1 --lanes 8
    bash scripts/launch_tmux.sh jobs/m1/lanes.tsv m1
"""
import argparse, re, subprocess, sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src.config import ROOT, RUNS_CSV, estimate_hours, load_runs, load_timing, out_dir
try:
    from src.utils import marker_is_live
except ImportError:                                   # older src/utils.py
    def marker_is_live(path, stale_minutes=120.0):
        return Path(path).exists()

ap = argparse.ArgumentParser()
ap.add_argument("--owner", required=True)
ap.add_argument("--lanes", type=int, default=8, help="number of runs on the GPU at the same time")
ap.add_argument("--priority", nargs="+", default=["P0", "P1", "P2"])
ap.add_argument("--exp", nargs="+", default=None)
ap.add_argument("--device", default=None, help="GPU UUID (default: first GPU of nvidia-smi -L)")
ap.add_argument("--runs-csv", default=str(RUNS_CSV))
args = ap.parse_args()

dev = args.device or re.findall(r"UUID:\s*(GPU-[^\s)]+)",
                                subprocess.run(["nvidia-smi", "-L"], capture_output=True, text=True).stdout)[0]
timing = load_timing()
slow = {int(k): v for k, v in (timing or {}).get("slowdown", {}).items()}.get(args.lanes, float(args.lanes))
jobs = [(estimate_hours(c, timing), c) for c in load_runs(args.runs_csv)
        if c.owner == args.owner and c.priority in args.priority and (not args.exp or c.exp in args.exp)
        and not (out_dir(c.run_id) / "DONE").exists() and not marker_is_live(out_dir(c.run_id) / "RUNNING")]
if not jobs:
    sys.exit(f"nothing to assign for {args.owner}")
lanes, loads = [[] for _ in range(args.lanes)], [0.0] * args.lanes
for prio in ("P0", "P1", "P2"):                       # P0 first in every lane, longest runs first
    for h, c in sorted((j for j in jobs if j[1].priority == prio), key=lambda j: -j[0]):
        i = loads.index(min(loads))
        lanes[i].append(c.run_id)
        loads[i] += h * slow
out = ROOT / "jobs" / args.owner
out.mkdir(parents=True, exist_ok=True)
for old in out.glob("lane_*.txt"):
    old.unlink()
with open(out / "lanes.tsv", "w") as f:
    f.write("lane\tdevice\tjob_list\n")
    for i, lane in enumerate(lanes):
        if lane:
            p = out / f"lane_g{i}.txt"
            p.write_text("\n".join(lane) + "\n")
            f.write(f"lane_g{i}\t{dev}\t{p}\n")
print(f"{args.owner}: {len(jobs)} runs on {args.lanes} lanes of {dev}; estimated makespan ~{max(loads):.1f} h")
for i, (lane, load) in enumerate(zip(lanes, loads)):
    print(f"  lane_g{i}: {len(lane):3d} runs  ~{load:5.1f} h   first: {lane[0] if lane else '-'}")
