"""Measure training speed on one MIG slice and write timing.json (used for estimates and scheduling).

Set CUDA_VISIBLE_DEVICES to one MIG UUID first, then:
    python scripts/pilot_timing.py single --model resnet18k --k 2 8 16 32 64 --epochs 3
    python scripts/pilot_timing.py single --model mcnn --k 4 16 64 --steps 2000
    python scripts/pilot_timing.py concurrent --model resnet18k --k 8 16 --procs 2 3
    python scripts/pilot_timing.py concurrent --model mcnn --k 4 16 --procs 2 3
    python scripts/pilot_timing.py merge          # -> timing.json, then: python scripts/make_runs.py --update-estimates

Measurements are appended to timing/measurements.jsonl. Real CIFAR-10 is used when data/ exists,
otherwise random images of the same shape (speed does not depend on the pixel values).
The training step, data pipeline and optimiser settings are exactly those of src/train.py.
"""
from __future__ import annotations

import argparse
import json
import statistics
import subprocess
import sys
import time
from collections import defaultdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np
import torch

from src.config import ROOT, TIMING_JSON, TRAIN_SIZE, RunConfig
from src.data import CIFAR10_MEAN, CIFAR10_STD, GPUData, load_split
from src.engine import train_step
from src.models import build_model
from src.schedules import build_optimizer, build_scheduler
from src.utils import device_name, hostname, now

LOG = ROOT / "timing" / "measurements.jsonl"


def make_data(device) -> GPUData:
    try:
        x_tr, y_tr = load_split("cifar10", "train")
        x_te, y_te = load_split("cifar10", "test")
    except FileNotFoundError:
        rng = np.random.default_rng(0)
        x_tr = rng.integers(0, 256, (TRAIN_SIZE, 32, 32, 3), dtype=np.uint8)
        y_tr = rng.integers(0, 10, TRAIN_SIZE)
        x_te, y_te = x_tr[:10000], y_tr[:10000]
    return GPUData(x_tr, y_tr, x_te, y_te, CIFAR10_MEAN, CIFAR10_STD, aug=True, batch_size=128, device=device)


def paper_cfg(model: str, k: int) -> RunConfig:
    if model == "resnet18k":
        return RunConfig("pilot", "pilot", "-", "-", model, "cifar10", k, 0.0, 0, 0, 0, TRAIN_SIZE, 1,
                         "adam", 1e-4, 0.0, 0.0, "const", "epochs", 1)
    return RunConfig("pilot", "pilot", "-", "-", model, "cifar10", k, 0.0, 0, 0, 0, TRAIN_SIZE, 1,
                     "sgd", 0.1, 0.0, 0.0, "invsqrt", "steps", 1)


def measure(model_name: str, k: int, epochs: int, steps: int, procs: int, compile_mode: str) -> dict:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    amp = device.type == "cuda"
    torch.backends.cudnn.benchmark = True
    data = make_data(device)
    cfg = paper_cfg(model_name, k)
    model = build_model(model_name, k, 10).to(device).to(memory_format=torch.channels_last)
    opt = build_optimizer(model, cfg, device)
    sched = build_scheduler(opt, cfg)
    step_model = model if compile_mode == "none" else torch.compile(
        model, mode=None if compile_mode == "default" else compile_mode)
    gen = torch.Generator(device=device)
    gen.manual_seed(0)

    def run_steps(n_steps: int) -> None:
        done = 0
        while done < n_steps:
            for xb, yb in data.train_batches(gen):
                train_step(step_model, opt, sched, xb, yb, amp)
                done += 1
                if done >= n_steps:
                    break

    spe = data.steps_per_epoch
    n_timed = epochs * spe if model_name == "resnet18k" else steps
    run_steps(spe if model_name == "resnet18k" else min(300, steps))   # warm-up (cudnn autotune, compile)
    if device.type == "cuda":
        torch.cuda.synchronize()
    t0 = time.time()
    run_steps(n_timed)
    if device.type == "cuda":
        torch.cuda.synchronize()
    sec_per_step = (time.time() - t0) / n_timed
    rec = {"model": model_name, "k": k, "procs": procs, "compile": compile_mode,
           "sec_per_step": sec_per_step, "sec_per_epoch": sec_per_step * spe,
           "imgs_per_sec": 128 / sec_per_step, "device": device_name(device), "host": hostname(), "time": now()}
    LOG.parent.mkdir(parents=True, exist_ok=True)
    with open(LOG, "a") as f:
        f.write(json.dumps(rec) + "\n")
    print(f"{model_name} k={k:<3d} procs={procs}  {rec['sec_per_epoch']:.2f} s/epoch  "
          f"{sec_per_step * 1000:.2f} ms/step  {rec['imgs_per_sec']:.0f} img/s", flush=True)
    return rec


def load_log() -> list[dict]:
    if not LOG.exists():
        return []
    return [json.loads(line) for line in LOG.read_text().splitlines() if line.strip()]


def merge() -> None:
    recs = load_log()
    single = defaultdict(list)
    for r in recs:
        if r["procs"] == 1 and r.get("compile", "none") == "none":
            single[(r["model"], r["k"])].append(r)
    timing: dict = {}
    for (model, k), rs in sorted(single.items()):
        key = "sec_per_epoch" if model == "resnet18k" else "sec_per_step"
        timing.setdefault(model, {}).setdefault(key, {})[str(k)] = statistics.median(x[key] for x in rs)
    ratios = defaultdict(list)                       # per-process slow-down under sharing
    for r in recs:
        base = single.get((r["model"], r["k"]))
        if r["procs"] > 1 and base:
            ratios[r["procs"]].append(r["sec_per_step"] / statistics.median(x["sec_per_step"] for x in base))
    timing["slowdown"] = {"1": 1.0, **{str(p): round(statistics.median(v), 3) for p, v in sorted(ratios.items())}}
    for p, v in sorted(ratios.items()):
        print(f"{p} processes per slice: per-process slow-down x{statistics.median(v):.2f} "
              f"-> slice throughput x{p / statistics.median(v):.2f}")
    TIMING_JSON.write_text(json.dumps(timing, indent=2))
    print(f"wrote {TIMING_JSON}:\n{json.dumps(timing, indent=2)}")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    for name in ("single", "concurrent"):
        p = sub.add_parser(name)
        p.add_argument("--model", required=True, choices=["resnet18k", "mcnn"])
        p.add_argument("--k", nargs="+", type=int, required=True)
        p.add_argument("--epochs", type=int, default=3, help="timed epochs (ResNet)")
        p.add_argument("--steps", type=int, default=2000, help="timed steps (CNN)")
        p.add_argument("--compile", default="none", choices=["none", "default", "reduce-overhead"])
        if name == "single":
            p.add_argument("--procs-tag", type=int, default=1, help=argparse.SUPPRESS)
        else:
            p.add_argument("--procs", nargs="+", type=int, default=[2, 3])
    sub.add_parser("merge")
    args = ap.parse_args()

    if args.cmd == "single":
        for k in args.k:
            measure(args.model, k, args.epochs, args.steps, args.procs_tag, args.compile)
    elif args.cmd == "concurrent":
        for k in args.k:
            for n in args.procs:
                cmd = [sys.executable, __file__, "single", "--model", args.model, "--k", str(k),
                       "--epochs", str(args.epochs), "--steps", str(args.steps),
                       "--compile", args.compile, "--procs-tag", str(n)]
                print(f"--- {n} concurrent processes, {args.model} k={k}")
                procs = [subprocess.Popen(cmd) for _ in range(n)]
                for p in procs:
                    p.wait()
    else:
        merge()


if __name__ == "__main__":
    main()
