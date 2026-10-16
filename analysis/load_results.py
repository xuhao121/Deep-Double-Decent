"""Load runs.csv and results/<run_id>.csv into pandas DataFrames."""
from __future__ import annotations

from dataclasses import asdict

import numpy as np
import pandas as pd

from src.config import RUNS_CSV, load_runs, out_dir, results_csv

RUN_COLS = ["run_id", "exp", "model", "dataset", "k", "noise", "noise_seed", "init_seed", "subset_seed",
            "n", "aug", "opt", "momentum", "wd", "sched", "budget_unit", "budget"]


def runs_df(runs_csv=RUNS_CSV) -> pd.DataFrame:
    return pd.DataFrame([asdict(r) for r in load_runs(runs_csv)])


def curves(exps=None, runs_csv=RUNS_CSV) -> pd.DataFrame:
    """All evaluation rows of the selected experiment blocks, joined with their run settings."""
    runs = runs_df(runs_csv)
    if exps is not None:
        runs = runs[runs["exp"].isin(exps)]
    frames = []
    for _, r in runs.iterrows():
        path = results_csv(r["run_id"])
        if not path.exists():
            continue
        c = pd.read_csv(path)
        if c.empty:
            continue
        for col in RUN_COLS:
            c[col] = r[col]
        frames.append(c)
    if not frames:
        return pd.DataFrame(columns=RUN_COLS)
    df = pd.concat(frames, ignore_index=True)
    df["tick"] = np.where(df["budget_unit"] == "epochs", df["epoch"], df["step"])
    return df


def final(curves_df: pd.DataFrame, only_complete: bool = True) -> pd.DataFrame:
    """Last evaluation row of every run (only runs that reached their budget, by default)."""
    if curves_df.empty:
        return curves_df
    last = curves_df.sort_values("step").groupby("run_id", as_index=False).tail(1)
    complete = last["tick"] >= last["budget"]
    if only_complete and (~complete).any():
        print(f"  (skipping {int((~complete).sum())} unfinished runs)")
        last = last[complete]
    return last.reset_index(drop=True)


def oracle_es(curves_df: pd.DataFrame, only_complete: bool = True) -> pd.DataFrame:
    """Per run: minimum test error over all evaluations (oracle early stopping) and where it occurs."""
    if curves_df.empty:
        return curves_df
    fin = final(curves_df, only_complete)
    sub = curves_df[curves_df["run_id"].isin(fin["run_id"])]
    idx = sub.groupby("run_id")["test_err"].idxmin()
    es = sub.loc[idx, ["run_id", "test_err", "tick"]].rename(columns={"test_err": "es_test_err", "tick": "es_tick"})
    return fin.merge(es, on="run_id")


def test_preds(run_id: str, which: str = "test"):
    """(ticks, predictions uint8 [n_ticks, N]) stored by src/train.py; rows never written are 255."""
    d = out_dir(run_id)
    mm = np.load(d / f"{which}_preds.npy", mmap_mode="r")
    ticks = np.load(d / f"{which}_preds_ticks.npy")
    return ticks, mm
