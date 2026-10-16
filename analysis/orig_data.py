"""The authors' raw results, converted to the same long format as ours.

Where the data is
    The authors' README links every experiment to a folder of the public Google Cloud bucket
    gs://hml-public/dd/<folder>/. Download with   bash scripts/download_orig_data.sh
    into orig_data/<folder>/ (or $DD_ORIG_DIR/<folder>/).

Format (from the authors' notebooks intro_resnet_plot.ipynb and intro_ocean_plot.ipynb)
    <folder>/Mlist   pickled list of dicts M with numpy arrays (keys 'Test Error', 'Train Error', ...)
                     M['Test Error'][s, j]  = test error of width ks[s] at logged point j
                     M['Train Error'][s, j] = training error on the (noisy) training labels
    <folder>/ks      pickled list of the widths

What the Mlist index means (FOLDERS below; analysis/orig_folders.json overrides it)
    cifar10-resnet18k-p15-adam-reps   5 repetitions, 15 % noise       documented in the authors' notebook
    cifar10-resnet18k-50k-adam        0 %, 10 %, 20 % noise           documented in the authors' notebook
    5-layer-CNN folders               not documented; identified from the data with
                                          python -m analysis.orig_data identify [--write]

Test-error scale
    The ResNet folders store CLEAN test error (the Fig. 1 notebook converts it to noisy error itself).
    A folder marked "test": "noisy" stores the error against NOISY test labels; folder_final and
    folder_curves convert it back with p + e(1 - pK/(K-1)) -> e and keep the stored value as test_err_raw.

identify
    1. Test-error scale: at the smallest, underfitting widths a noisy test set gives train ~= test error,
       a clean one gives train - test ~= p(1 - K*test/(K-1)) (about +0.05 at p = 0.2).
    2. Entry -> setting: the final training-error curve of every entry is compared with those of OUR
       finished runs (mean |difference| over the widths both have). Training error on the noisy training
       labels does not depend on the test-error scale and shifts strongly with the training-set size and
       the noise level, because both move the interpolation threshold.
    3. A suggested FOLDERS entry per folder; --write saves the clear-cut ones to analysis/orig_folders.json
       (commit it: it records how the folders were identified).

Note: the notebook that draws Fig. 1 converts clean to noisy test error with
1 - (1-p)(1-e) + e*p/9, whose last term has the wrong sign; see analysis.metrics.

Fallback without raw data: digitise a figure from the paper (WebPlotDigitizer) and store the points as
analysis/orig_digitized/<figure>.csv with the columns listed in FIGURES; it is used automatically.

Security: the raw files are pickles, which can execute code when loaded. Only load the authors' files.
"""
from __future__ import annotations

import argparse
import functools
import json
import os
import pickle
import sys
from pathlib import Path

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np
import pandas as pd

from analysis.metrics import clean_from_noisy, find_peak, interpolation_threshold
from src.config import ROOT

ORIG_DIR = Path(os.environ.get("DD_ORIG_DIR", ROOT / "orig_data"))
DIGITIZED_DIR = ROOT / "analysis" / "orig_digitized"
OVERRIDES_FILE = ROOT / "analysis" / "orig_folders.json"

# Meaning of the Mlist index for every folder listed in the authors' README.
#   index = "rep":   entries are repetitions of one setting (noise / aug / n fixed below)
#   index = "noise": entry i has label noise values[i]
#   index = "n":     entry i was trained on values[i] samples (None = still unknown -> identify)
#   test  = "clean" | "noisy" | None: scale of the stored test error (None = unknown -> identify)
FOLDERS = {
    # ResNet18: documented in the authors' notebooks
    "cifar10-resnet18k-p15-adam-reps": {"index": "rep", "noise": 0.15, "aug": 1, "n": 50000, "test": "clean",
                                        "verified": True},
    "cifar10-resnet18k-50k-adam": {"index": "noise", "values": [0.0, 0.10, 0.20], "aug": 1, "n": 50000,
                                   "test": "clean", "verified": True},
    # 5-layer CNN, no augmentation. README: "0, 10, 20 % label noise and no data-aug". `identify`: training-error
    # curves closest to ours at 0 / 10 / 20 % noise (thresholds 8 / 10 / 11 vs ours 8 / 10 / 12); clean test set
    # (20 % entry at k=1: train - test = +0.065, clean predicts +0.073; the 10 % entry failed at k=1).
    "cifar10-mcnn-noaug-sgd": {"index": "noise", "values": [0.0, 0.10, 0.20], "aug": 0, "n": 50000,
                               "test": "clean", "verified": True},
    # 5-layer CNN with augmentation, 20 % noise, "varying data size" (README, Sample DD), 5 repetitions.
    # `identify`: test error against NOISY test labels (all 15 entries: train - test = -0.012..+0.012 at k=1-3,
    # a clean test set would give about +0.08); entries = n 12500 / 25000 / 50000 (training-error curves
    # closest to our runs, in every repetition; thresholds 19-21 / 29-30 / 42-46 vs ours 20 / 32 / 48). After
    # conversion the error at k=64 is 0.294 / 0.272 / 0.261 (mean of 5) vs ours 0.308 / 0.277 / 0.261.
    # rep2-rep5 log 975 points instead of 1952 (logged every 512 steps, or trained for fewer steps).
    "cifar10-mcnn-p20-sgd": {"index": "n", "values": [12500, 25000, 50000], "noise": 0.20, "aug": 1, "rep": 1,
                             "test": "noisy", "verified": True},
    "cifar10-mcnn-p20-sgd-rep2": {"index": "n", "values": [12500, 25000, 50000], "noise": 0.20, "aug": 1, "rep": 2,
                                  "test": "noisy", "verified": True},
    "cifar10-mcnn-p20-sgd-rep3": {"index": "n", "values": [12500, 25000, 50000], "noise": 0.20, "aug": 1, "rep": 3,
                                  "test": "noisy", "verified": True},
    "cifar10-mcnn-p20-sgd-rep4": {"index": "n", "values": [12500, 25000, 50000], "noise": 0.20, "aug": 1, "rep": 4,
                                  "test": "noisy", "verified": True},
    "cifar10-mcnn-p20-sgd-rep5": {"index": "n", "values": [12500, 25000, 50000], "noise": 0.20, "aug": 1, "rep": 5,
                                  "test": "noisy", "verified": True},
    # 5-layer CNN with augmentation, 10 % noise (README lists it under Model DD and Sample DD). `identify`: noisy
    # test labels (train - test = -0.014..+0.005 at k=1-3, clean would give +0.047); entry 2 = n 50000 (closest to
    # our 10 % run: curve distance 0.013, thr10% 24 = ours, error at k=64 0.187 vs ours 0.183); entries 0 / 1 =
    # 12500 / 25000 because the thresholds 17 : 25 : 36 have the same ratios as in the 20 % folder (19 : 29 : 42).
    # We have no 10 %-noise runs at other sizes, so `identify` cannot map entries 0 and 1 by itself.
    "cifar10-mcnn-p10-sgd": {"index": "n", "values": [12500, 25000, 50000], "noise": 0.10, "aug": 1, "rep": 1,
                             "test": "noisy", "verified": True},
    # 5-layer CNN on CIFAR-100 without noise or augmentation (Fig. 7 / 20). The README row says "0, 10, 20 %
    # label noise and data-aug" (copied text); `inspect`: 5 nearly identical curves (0.421-0.435 at the widest
    # width) = repetitions, as the folder name says.
    "pct-cifar100-mcnn-p0-sgd-noaug-reps": {"index": "rep", "noise": 0.0, "aug": 0, "n": 50000, "test": "clean",
                                            "verified": True},
    # ResNet18 on CIFAR-100 (not used by our figures, listed for completeness)
    "cifar100-resnet18k-50k-adam": {"index": "noise", "values": [0.0, 0.10, 0.20], "aug": 1, "n": 50000,
                                    "test": "clean", "verified": False},
}
_FOLDER_KEYS = ("index", "values", "noise", "aug", "n", "rep", "test", "verified")

# Columns that load_original(figure) returns (also the columns expected in orig_digitized/<figure>.csv).
FIGURES = {
    "fig1": "k, test_err, test_err_std, train_err   (15 % noise, final epoch, mean over 5 repetitions; a"
            " digitised Fig. 1 gives test_err_noisy instead of test_err)",
    "fig2": "k, epoch, test_err, train_err            (15 % noise, mean over repetitions)",
    "fig4b": "k, noise, test_err, train_err           (final epoch, one row per noise level and width)",
    "fig9": "k, epoch, test_err, train_err            (20 % noise)",
    "fig5": "k, noise, aug, test_err, train_err       (5-layer CNN, n = 50000)",
    "fig11a": "k, n, test_err, test_err_std, train_err (5-layer CNN, 20 % noise, mean over repetitions)",
    "fig7": "k, test_err, test_err_std, train_err     (5-layer CNN, CIFAR-100, no noise, no augmentation)",
    "fig21": "k, wd, test_err   (no raw data published - digitise if needed)",
    "fig28": "k, kind, test_err (kind in {single, ensemble}; no raw data published - digitise if needed)",
}


# ----------------------------------------------------------------------------- folder settings
def _load_overrides() -> dict:
    if not OVERRIDES_FILE.exists():
        return {}
    try:
        return json.loads(OVERRIDES_FILE.read_text())
    except json.JSONDecodeError as e:
        print(f"  [orig_data] {OVERRIDES_FILE.name} is not valid JSON ({e}) - ignored")
        return {}


def _apply_overrides() -> None:
    """Settings found by `identify --write` (analysis/orig_folders.json) replace the defaults above."""
    for folder, upd in _load_overrides().get("folders", {}).items():
        FOLDERS.setdefault(folder, {"index": "rep", "verified": False}).update(
            {k: v for k, v in upd.items() if k in _FOLDER_KEYS})


_apply_overrides()


def num_classes(folder: str) -> int:
    return 100 if "cifar100" in folder else 10


def _entry_noise(meta: dict, i: int) -> float:
    """Label noise of entry i (NaN if unknown)."""
    if meta.get("index") == "noise":
        values = meta.get("values")
        return float(values[i]) if values is not None and i < len(values) else np.nan
    return float(meta.get("noise", np.nan))


def _to_clean(folder: str, meta: dict, i: int, te: np.ndarray) -> np.ndarray:
    """Stored test error -> clean test error (unchanged unless the folder stores noisy test error)."""
    if meta.get("test") != "noisy":
        return te
    return clean_from_noisy(te, _entry_noise(meta, i), num_classes(folder))


# ----------------------------------------------------------------------------- raw folders
@functools.lru_cache(maxsize=None)
def load_folder(folder: str):
    """(Ms, ks) of one downloaded folder, or None if it has not been downloaded."""
    d = ORIG_DIR / folder
    if not (d / "Mlist").exists() or not (d / "ks").exists():
        return None
    with open(d / "Mlist", "rb") as f:
        Ms = pickle.load(f)
    with open(d / "ks", "rb") as f:
        ks = [int(k) for k in pickle.load(f)]
    return Ms, ks


def _last_finite(a: np.ndarray) -> np.ndarray:
    """Per row (width), the last non-NaN value along the epoch axis (robust to unfinished rows)."""
    a = np.asarray(a, dtype=float)
    out = np.full(a.shape[0], np.nan)
    for s in range(a.shape[0]):
        ok = np.flatnonzero(np.isfinite(a[s]))
        if len(ok):
            out[s] = a[s, ok[-1]]
    return out


def _failed(folder: str, te_raw: np.ndarray, tr: np.ndarray) -> np.ndarray:
    """Widths whose training failed: final train AND test error at chance level (K-1)/K, e.g. the k=1 run of
    the 10 %-noise no-augmentation sweep (0.900 / 0.900). These points are dropped (NaN) from every curve."""
    chance = (num_classes(folder) - 1) / num_classes(folder)
    return (np.asarray(tr) >= 0.98 * chance) & (np.asarray(te_raw) >= 0.98 * chance)


def _entry_label(meta: dict, i: int) -> dict:
    """Setting of entry i of a folder according to FOLDERS."""
    lab = {k: meta[k] for k in ("noise", "aug", "n") if k in meta}
    if meta["index"] == "rep":
        lab["rep"] = i
    else:
        values = meta.get("values")
        lab[meta["index"]] = values[i] if values is not None and i < len(values) else np.nan
        lab["rep"] = meta.get("rep", 0)
    return lab


def folder_final(folder: str) -> pd.DataFrame | None:
    """Final errors per (entry, width): columns idx, k, test_err (clean), test_err_raw, train_err + setting."""
    loaded = load_folder(folder)
    if loaded is None:
        return None
    Ms, ks = loaded
    meta = FOLDERS.get(folder, {"index": "rep"})
    rows = []
    for i, M in enumerate(Ms):
        raw, tr = _last_finite(M["Test Error"]), _last_finite(M["Train Error"])
        te = _to_clean(folder, meta, i, raw)
        bad = _failed(folder, raw, tr)
        for s, k in enumerate(ks):
            rows.append({"idx": i, "k": k, "test_err": np.nan if bad[s] else te[s], "test_err_raw": raw[s],
                         "train_err": np.nan if bad[s] else tr[s], "failed": bool(bad[s]), **_entry_label(meta, i)})
    return pd.DataFrame(rows)


def folder_curves(folder: str, idx: int | None = None) -> pd.DataFrame | None:
    """Per-epoch curves: columns idx, k, epoch (1-based), test_err (clean), train_err (entry idx, or all)."""
    loaded = load_folder(folder)
    if loaded is None:
        return None
    Ms, ks = loaded
    meta = FOLDERS.get(folder, {"index": "rep"})
    frames = []
    for i, M in enumerate(Ms):
        if idx is not None and i != idx:
            continue
        te = _to_clean(folder, meta, i, np.array(M["Test Error"], dtype=float))   # a copy: the cache stays intact
        tr = np.asarray(M["Train Error"], float)
        bad = _failed(folder, _last_finite(M["Test Error"]), _last_finite(M["Train Error"]))
        te[bad, :] = np.nan                                      # failed runs: whole curve dropped below
        n_ep = te.shape[1]
        frames.append(pd.DataFrame({"idx": i, "k": np.repeat(ks, n_ep), "epoch": np.tile(np.arange(1, n_ep + 1), len(ks)),
                                    "test_err": te.reshape(-1), "train_err": tr.reshape(-1)}))
    return pd.concat(frames, ignore_index=True).dropna(subset=["test_err"]) if frames else None


def _mean_over_reps(df: pd.DataFrame, by: list[str]) -> pd.DataFrame:
    return (df.groupby(by, as_index=False)
              .agg(test_err=("test_err", "mean"), test_err_std=("test_err", "std"), train_err=("train_err", "mean"),
                   reps=("test_err", "count")))


def _warn_unverified(*folders: str) -> None:
    bad = [f for f in folders if not FOLDERS.get(f, {}).get("verified", False)]
    if bad:
        print(f"  [orig_data] setting of {', '.join(bad)} is NOT verified - "
              f"check with `python -m analysis.orig_data identify`")


# ----------------------------------------------------------------------------- per figure
def _fig1():
    f = folder_final("cifar10-resnet18k-p15-adam-reps")
    return None if f is None else _mean_over_reps(f, ["k"]).assign(noise=0.15)


def _fig2():
    c = folder_curves("cifar10-resnet18k-p15-adam-reps")
    return None if c is None else c.groupby(["k", "epoch"], as_index=False)[["test_err", "train_err"]].mean()


def _fig4b():
    parts = []
    f = folder_final("cifar10-resnet18k-50k-adam")
    if f is not None:
        parts.append(f[["k", "noise", "test_err", "train_err"]])
    r = _fig1()
    if r is not None:                                        # the 15 % curve comes from the repetitions
        parts.append(r[["k", "noise", "test_err", "train_err"]])
    return pd.concat(parts, ignore_index=True) if parts else None


def _fig9():
    meta = FOLDERS["cifar10-resnet18k-50k-adam"]
    c = folder_curves("cifar10-resnet18k-50k-adam", idx=meta["values"].index(0.20))
    return None if c is None else c.drop(columns="idx").assign(noise=0.20)


def _cnn_sizes(prefix: str = "cifar10-mcnn-p20-sgd"):
    """Final errors of all downloaded repetitions of the varying-n CNN sweep (needs n values in FOLDERS)."""
    folders = [f for f in FOLDERS if f == prefix or f.startswith(prefix + "-rep")]
    parts = [folder_final(f) for f in folders]
    parts = [p for p in parts if p is not None]
    if not parts:
        return None
    df = pd.concat(parts, ignore_index=True)
    if df["n"].isna().all() or df["test_err"].isna().all():
        print(f"  [orig_data] training-set sizes or test-error scale of {prefix}* unknown: run "
              f"`python -m analysis.orig_data identify`")
        return None
    _warn_unverified(*folders)
    return df.dropna(subset=["n", "test_err"])


def _fig5():
    parts = []
    f = folder_final("cifar10-mcnn-noaug-sgd")
    if f is not None:
        _warn_unverified("cifar10-mcnn-noaug-sgd")
        parts.append(f[["k", "noise", "aug", "test_err", "train_err"]])
    for prefix in ("cifar10-mcnn-p20-sgd", "cifar10-mcnn-p10-sgd"):
        d = _cnn_sizes(prefix)
        if d is not None and (d["n"] == 50000).any():        # with augmentation: full-data entries
            d = d[d["n"] == 50000]
            parts.append(d.groupby(["k", "noise", "aug"], as_index=False)[["test_err", "train_err"]].mean())
    return pd.concat(parts, ignore_index=True) if parts else None


def _fig11a():
    d = _cnn_sizes("cifar10-mcnn-p20-sgd")
    return None if d is None else _mean_over_reps(d, ["n", "k"])


def _fig7():
    f = folder_final("pct-cifar100-mcnn-p0-sgd-noaug-reps")
    if f is None:
        return None
    _warn_unverified("pct-cifar100-mcnn-p0-sgd-noaug-reps")
    return _mean_over_reps(f, ["k"])


BUILDERS = {"fig1": _fig1, "fig2": _fig2, "fig4b": _fig4b, "fig9": _fig9, "fig5": _fig5,
            "fig11a": _fig11a, "fig7": _fig7}


def _from_raw(figure: str) -> pd.DataFrame | None:
    builder = BUILDERS.get(figure)
    return builder() if builder else None


@functools.lru_cache(maxsize=None)
def _load_original_cached(figure: str):
    df = _from_raw(figure)
    if df is not None:
        return df
    csv_path = DIGITIZED_DIR / f"{figure}.csv"
    return pd.read_csv(csv_path) if csv_path.exists() else None


def load_original(figure: str) -> pd.DataFrame | None:
    """The authors' data for one figure (raw data first, then a digitised CSV), or None."""
    df = _load_original_cached(figure)
    return None if df is None else df.copy()


# ----------------------------------------------------------------------------- identify
def _ranges(ks) -> str:
    """[1, 2, 3, 5, 6, 8] -> '1-3, 5-6, 8'."""
    ks = sorted(int(k) for k in ks)
    parts, start, prev = [], ks[0], ks[0]
    for k in ks[1:] + [None]:
        if k is not None and k == prev + 1:
            prev = k
            continue
        parts.append(f"{start}-{prev}" if prev > start else str(start))
        if k is not None:
            start = prev = k
    return ", ".join(parts)


def _test_scale(ks, te, tr, p: float, K: int) -> dict:
    """Clean or noisy stored test error? Judged at the smallest width that trained (test error below chance)
    and still underfits: there the generalization gap is smallest. It grows with width, above all without
    augmentation, and a gap g lowers train - test by about g(1 - pK/(K-1)), which looks like a noisy test set."""
    if not (p > 0):
        return {"verdict": "n/a"}
    chance = (K - 1) / K
    for s in np.argsort(ks):
        if tr[s] > 0.25 and 0.05 < te[s] < 0.8 * chance:
            d, pred = float(tr[s] - te[s]), float(p * (1 - K * te[s] / (K - 1)))
            return {"verdict": "noisy" if abs(d) < abs(d - pred) else "clean", "k": int(ks[s]),
                    "train": float(tr[s]), "test": float(te[s]), "d": d, "pred_clean": pred}
    return {"verdict": "unknown"}


def _ours_final() -> pd.DataFrame | None:
    """Seed-averaged final errors of our finished 5-layer-CNN runs without weight decay."""
    try:
        from analysis.load_results import curves, final
        c = curves(["R8", "R9", "R10", "R12", "R13"])
    except FileNotFoundError:
        return None
    if c.empty:
        return None
    f = final(c)
    f = f[(f["model"] == "mcnn") & (f["wd"] == 0)]
    if f.empty:
        return None
    return (f.groupby(["dataset", "noise", "aug", "n", "k"], as_index=False)
             .agg(train_err=("train_err", "mean"), test_err=("test_err", "mean"), seeds=("test_err", "count")))


def _candidates(ours, dataset: str, aug, index: str, meta: dict):
    """Our curves to compare an entry with, keyed by what the folder's index stands for; and a note."""
    if ours is None:
        return {}, "no finished runs of ours found (results/ and runs.csv needed)"
    o = ours[(ours["dataset"] == dataset) & (ours["aug"] == aug)]
    if index == "noise":
        o = o[o["n"] == 50000]
        return {float(p): g for p, g in o.groupby("noise")}, "our runs at n = 50000, one per noise level"
    if index == "n":
        same = o[np.isclose(o["noise"], meta.get("noise", np.nan))]
        if same["n"].nunique() >= 2:
            return {int(n): g for n, g in same.groupby("n")}, f"our runs at {meta['noise']:.0%} noise, one per n"
        return ({(float(p), int(n)): g for (p, n), g in o.groupby(["noise", "n"])},
                "our runs at ALL noise levels (fewer than 2 sizes at this noise); keys are (noise, n)")
    same = o[np.isclose(o["noise"], meta.get("noise", np.nan)) & (o["n"] == meta.get("n", 50000))]
    return ({"ours": same} if len(same) else {}), "our runs with the same setting"


def _curve_distance(ks, tr, ref: pd.DataFrame) -> float:
    """Mean |train error difference| over the widths both curves have (NaN if fewer than 4)."""
    m = dict(zip(ref["k"].astype(int), ref["train_err"]))
    pairs = [(tr[s], m[k]) for s, k in enumerate(ks) if k in m and np.isfinite(tr[s])]
    return float(np.mean([abs(a - b) for a, b in pairs])) if len(pairs) >= 4 else np.nan


def _key_str(key) -> str:
    if isinstance(key, tuple):
        return f"({key[0]:.0%}, n={key[1]})"
    if isinstance(key, float):
        return f"{key:.0%}"
    return f"n={key}" if isinstance(key, int) else str(key)


def _summary(ks, te, tr) -> dict:
    """Threshold, peak (widths <= 64) and error at the widest width <= 64 of one clean curve."""
    ks = np.asarray(ks)
    within = ks <= 64
    pk = find_peak(ks[within], te[within])
    k64 = int(ks[within].max())
    return {"thr1": interpolation_threshold(ks, tr, 0.01), "thr10": interpolation_threshold(ks, tr, 0.10),
            "peak_k": pk["peak_k"], "rise": pk["prominence"], "k64": k64, "err64": float(te[list(ks).index(k64)])}


def _identify_folder(folder: str, ours) -> dict | None:
    loaded = load_folder(folder)
    if loaded is None:
        print(f"\n== {folder}: not downloaded")
        return None
    Ms, ks = loaded
    meta = FOLDERS.get(folder, {"index": "rep"})
    K, index = num_classes(folder), meta.get("index", "rep")
    dataset = "cifar100" if K == 100 else "cifar10"
    logged = sorted({np.asarray(M["Test Error"]).shape[1] for M in Ms})
    print(f"\n== {folder}: {len(Ms)} entries | widths {_ranges(ks)} ({len(ks)}) | logged points {logged}")
    finals = [(_last_finite(M["Test Error"]), _last_finite(M["Train Error"])) for M in Ms]

    # 1. scale of the stored test error
    scales = [_test_scale(ks, te, tr, _entry_noise(meta, i), K) for i, (te, tr) in enumerate(finals)]
    judged = [s for s in scales if s["verdict"] in ("clean", "noisy")]
    kmin = min((s["k"] for s in judged), default=None)          # decide with the smallest widths only
    deciding = {s["verdict"] for s in judged if s["k"] == kmin}
    scale = (deciding.pop() if len(deciding) == 1 else "mixed") if judged else (meta.get("test") or "clean")
    print(f"   stored test error: {scale.upper()}" + ("" if judged else " (no label noise or no underfitting widths)"))
    for i, s in enumerate(scales):
        if "d" in s:
            print(f"     [{i}] k={s['k']}: train {s['train']:.3f}, test {s['test']:.3f}, train - test = {s['d']:+.3f}"
                  f"   (clean test set would give {s['pred_clean']:+.3f}, noisy ~0) -> {s['verdict']}"
                  + ("" if s["k"] == kmin else f"   [wider than k={kmin}: not used]"))

    # 2. entries vs our runs (training-error curves) + clean-scale summary
    cands, note = _candidates(ours, dataset, meta.get("aug", 1), index, meta)
    print(f"   compared with: {note}")
    print("   (thresholds from training error; peak and error on the clean test-error scale; rise = peak minus\n"
          "    the lowest error at smaller widths, small values are noise)")
    print(f"   {'entry':5} {'thr1%':>5} {'thr10%':>6} {'peak k':>6} {'rise':>6} {'clean err@k':>12}   "
          f"mean |train-error difference| to our runs (best first)")
    best, clear = [], True
    for i, (te_raw, tr) in enumerate(finals):
        te = clean_from_noisy(te_raw, _entry_noise(meta, i), K) if scale == "noisy" else te_raw
        sm = _summary(ks, te, tr)
        scores = sorted(((_curve_distance(ks, tr, g), key) for key, g in cands.items()),
                        key=lambda x: (np.isnan(x[0]), x[0]))
        txt = " | ".join(f"{_key_str(k)}: {d:.3f}" for d, k in scores) if scores else "-"
        print(f"   [{i}]   {sm['thr1']:5.0f} {sm['thr10']:6.0f} {sm['peak_k']:6.0f} {sm['rise']:6.3f} "
              f"{sm['err64']:7.3f}@{sm['k64']:<4d}   {txt}")
        if scores and np.isfinite(scores[0][0]):
            best.append(scores[0][1])
            if len(scores) > 1 and np.isfinite(scores[1][0]) and not scores[0][0] < 0.6 * scores[1][0]:
                clear = False
                print(f"         ^ not clear-cut: best {scores[0][0]:.3f} vs second {scores[1][0]:.3f}")
            g = cands[scores[0][1]]
            mine = _summary(g["k"].to_numpy(), g["test_err"].to_numpy(), g["train_err"].to_numpy())
            print(f"         ours {_key_str(scores[0][1])}: thr1% {mine['thr1']:.0f}, thr10% {mine['thr10']:.0f}, "
                  f"peak k {mine['peak_k']:.0f} (rise {mine['rise']:.3f}), err@{mine['k64']} {mine['err64']:.3f}")

    # 3. suggestion
    sugg = {"test": scale if scale in ("clean", "noisy") else None}
    ok = scale != "mixed"
    if index in ("n", "noise"):
        if len(best) == len(finals):
            sugg["values"] = [b for b in best]
            injective = len(set(best)) == len(best)
            same_noise = not any(isinstance(b, tuple) for b in best)
            ok = ok and injective and clear and same_noise
            if not injective:
                print("   ! two entries match the same run of ours - check by eye")
            if not same_noise:
                print("   ! matched against other noise levels - decide by eye, not written by --write")
        else:
            ok = False
            print("   ! no comparable runs of ours - cannot map the entries")
    else:
        tes = np.vstack([te for te, _ in finals])
        cols = np.asarray(ks) >= 4
        spread = float(np.nanmedian(np.nanstd(tes[:, cols], axis=0))) if len(finals) > 1 else np.nan
        print(f"   repetitions: median std across entries (widths >= 4) = {spread:.4f}")
        ok = ok and np.isfinite(spread) and spread < 0.02
    current = {k: meta.get(k) for k in ("values", "test")}
    mappable = not any(isinstance(b, tuple) for b in sugg.get("values") or [])
    if meta.get("verified") and not mappable:
        print(f"   current verified setting kept: {current}")
        sugg = {"test": sugg["test"], "values": current["values"], "_kept": True}
        ok = sugg["test"] in (None, current["test"])
    elif meta.get("verified") and any(k in sugg and sugg[k] is not None and sugg[k] != current[k] for k in current):
        print(f"   ! DIFFERS from the current verified setting {current} - check by eye, not written")
        ok = False
    elif meta.get("verified") and scale != "mixed":
        ok, sugg["_agrees"] = True, True                     # the data confirm the setting already in FOLDERS
    sugg["verified"] = bool(ok)
    sugg["_valid"] = bool(scale != "mixed" and mappable and sugg.get("values") is not None
                          and len(set(map(str, sugg["values"]))) == len(sugg["values"])) if index != "rep" else False
    return sugg


def identify(folders: list[str], write: bool) -> None:
    ours = _ours_final()
    if not folders:
        folders = [f for f in FOLDERS if "mcnn" in f and load_folder(f) is not None]
    suggestions = {f: s for f in folders if (s := _identify_folder(f, ours)) is not None}
    for f, s in suggestions.items():        # a repetition folder agreeing with a clear-cut base folder is OK too
        base = suggestions.get(f.split("-rep")[0]) if "-rep" in f else None
        if (base and base["verified"] and not s["verified"] and s.get("_valid")
                and s.get("values") == base.get("values") and s.get("test") == base.get("test")):
            s["verified"], s["_same_as_base"] = True, True
    print("\nSuggested settings (OK = clear-cut):")
    for f, s in suggestions.items():
        shown = {k: (v if not isinstance(v, list) else [x if not isinstance(x, tuple) else list(x) for x in v])
                 for k, v in s.items() if k in ("test", "values")}
        why = (" (same mapping as the base folder)" if s.get("_same_as_base") else
               " (current setting kept: cannot be mapped automatically)" if s.get("_kept") else
               " (confirms the setting in FOLDERS)" if s.get("_agrees") else "")
        print(f"  {f:38s} {json.dumps(shown)}  {('OK' if s['verified'] else 'needs a look') + why}")
    if not write:
        print("Nothing written. Send this output for a check, or save the OK ones with --write.")
        return
    data = _load_overrides()
    data.setdefault("folders", {})
    data["method"] = ("python -m analysis.orig_data identify --write: test-error scale from train vs test error at "
                      "underfitting widths; entries matched to our runs by their final training-error curves")
    written = []
    for f, s in suggestions.items():
        if s["verified"]:
            data["folders"][f] = {k: v for k, v in s.items() if v is not None and not k.startswith("_")}
            written.append(f)
    OVERRIDES_FILE.write_text(json.dumps(data, indent=2) + "\n")
    print(f"wrote {len(written)} folder(s) to {OVERRIDES_FILE}: {', '.join(written) or '-'}")


# ----------------------------------------------------------------------------- command line
def inspect(folder: str) -> None:
    """Print the structure of a downloaded folder."""
    loaded = load_folder(folder)
    if loaded is None:
        sys.exit(f"{ORIG_DIR / folder}/Mlist or /ks not found - download it first (scripts/download_orig_data.sh)")
    Ms, ks = loaded
    print(f"{folder}: {len(Ms)} entries, widths {_ranges(ks)} ({len(ks)} widths)")
    print(f"current setting in FOLDERS: {FOLDERS.get(folder, 'not listed')}")
    for i, M in enumerate(Ms):
        keys = list(M.keys()) if hasattr(M, "keys") else type(M).__name__
        te = np.asarray(M["Test Error"], float)
        tr = np.asarray(M["Train Error"], float)
        fte, ftr = _last_finite(te), _last_finite(tr)
        print(f"  [{i}] keys={keys}")
        print(f"      'Test Error' shape {te.shape} (widths x logged points); final test err: smallest k "
              f"{fte[0]:.3f}, largest k {fte[-1]:.3f}; final train err: smallest k {ftr[0]:.3f}, "
              f"largest k {ftr[-1]:.3f}")
    print("For the meaning of the entries run: python -m analysis.orig_data identify")


def check() -> None:
    """Which figures can be overlaid with the data that is available now."""
    print(f"raw data directory: {ORIG_DIR}")
    if OVERRIDES_FILE.exists():
        print(f"settings from {OVERRIDES_FILE.name} applied")
    for folder, meta in FOLDERS.items():
        state = "downloaded" if load_folder(folder) is not None else "missing"
        scale = meta.get("test") or "?"
        print(f"  {folder:40s} {state:11s} test {scale:6s} {'verified' if meta['verified'] else 'NOT verified'}")
    print("figures:")
    for fig, cols in FIGURES.items():
        df = load_original(fig)
        print(f"  {fig:7s} {'available (' + str(len(df)) + ' rows)' if df is not None else 'not available':24s} {cols}")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    sub.add_parser("check", help="which folders are downloaded and which figures can be overlaid")
    p = sub.add_parser("inspect", help="structure of one downloaded folder")
    p.add_argument("folder")
    p = sub.add_parser("identify", help="test-error scale and meaning of the entries of the CNN folders")
    p.add_argument("folders", nargs="*", help="default: all downloaded 5-layer-CNN folders")
    p.add_argument("--write", action="store_true", help="save clear-cut settings to analysis/orig_folders.json "
                   "(not needed for the folders already verified in FOLDERS)")
    args = ap.parse_args()
    if args.cmd == "inspect":
        inspect(args.folder)
    elif args.cmd == "identify":
        identify(args.folders, args.write)
    else:
        check()


if __name__ == "__main__":
    main()
