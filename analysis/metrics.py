"""Quantities used in the report: interpolation threshold, double-descent peak, ensembles, conversions."""
from __future__ import annotations

import math

import numpy as np
import pandas as pd


def agg_seeds(df: pd.DataFrame, by: list[str], cols=("test_err", "train_err")) -> pd.DataFrame:
    """Mean, standard deviation and count over seeds for every group (e.g. by=['k'])."""
    spec = {}
    for c in cols:
        spec[f"{c}_mean"] = (c, "mean")
        spec[f"{c}_std"] = (c, "std")
    spec["n_seeds"] = (cols[0], "count")
    return df.groupby(by, as_index=False).agg(**spec).sort_values(by).reset_index(drop=True)


def interpolation_threshold(ks, train_err, thr: float = 0.01):
    """Smallest width whose (mean) final training error is below thr; NaN if none.

    thr=0.01 ~ 'reaches ~0 train error'; thr=0.1 is the epsilon the paper uses heuristically for EMC.
    """
    ks, train_err = np.asarray(ks), np.asarray(train_err)
    order = np.argsort(ks)
    for k, e in zip(ks[order], train_err[order]):
        if e < thr:
            return k
    return math.nan


def find_peak(ks, mean, std=None) -> dict:
    """Most prominent interior local maximum of a test-error-vs-width curve.

    prominence = peak value - minimum to its left (the first descent's valley).
    significant = prominence > 2 * max(std at peak, std at valley)  (criterion of the plan, step 6).
    """
    ks = np.asarray(ks)
    m = np.asarray(mean, dtype=float)
    s = np.full_like(m, np.nan) if std is None else np.asarray(std, dtype=float)
    order = np.argsort(ks)
    ks, m, s = ks[order], m[order], s[order]
    best = None
    for i in range(1, len(m) - 1):
        if m[i] > m[i - 1] and m[i] >= m[i + 1]:
            v = int(np.argmin(m[:i]))
            prom = m[i] - m[v]
            if best is None or prom > best["prominence"]:
                noise = np.nanmax([s[i], s[v]]) if not (np.isnan(s[i]) and np.isnan(s[v])) else math.nan
                best = {"peak_k": ks[i], "peak_err": m[i], "peak_std": s[i], "valley_k": ks[v],
                        "valley_err": m[v], "prominence": prom,
                        "significant": (prom > 2 * noise) if not math.isnan(noise) else math.nan,
                        "peak_idx": i}
    if best is None:
        return {"peak_k": math.nan, "peak_err": math.nan, "peak_std": math.nan, "valley_k": math.nan,
                "valley_err": math.nan, "prominence": 0.0, "significant": False, "peak_idx": -1}
    return best


def grid_steps_between(ks, a, b) -> float:
    """Number of grid positions between widths a and b on the sorted grid ks."""
    ks = sorted(ks)
    if any(isinstance(x, float) and math.isnan(x) for x in (a, b)) or a not in ks or b not in ks:
        return math.nan
    return ks.index(a) - ks.index(b)


def curve_summary(ks, train_mean, test_mean, test_std=None) -> dict:
    """Threshold + peak + location check for one model-wise curve (one row of tables T1/T2)."""
    thr1 = interpolation_threshold(ks, train_mean, 0.01)
    thr10 = interpolation_threshold(ks, train_mean, 0.10)
    pk = find_peak(ks, test_mean, test_std)
    out = {"interp_thr_1pct": thr1, "interp_thr_10pct": thr10, **{k: v for k, v in pk.items() if k != "peak_idx"}}
    out["peak_minus_thr_steps"] = grid_steps_between(list(ks), pk["peak_k"], thr1)
    order = np.argsort(np.asarray(ks))
    out["final_err_kmax"] = float(np.asarray(test_mean)[order][-1])
    return out


def epochwise_peak(ticks, err, smooth: int = 5) -> dict:
    """First minimum, following maximum and final value of a test-error-vs-epoch curve.

    A rolling median (window `smooth`) suppresses single-evaluation noise. Epoch-wise double
    descent = the curve rises after its first minimum and ends below that maximum.
    """
    s = pd.Series(np.asarray(err, dtype=float)).rolling(smooth, center=True, min_periods=1).median().to_numpy()
    t = np.asarray(ticks)
    i_min = 0
    for i in range(1, len(s) - 1):
        if s[i] <= s[i - 1] and s[i] < s[i + 1]:
            i_min = i
            break
    i_max = i_min + int(np.argmax(s[i_min:]))
    return {"min1_tick": t[i_min], "min1_err": s[i_min], "peak_tick": t[i_max], "peak_err": s[i_max],
            "final_err": s[-1], "rise": s[i_max] - s[i_min], "second_descent": s[i_max] - s[-1]}


def plurality_vote(preds: np.ndarray, num_classes: int) -> np.ndarray:
    """Majority vote over models (rows of preds, ordered by seed). Ties: the prediction of the
    lowest-seed model among the tied classes (state this rule in the report)."""
    m, n = preds.shape
    cols = np.arange(n)
    counts = np.zeros((num_classes, n), dtype=np.int32)
    for row in preds:
        counts[row, cols] += 1
    maxc = counts.max(0)
    out = np.full(n, -1, dtype=np.int64)
    for row in preds:
        sel = (out == -1) & (counts[row, cols] == maxc)
        out[sel] = row[sel]
    return out


def noisy_from_clean(err_clean, p: float, num_classes: int = 10):
    """Expected test error on the noisy distribution from the clean test error e.

    A label is replaced with probability p by a uniformly random wrong class, independently of the
    prediction, so P(prediction = noisy label) = (1-p)(1-e) + p*e/(K-1) and the noisy error is
    p + e * (1 - p*K/(K-1)). (Checked by Monte-Carlo simulation.)
    """
    return p + (1 - p * num_classes / (num_classes - 1)) * np.asarray(err_clean)


def clean_from_noisy(err_noisy, p: float, num_classes: int = 10):
    return (np.asarray(err_noisy) - p) / (1 - p * num_classes / (num_classes - 1))


def noisy_from_clean_authors(err_clean, p: float, num_classes: int = 10):
    """The conversion used in the authors' notebook intro_resnet_plot.ipynb (which draws Fig. 1):
    1 - (1-p)(1-e) + e*p/(K-1) = p + e * (1 - p + p/(K-1)).

    The sign of the e*p/(K-1) term is flipped compared with noisy_from_clean, so the published noisy
    test error is too high by 2*p*e/(K-1) (about 1 percentage point at p = 0.15, e = 0.3).
    """
    return p + (1 - p + p / (num_classes - 1)) * np.asarray(err_clean)


def clean_from_noisy_authors(err_noisy, p: float, num_classes: int = 10):
    """Inverse of noisy_from_clean_authors: recovers the authors' clean error from a value read off Fig. 1."""
    return (np.asarray(err_noisy) - p) / (1 - p + p / (num_classes - 1))


def no_help_widths(g_small: pd.DataFrame, g_large: pd.DataFrame) -> list:
    """Widths where more data does NOT reduce test error: mean_small - mean_large < std_small + std_large."""
    mg = g_small.merge(g_large, on="k", suffixes=("_s", "_l"))
    sd = mg["test_err_std_s"].fillna(0) + mg["test_err_std_l"].fillna(0)
    return mg.loc[(mg["test_err_mean_s"] - mg["test_err_mean_l"]) < sd, "k"].tolist()
