"""Figures and tables for the report.

    python -m analysis.make_figures all              # everything that has finished runs
    python -m analysis.make_figures resnet           # A1-A7 + T1 (member 1)
    python -m analysis.make_figures cnn              # B1-B7 + T2 (member 2)
    python -m analysis.make_figures a4 --small 4 --medium 12 --large 64
    python -m analysis.make_figures b3 --b3-widths 3 12 64

Outputs: figures/<name>.pdf + .png and tables/<name>.csv + .md. Unfinished runs are skipped in
final-value plots (heat maps and epoch curves use whatever has been logged so far).
The analysis needs only numpy, pandas and matplotlib (it never imports PyTorch, which avoids
OpenMP clashes such as "OMP: Error #15" on Windows/conda). Test error is always the 0-1 error on
the clean test set. The authors' raw data (analysis/orig_data.py)
also stores clean test error; a value digitised from the published Fig. 1 (noisy test error) is converted
back with the authors' own formula (analysis.metrics.clean_from_noisy_authors).
"""
from __future__ import annotations

import argparse
import sys
import traceback
from pathlib import Path

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np
import pandas as pd
from matplotlib.lines import Line2D
from matplotlib.patches import Patch

from analysis.load_results import curves, final, oracle_es, test_preds
from analysis.metrics import (agg_seeds, clean_from_noisy_authors, curve_summary, epochwise_peak, find_peak,
                              grid_steps_between, interpolation_threshold, no_help_widths, plurality_vote)
from analysis.orig_data import load_original
from analysis.plot_style import COL_W, FULL_W, plt, save, save_table
from src.config import DATA_DIR    # analysis deliberately does not import PyTorch

NOISES = [0.0, 0.10, 0.15, 0.20]
SUBSETS = [6250, 12500, 25000, 50000]
COLORS = plt.rcParams["axes.prop_cycle"].by_key()["color"]


# ----------------------------------------------------------------------------- helpers
def _require(df, name: str) -> bool:
    if df is None or len(df) == 0:
        print(f"{name}: no finished runs yet - skipped")
        return False
    return True


def _plot_mean_std(ax, g, col, label, color=None, marker="o", ls="-"):
    """Mean over seeds as a line, +- one standard deviation as a band (when >1 seed)."""
    line, = ax.plot(g["k"], g[f"{col}_mean"], marker=marker, ls=ls, label=label, color=color)
    if (g["n_seeds"] > 1).any():
        s = g[f"{col}_std"].fillna(0)
        ax.fill_between(g["k"], g[f"{col}_mean"] - s, g[f"{col}_mean"] + s, alpha=0.2, lw=0, color=line.get_color())


def _overlay_orig(ax, figure, query=None, col="test_err", label="paper", noisy_p=None, x="k",
                  color="0.45", ls="--") -> bool:
    """Curve from the authors' data (analysis/orig_data.py), if available; returns True if drawn."""
    o = load_original(figure)
    if o is None:
        return False
    if query:
        o = o.query(query)
    if noisy_p is not None and "test_err_noisy" in o and "test_err" not in o:
        # digitised from the published Fig. 1, which used the authors' noisy-error formula
        o = o.assign(test_err=clean_from_noisy_authors(o["test_err_noisy"], noisy_p))
    if col not in o or o.empty:
        return False
    o = o.dropna(subset=[x, col]).sort_values(x)    # NaN = a run of the authors that failed (orig_data._failed)
    if x == "k":                              # only the width range we trained (our curves are drawn first)
        xmax = ax.dataLim.x1
        if np.isfinite(xmax) and xmax >= 1:
            o = o[o["k"] <= xmax * 1.0001]
        if o.empty:
            return False
    ax.plot(o[x], o[col], ls=ls, color=color, lw=1.0, alpha=0.85, label=label)
    return True


def _paper_legend(ax, drew: bool, **kw) -> None:
    """Legend with one extra entry explaining the dashed paper curves (if any were drawn)."""
    handles, _ = ax.get_legend_handles_labels()
    if drew:
        handles = handles + [Line2D([], [], color="0.3", ls="--", lw=1.0, label="paper (dashed)")]
    ax.legend(handles=handles, **kw)


def _paper_cols(o, grid) -> dict:
    """A paper curve (columns k, test_err[, train_err, test_err_std]) summarised like ours.

    Threshold and peak are taken at OUR widths `grid` (same grid -> comparable positions); the paper's own,
    finer grid up to our widest width gives the *_dense columns (3-point running median over widths before the
    peak search); final error = the paper's widest width <= ours.
    """
    if o is None or len(o) == 0:
        return {}
    grid = sorted(int(k) for k in grid)
    o = o[o["k"] <= grid[-1]].dropna(subset=["test_err"]).sort_values("k")   # NaN = failed run of the authors
    if len(o) < 3:
        return {}

    def summary(d):
        if len(d) < 3:
            return {"interp_thr_1pct": np.nan, "peak_k": np.nan, "peak_err": np.nan}
        tr = d["train_err"] if "train_err" in d else pd.Series(np.nan, index=d.index)
        std = d["test_err_std"].to_numpy() if "test_err_std" in d else None
        return curve_summary(d["k"].to_numpy(), tr.to_numpy(), d["test_err"].to_numpy(), std)

    # the paper's dense grid (every width) from single runs wiggles: 3-point running median before the peak search
    smooth = o.assign(test_err=o["test_err"].rolling(3, center=True, min_periods=1).median())
    on_grid, dense = summary(o[o["k"].isin(grid)]), summary(smooth)
    widest = o.iloc[-1]
    return {"paper_interp_thr_1pct": on_grid["interp_thr_1pct"], "paper_peak_k": on_grid["peak_k"],
            "paper_peak_err": on_grid["peak_err"], "paper_final_err_kmax": float(widest["test_err"]),
            "paper_final_k": int(widest["k"]), "paper_interp_thr_1pct_dense": dense["interp_thr_1pct"],
            "paper_peak_k_dense": dense["peak_k"]}


def _with_deltas(row: dict, grid=None) -> dict:
    """Add ours-minus-paper differences (the quantitative criterion of the plan, step 6)."""
    if "paper_peak_err" in row:
        row["d_peak_k"] = row["peak_k"] - row["paper_peak_k"]
        if grid is not None:                  # in steps of our width grid
            row["d_peak_steps"] = grid_steps_between(sorted(int(k) for k in grid), row["peak_k"], row["paper_peak_k"])
        row["d_peak_err"] = row["peak_err"] - row["paper_peak_err"]
        row["d_final_err_kmax"] = row["final_err_kmax"] - row["paper_final_err_kmax"]
    return row


def _k_axis_log2(ax, ks):
    """Log2 width axis: a tick at every width of the grid, labels only at powers of two (no crowding)."""
    ax.set_xscale("log", base=2)
    ax.set_xticks(ks)
    ax.set_xticklabels([str(k) if k & (k - 1) == 0 else "" for k in ks], fontsize=6)
    ax.minorticks_off()


def _as_test(g, col):
    """Rename <col>_mean/_std to test_err_mean/_std (for metrics that expect those names)."""
    return g.rename(columns={f"{col}_mean": "test_err_mean", f"{col}_std": "test_err_std"})


def _heatmap_matrix(ax, piv, title, ylabel="epoch", cmap="viridis", vmin=0, vmax=None):
    """Draw a pivot table (index: epoch/step, columns: width) as a heat map with a log time axis."""
    ks = list(piv.columns)
    t = piv.index.to_numpy(dtype=float)
    xe = np.arange(len(ks) + 1) - 0.5
    ye = np.concatenate([[t[0] / 1.2], np.sqrt(t[:-1] * t[1:]), [t[-1] * 1.2]])
    im = ax.pcolormesh(xe, ye, piv.to_numpy(), shading="flat", cmap=cmap, vmin=vmin, vmax=vmax)
    ax.set_yscale("log")
    ax.set_xticks(range(len(ks)))
    many = len(ks) > 20                                  # e.g. the paper's 64 widths: label powers of two only
    ax.set_xticklabels([str(k) if not many or k & (k - 1) == 0 else "" for k in ks], rotation=90, fontsize=6)
    ax.set(xlabel="width k", ylabel=ylabel, title=title)
    ax.grid(False)
    return im


def _heatmap(ax, c, value, title):
    """Width (categorical) x epoch/step (log) heat map of the mean of `value`."""
    piv = c.pivot_table(index="tick", columns="k", values=value, aggfunc="mean").sort_index()
    return _heatmap_matrix(ax, piv, title, "epoch" if c["budget_unit"].iloc[0] == "epochs" else "step")


def _spans(ax, ks_marked, grid, color):
    """Shade the given widths (half-way to their neighbours on the grid)."""
    grid = sorted(grid)
    for k in ks_marked:
        i = grid.index(k)
        lo = (grid[i - 1] * k) ** 0.5 if i > 0 else k / 1.2
        hi = (grid[i + 1] * k) ** 0.5 if i < len(grid) - 1 else k * 1.2
        ax.axvspan(lo, hi, color=color, alpha=0.12, lw=0)


# ----------------------------------------------------------------------------- ResNet line (member 1)
def fig_a1(args):
    """A1 / Fig. 1 left: final train and test error vs width, 15 % noise, mean +- std over 5 inits."""
    fin = final(curves(["R1", "R3", "R5"]))
    if not _require(fin, "A1"):
        return
    g = agg_seeds(fin[(fin["noise"] == 0.15) & (fin["noise_seed"] == 0)], ["k"])
    fig, ax = plt.subplots(figsize=(COL_W, 2.4))
    _plot_mean_std(ax, g, "test_err", "test (ours)")
    _plot_mean_std(ax, g, "train_err", "train (ours)", marker="s", ls="--")
    _overlay_orig(ax, "fig1", label="test (paper)", noisy_p=0.15)
    _overlay_orig(ax, "fig1", col="train_err", label="train (paper)", ls=":")
    thr = interpolation_threshold(g["k"], g["train_err_mean"])
    if not np.isnan(thr):
        ax.axvline(thr, color="k", lw=0.8, ls=":", label=f"train err < 1% from k={thr:g}")
    ax.set(xlabel="ResNet18 width k", ylabel="error", title="CIFAR-10, 15% label noise, 4K epochs")
    ax.legend()
    save(fig, "A1_fig1_resnet_p15")
    save_table(g, "A1_fig1_resnet_p15")


def fig_a2(args):
    """A2 / Fig. 2 + Fig. 1 right: width x epoch heat maps (15 %), and test error vs width at fixed epochs."""
    c = curves(["R1", "R3"])
    c = c[(c["noise"] == 0.15) & (c["init_seed"] == 0)] if len(c) else c
    if not _require(c, "A2"):
        return
    fig, axes = plt.subplots(1, 2, figsize=(FULL_W, 2.6), sharey=True)
    for ax, col, name in zip(axes, ["test_err", "train_err"], ["test error", "train error"]):
        fig.colorbar(_heatmap(ax, c, col, f"{name}, 15% noise"), ax=ax)
    save(fig, "A2_fig2_heatmap_p15")
    o = load_original("fig2")
    if o is not None:                                   # the paper's heat map on our evaluation epochs
        o = o[o["epoch"].isin(set(c["tick"]))]
        ours = c.pivot_table(index="tick", columns="k", values="test_err", aggfunc="mean").sort_index()
        theirs = o.pivot_table(index="epoch", columns="k", values="test_err", aggfunc="mean").sort_index()
        common_k = [k for k in ours.columns if k in theirs.columns]
        common_t = [t for t in ours.index if t in theirs.index]
        diff = ours.loc[common_t, common_k] - theirs.loc[common_t, common_k]
        lim = float(np.nanmax(np.abs(diff.to_numpy()))) if diff.size else 0.1
        fig, axes = plt.subplots(1, 2, figsize=(FULL_W, 2.6), sharey=True)
        fig.colorbar(_heatmap_matrix(axes[0], theirs, "paper: test error, 15% noise"), ax=axes[0])
        fig.colorbar(_heatmap_matrix(axes[1], diff, "ours - paper (test error)", cmap="RdBu_r",
                                     vmin=-lim, vmax=lim), ax=axes[1])
        save(fig, "A2_fig2_heatmap_p15_vs_paper")
    fig, ax = plt.subplots(figsize=(COL_W, 2.4))
    avail = np.sort(c["tick"].unique())
    for e in (50, 200, 1000, 4000):
        t = avail[np.abs(avail - e).argmin()]
        g = c[c["tick"] == t].groupby("k", as_index=False)["test_err"].mean()
        ax.plot(g["k"], g["test_err"], marker="o", label=f"epoch {t:g}")
    ax.set(xlabel="ResNet18 width k", ylabel="test error", title="15% noise: test error after t epochs")
    ax.legend()
    save(fig, "A2_fig1right_epochs_p15")


def fig_a3(args):
    """A3 / Fig. 4b: final test error vs width for 0/10/15/20 % noise; peak height and position."""
    fin = final(curves(["R1", "R2", "R3", "R4", "R5"]))
    if not _require(fin, "A3"):
        return
    fin = fin[fin["noise_seed"] == 0]
    fig, ax = plt.subplots(figsize=(COL_W, 2.4))
    rows, drew = [], False
    for i, p in enumerate(NOISES):
        d = fin[fin["noise"] == p]
        if d.empty:
            continue
        g = agg_seeds(d, ["k"])
        _plot_mean_std(ax, g, "test_err", f"{p:.0%} noise", color=COLORS[i])
        drew |= _overlay_orig(ax, "fig4b", query=f"noise == {p}", label=None, color=COLORS[i])
        rows.append({"noise": p, "runs": len(d),
                     **curve_summary(g["k"], g["train_err_mean"], g["test_err_mean"], g["test_err_std"])})
    ax.set(xlabel="ResNet18 width k", ylabel="test error", title="CIFAR-10, ResNet18, 4K epochs")
    _paper_legend(ax, drew)
    save(fig, "A3_fig4b_noise_levels")
    save_table(pd.DataFrame(rows), "A3_noise_peaks")


def _regime_widths(fin, args):
    """Small (never interpolates), medium (smallest width reaching <1 % train error), large (max width)."""
    medium = args.medium if args.medium else interpolation_threshold(fin["k"], fin["train_err"])
    if pd.isna(medium):
        medium = 16
    small = args.small if args.small else (4 if 4 in set(fin["k"]) else int(fin["k"].min()))
    large = args.large if args.large else int(fin["k"].max())
    return small, medium, large


def fig_a4(args):
    """A4 / Fig. 9: 20 % noise heat map and test/train error vs epoch for three regimes."""
    c = curves(["R2", "R3"])
    c = c[(c["noise"] == 0.20) & (c["init_seed"] == 0)] if len(c) else c
    if not _require(c, "A4"):
        return
    small, medium, large = _regime_widths(final(c, only_complete=False), args)
    fig, axes = plt.subplots(1, 2, figsize=(FULL_W, 2.6))
    fig.colorbar(_heatmap(axes[0], c, "test_err", "test error, 20% noise"), ax=axes[0])
    ax = axes[1]
    rows, drew = [], False
    for i, (name, k) in enumerate([("small", small), ("medium", medium), ("large", large)]):
        d = c[c["k"] == k].sort_values("tick")
        if d.empty:
            continue
        ax.plot(d["tick"], d["test_err"], color=COLORS[i], label=f"{name} k={k:g} test")
        ax.plot(d["tick"], d["train_err"], color=COLORS[i], ls="--", lw=0.8, label=f"{name} train")
        drew |= _overlay_orig(ax, "fig9", query=f"k == {k}", label=None, x="epoch", color=COLORS[i], ls=":")
        row = {"regime": name, "k": k, **epochwise_peak(d["tick"], d["test_err"])}
        o9 = load_original("fig9")
        po = o9[o9["k"] == k].dropna(subset=["test_err"]).sort_values("epoch") if o9 is not None else []
        if len(po) > 5:                                     # the authors' run of the same width
            row.update({f"paper_{key}": v for key, v in epochwise_peak(po["epoch"], po["test_err"]).items()})
        rows.append(row)
    ax.set_xscale("log")
    ax.set(xlabel="epoch", ylabel="error", title="ResNet18, 20% noise")
    handles, _ = ax.get_legend_handles_labels()
    if drew:
        handles.append(Line2D([], [], color="0.3", ls=":", lw=1.0, label="paper test (dotted)"))
    ax.legend(handles=handles, ncol=2, fontsize=6)
    save(fig, "A4_fig9_epochwise_p20")
    save_table(pd.DataFrame(rows), "A4_epochwise_peaks")


def fig_a5(args):
    """A5 (extension): fraction of noisy labels memorised vs test error over epochs, 20 % noise."""
    c = curves(["R2", "R3"])
    c = c[(c["noise"] == 0.20) & (c["init_seed"] == 0)] if len(c) else c
    if not _require(c, "A5"):
        return
    _, medium, large = _regime_widths(final(c, only_complete=False), args)
    widths = [k for k in dict.fromkeys([medium, large]) if k in set(c["k"])]
    fig, axes = plt.subplots(1, len(widths), figsize=(FULL_W if len(widths) > 1 else COL_W, 2.4), squeeze=False)
    rows = []
    for ax, k in zip(axes[0], widths):
        d = c[c["k"] == k].sort_values("tick")
        ax.plot(d["tick"], d["test_err"], color=COLORS[0], label="test error")
        ax2 = ax.twinx()
        ax2.plot(d["tick"], d["noisy_memorized"], color=COLORS[3], label="noisy labels memorised")
        ax2.set_ylim(0, 1.02)
        ax2.set_ylabel("fraction memorised")
        ax2.grid(False)
        hit = d.loc[d["train_err"] < 0.01, "tick"]
        t_interp = float(hit.iloc[0]) if len(hit) else np.nan
        if not np.isnan(t_interp):
            ax.axvline(t_interp, color="k", ls=":", lw=0.8, label="train err < 1%")
        pk = epochwise_peak(d["tick"], d["test_err"])
        at_peak = d.loc[d["tick"] == pk["peak_tick"]]
        hit10 = d.loc[d["train_err"] < 0.10, "tick"]        # the paper's EMC heuristic uses epsilon = 0.1
        rows.append({"k": k, "interp_epoch": t_interp,
                     "train_below_10pct_epoch": float(hit10.iloc[0]) if len(hit10) else np.nan,
                     "train_err_at_test_peak": at_peak["train_err"].iloc[0] if len(at_peak) else np.nan,
                     "memorized_at_test_peak": at_peak["noisy_memorized"].iloc[0] if len(at_peak) else np.nan, **pk})
        ax.set_xscale("log")
        ax.set(xlabel="epoch", ylabel="test error", title=f"ResNet18 k={k:g}, 20% noise")
        h1, l1 = ax.get_legend_handles_labels()
        h2, l2 = ax2.get_legend_handles_labels()
        ax.legend(h1 + h2, l1 + l2, fontsize=6)
    save(fig, "A5_memorization_p20")
    save_table(pd.DataFrame(rows), "A5_memorization")


def fig_a6(args):
    """A6: final vs oracle-early-stopped test error per noise level (does the peak disappear?)."""
    es = oracle_es(curves(["R1", "R2", "R3", "R4", "R5"]))
    if not _require(es, "A6"):
        return
    es = es[es["noise_seed"] == 0]
    fig, axes = plt.subplots(1, 4, figsize=(FULL_W, 2.1), sharey=True)
    rows = []
    for ax, p in zip(axes, NOISES):
        d = es[es["noise"] == p]
        ax.set(title=f"{p:.0%} noise", xlabel="width k")
        if d.empty:
            continue
        g = agg_seeds(d, ["k"], cols=("test_err", "es_test_err", "train_err"))
        _plot_mean_std(ax, g, "test_err", "final")
        _plot_mean_std(ax, g, "es_test_err", "oracle early stop", marker="s")
        f = find_peak(g["k"], g["test_err_mean"], g["test_err_std"])
        e = find_peak(g["k"], g["es_test_err_mean"], g["es_test_err_std"])
        rows.append({"noise": p, "final_peak_k": f["peak_k"], "final_prominence": f["prominence"],
                     "final_significant": f["significant"], "es_peak_k": e["peak_k"],
                     "es_prominence": e["prominence"], "es_significant": e["significant"]})
    axes[0].set_ylabel("test error")
    axes[0].legend(fontsize=6)
    save(fig, "A6_early_stopping")
    save_table(pd.DataFrame(rows), "A6_early_stopping")


def fig_a7(args):
    """A7 / Fig. 28: 5-model plurality-vote ensembles (independent noise draws) vs single models."""
    fin = final(curves(["R1", "R6"]))
    if not _require(fin, "A7"):
        return
    fin = fin[fin["noise"] == 0.15]
    widths = sorted(fin.loc[fin["exp"] == "R6", "k"].unique())
    if not widths:
        print("A7: no finished R6 runs yet - skipped")
        return
    y_path = DATA_DIR / "cifar10_test_y.npy"     # plain numpy: no PyTorch needed for the analysis
    if not y_path.exists():
        print(f"A7: {y_path} not found - run scripts/prepare_data.py here or copy that file")
        return
    y_test = np.load(y_path).astype(np.int64)
    rows = []
    for k in widths:
        d = fin[fin["k"] == k].sort_values("init_seed")   # R1: seed 0 / noise draw 0; R6: seeds 1-4 / draws 1-4
        preds = []
        for rid in d["run_id"]:
            _, mm = test_preds(rid)
            if not (mm[-1] == 255).all():
                preds.append(np.asarray(mm[-1], dtype=np.int64))
        if len(preds) < 2:
            continue
        P = np.stack(preds)
        single = (P != y_test[None]).mean(1)
        ens = float((plurality_vote(P, 10) != y_test).mean())
        rows.append({"k": k, "models": len(P), "single_mean": single.mean(), "single_std": single.std(ddof=1),
                     "ensemble": ens, "gain": single.mean() - ens})
    t = pd.DataFrame(rows)
    if t.empty:
        print("A7: not enough finished models per width - skipped")
        return
    fig, ax = plt.subplots(figsize=(COL_W, 2.4))
    ax.errorbar(t["k"], t["single_mean"], yerr=t["single_std"], marker="o", capsize=2, label="single model (mean)")
    ax.plot(t["k"], t["ensemble"], marker="s", label="5-model ensemble")
    _overlay_orig(ax, "fig28", query="kind == 'ensemble'", label="ensemble (paper)")
    ax.set(xlabel="ResNet18 width k", ylabel="test error", title="15% noise: ensembling")
    ax.legend()
    save(fig, "A7_fig28_ensemble")
    save_table(t, "A7_ensemble")


def table_t1(args):
    """T1: per noise level - interpolation threshold, peak, location check, final and early-stopped error."""
    c = curves(["R1", "R2", "R3", "R4", "R5"])
    fin = final(c)
    if not _require(fin, "T1"):
        return
    es = oracle_es(c)
    fin, es = fin[fin["noise_seed"] == 0], es[es["noise_seed"] == 0]
    rows = []
    for p in NOISES:
        d = fin[fin["noise"] == p]
        if d.empty:
            continue
        g = agg_seeds(d, ["k"])
        s = curve_summary(g["k"], g["train_err_mean"], g["test_err_mean"], g["test_err_std"])
        ge = agg_seeds(es[es["noise"] == p], ["k"], cols=("es_test_err",))
        at_peak = ge.loc[ge["k"] == s["peak_k"], "es_test_err_mean"]
        o = load_original("fig4b")
        paper = _paper_cols(o[np.isclose(o["noise"], p)] if o is not None and "noise" in o else None, g["k"])
        rows.append(_with_deltas({"noise": p, "widths": len(g), "max_seeds": int(g["n_seeds"].max()), **s,
                     "es_err_at_peak_k": at_peak.iloc[0] if len(at_peak) else np.nan,
                     "es_peak_significant": find_peak(ge["k"], ge["es_test_err_mean"], ge["es_test_err_std"])["significant"],
                     **paper}, grid=g["k"]))
    t = pd.DataFrame(rows)
    save_table(t, "T1_resnet_summary")
    print(t.to_string(index=False))


# ----------------------------------------------------------------------------- CNN line (member 2)
def fig_b1(args, use_es: bool = False):
    """B1 / Fig. 11a: test error vs width for 4 training-set sizes; widths where 2x / 4x data does not help."""
    c = curves(["R8", "R12"])
    fin = oracle_es(c) if use_es else final(c)
    name = "B4_fig11a_early_stopped" if use_es else "B1_fig11a_samplewise"
    if not _require(fin, name):
        return
    col = "es_test_err" if use_es else "test_err"
    fig, ax = plt.subplots(figsize=(COL_W, 2.5))
    groups, drew = {}, False
    for i, n in enumerate(SUBSETS):
        d = fin[fin["n"] == n]
        if d.empty:
            continue
        groups[n] = agg_seeds(d, ["k"], cols=(col, "train_err"))
        _plot_mean_std(ax, groups[n], col, f"n = {n:,}", color=COLORS[i])
        if not use_es:
            drew |= _overlay_orig(ax, "fig11a", query=f"n == {n}", label=None, color=COLORS[i])
    rows, extra = [], []
    grid = sorted(fin["k"].unique())
    for (small, large), color in [((25000, 50000), "green"), ((12500, 50000), "red")]:
        if small in groups and large in groups:
            ks = no_help_widths(_as_test(groups[small], col), _as_test(groups[large], col))
            # Only the critical regime is shaded: widths from the valley of the smaller set's curve on. Below it the
            # tiny models are capacity-limited and gain nothing from more data anyway (listed in the table only).
            gs = groups[small]
            valley = find_peak(gs["k"], gs[f"{col}_mean"])["valley_k"]
            crit = [k for k in ks if np.isnan(valley) or k >= valley]
            rows.append({"comparison": f"{large // small}x data ({small} -> {large})",
                         "widths_no_help": " ".join(map(str, ks)), "count": len(ks),
                         "valley_k_smaller_n": valley, "widths_no_help_critical": " ".join(map(str, crit)),
                         "count_critical": len(crit)})
            _spans(ax, crit, grid, color)
            if crit:   # legend entry for the shading, as in the paper's Fig. 11a
                extra.append(Patch(color=color, alpha=0.25, label=f"{large // small}x data does not help"))
    _k_axis_log2(ax, grid)
    ax.set(xlabel="CNN width k", ylabel="test error" + (" (early stopped)" if use_es else ""),
           title="CIFAR-10, 20% noise" + (", oracle early stopping" if use_es else ""))
    handles, _ = ax.get_legend_handles_labels()
    if drew:
        extra.append(Line2D([], [], color="0.3", ls="--", lw=1.0, label="paper (dashed)"))
    ax.legend(handles=handles + extra, fontsize=6)
    save(fig, name)
    save_table(pd.DataFrame(rows), name + "_nohelp")


def fig_b2(args):
    """B2 / Fig. 12 left: width x training-set-size heat map with the interpolation threshold."""
    fin = final(curves(["R8", "R12"]))
    if not _require(fin, "B2"):
        return
    g = agg_seeds(fin, ["n", "k"])
    piv = g.pivot(index="n", columns="k", values="test_err_mean").sort_index()
    ks, ns = list(piv.columns), list(piv.index)
    fig, ax = plt.subplots(figsize=(COL_W, 2.4))
    im = ax.pcolormesh(np.arange(len(ks) + 1) - 0.5, np.arange(len(ns) + 1) - 0.5, piv.to_numpy(),
                       shading="flat", cmap="viridis")
    fig.colorbar(im, ax=ax, label="test error")
    thr = []
    for j, n in enumerate(ns):
        gn = g[g["n"] == n]
        t = interpolation_threshold(gn["k"], gn["train_err_mean"])
        thr.append({"n": n, "interp_thr_1pct": t})
        if not pd.isna(t):
            ax.plot(ks.index(t), j, marker="o", color="white", ms=4)
    ax.set_xticks(range(len(ks)))
    ax.set_xticklabels([str(k) for k in ks], fontsize=6)
    ax.set_yticks(range(len(ns)))
    ax.set_yticklabels([f"{n:,}" for n in ns])
    ax.set(xlabel="CNN width k", ylabel="training samples n", title="test error (dots: interpolation threshold)")
    ax.grid(False)
    save(fig, "B2_fig12_heatmap")
    save_table(pd.DataFrame(thr), "B2_thresholds")


def fig_b3(args, use_es: bool = False):
    """B3 / Fig. 12 right: test error vs n for a small, a near-critical and a large width."""
    c = curves(["R8", "R12"])
    fin = oracle_es(c) if use_es else final(c)
    name = "B4_fig12right_early_stopped" if use_es else "B3_fig12right_slices"
    if not _require(fin, name):
        return
    col = "es_test_err" if use_es else "test_err"
    g = agg_seeds(fin, ["k", "n"], cols=(col, "train_err"))
    if args.b3_widths:
        widths = args.b3_widths
    else:
        g25 = g[g["n"] == 25000]
        crit = interpolation_threshold(g25["k"], g25["train_err_mean"]) if len(g25) else np.nan
        widths = [3, 12 if pd.isna(crit) else crit, int(g["k"].max())]
    fig, ax = plt.subplots(figsize=(COL_W, 2.4))
    rows = []
    for i, (label, k) in enumerate(zip(["small", "near-critical", "large"], widths)):
        d = g[g["k"] == k].sort_values("n")
        if d.empty:
            continue
        ax.errorbar(d["n"], d[f"{col}_mean"], yerr=d[f"{col}_std"].fillna(0), marker="o", capsize=2,
                    color=COLORS[i], label=f"{label} (k={k:g})")
        m, s = d[f"{col}_mean"].to_numpy(), d[f"{col}_std"].fillna(0).to_numpy()
        worse = [f"{a}->{b}" for a, b, ma, mb, sa, sb in zip(d["n"], d["n"][1:], m, m[1:], s, s[1:]) if mb - ma > sa + sb]
        rows.append({"width": k, "regime": label, **{f"n={n}": v for n, v in zip(d["n"], m)},
                     "more_data_hurts_at": " ".join(worse)})
    ax.set_xscale("log", base=2)
    ax.set_xticks(SUBSETS)
    ax.set_xticklabels([f"{n // 1000}k" if n % 1000 == 0 else f"{n / 1000:g}k" for n in SUBSETS])
    ax.minorticks_off()
    ax.set(xlabel="training samples n", ylabel="test error" + (" (early stopped)" if use_es else ""),
           title="effect of more data, 20% noise")
    ax.legend(fontsize=6)
    save(fig, name)
    save_table(pd.DataFrame(rows), name)


def fig_b4(args):
    """B4: B1 and B3 with oracle early stopping (paper: more data never hurt with optimal early stopping)."""
    fig_b1(args, use_es=True)
    fig_b3(args, use_es=True)


def fig_b5(args):
    """B5 / Fig. 5: without vs with augmentation, 0/10/20 % noise, n = 50k, seed 0."""
    fin = final(curves(["R8", "R9", "R10"]))
    if not _require(fin, "B5"):
        return
    fin = fin[(fin["n"] == 50000) & (fin["init_seed"] == 0) & (fin["wd"] == 0)]
    fig, axes = plt.subplots(1, 2, figsize=(FULL_W, 2.4), sharey=True)
    rows, drew = [], False
    for ax, aug in zip(axes, [0, 1]):
        for i, p in enumerate([0.0, 0.10, 0.20]):
            d = fin[(fin["aug"] == aug) & (fin["noise"] == p)]
            if d.empty:
                continue
            g = agg_seeds(d, ["k"])
            _plot_mean_std(ax, g, "test_err", f"{p:.0%} noise", color=COLORS[i])
            drew |= _overlay_orig(ax, "fig5", query=f"noise == {p} and aug == {aug}", label=None, color=COLORS[i])
            rows.append({"aug": aug, "noise": p,
                         **curve_summary(g["k"], g["train_err_mean"], g["test_err_mean"], g["test_err_std"])})
        _k_axis_log2(ax, sorted(fin["k"].unique()))
        ax.set(title="with augmentation" if aug else "without augmentation", xlabel="CNN width k")
    axes[0].set_ylabel("test error")
    _paper_legend(axes[1], drew, fontsize=6)
    save(fig, "B5_fig5_augmentation")
    save_table(pd.DataFrame(rows), "B5_augmentation")


def fig_b6(args):
    """B6 / Fig. 21: weight decay 5e-4 vs none (augmentation, 10 % noise)."""
    c = curves(["R10", "R11"])
    c = c[(c["noise"] == 0.10) & (c["aug"] == 1) & (c["n"] == 50000)] if len(c) else c
    fin = final(c) if len(c) else c
    if not _require(fin, "B6"):
        return
    fig, axes = plt.subplots(1, 2, figsize=(FULL_W, 2.4))
    rows, base = [], None
    for i, wd in enumerate(sorted(fin["wd"].unique())):
        g = agg_seeds(fin[fin["wd"] == wd], ["k"])
        _plot_mean_std(axes[0], g, "test_err", f"weight decay {wd:g}", color=COLORS[i])
        s = curve_summary(g["k"], g["train_err_mean"], g["test_err_mean"], g["test_err_std"])
        rows.append({"wd": wd, **s})
        if wd == 0:
            base = s["interp_thr_1pct"]
    _k_axis_log2(axes[0], sorted(fin["k"].unique()))
    axes[0].set(xlabel="CNN width k", ylabel="test error", title="10% noise, augmentation")
    axes[0].legend(fontsize=6)
    crit = args.b6_width or (base if base is not None and not pd.isna(base) else 16)
    for i, wd in enumerate(sorted(fin["wd"].unique())):
        d = c[(c["wd"] == wd) & (c["k"] == crit)].sort_values("step")
        axes[1].plot(d["step"], d["test_err"], color=COLORS[i], label=f"weight decay {wd:g}")
    axes[1].set_xscale("log")
    axes[1].set(xlabel="gradient step", ylabel="test error", title=f"width k={crit:g} over training")
    axes[1].legend(fontsize=6)
    save(fig, "B6_fig21_weight_decay")
    save_table(pd.DataFrame(rows), "B6_weight_decay")


def fig_b7(args):
    """B7 / Fig. 7: CIFAR-100 without label noise or augmentation (5 seeds), final and early stopped."""
    es = oracle_es(curves(["R13"]))
    if not _require(es, "B7"):
        return
    g = agg_seeds(es, ["k"], cols=("test_err", "es_test_err", "train_err"))
    fig, ax = plt.subplots(figsize=(COL_W, 2.4))
    _plot_mean_std(ax, g, "test_err", "final")
    _plot_mean_std(ax, g, "es_test_err", "oracle early stop", marker="s")
    _overlay_orig(ax, "fig7", label="paper")
    _k_axis_log2(ax, sorted(g["k"].unique()))
    ax.set(xlabel="CNN width k", ylabel="test error", title="CIFAR-100, no label noise")
    ax.legend(fontsize=6)
    save(fig, "B7_fig7_cifar100")
    rows = [{"curve": "final", **curve_summary(g["k"], g["train_err_mean"], g["test_err_mean"], g["test_err_std"])},
            {"curve": "early stopped", **curve_summary(g["k"], g["train_err_mean"], g["es_test_err_mean"], g["es_test_err_std"])}]
    save_table(pd.DataFrame(rows), "B7_cifar100")


def table_t2(args):
    """T2: threshold / peak / location check for every CNN condition."""
    rows = []
    fin = final(curves(["R8", "R12"]))
    for n in SUBSETS:
        d = fin[fin["n"] == n] if len(fin) else fin
        if len(d):
            g = agg_seeds(d, ["k"])
            o = load_original("fig11a")
            rows.append(_with_deltas({"condition": f"CIFAR-10, 20% noise, aug, n={n}", "max_seeds": int(g["n_seeds"].max()),
                         **curve_summary(g["k"], g["train_err_mean"], g["test_err_mean"], g["test_err_std"]),
                         **_paper_cols(o[o["n"] == n] if o is not None and "n" in o else None, g["k"])}, grid=g["k"]))
    fin = final(curves(["R8", "R9", "R10", "R11"]))
    if len(fin):
        fin = fin[(fin["n"] == 50000) & (fin["init_seed"] == 0)]
        for (aug, p, wd), d in fin.groupby(["aug", "noise", "wd"]):
            g = agg_seeds(d, ["k"])
            o = load_original("fig5") if wd == 0 else None
            sel = o[np.isclose(o["noise"], p) & (o["aug"] == aug)] if o is not None and "aug" in o else None
            rows.append(_with_deltas({"condition": f"CIFAR-10, {p:.0%} noise, aug={aug}, wd={wd:g}, n=50000", "max_seeds": 1,
                         **curve_summary(g["k"], g["train_err_mean"], g["test_err_mean"], g["test_err_std"]),
                         **_paper_cols(sel, g["k"])}, grid=g["k"]))
    fin = final(curves(["R13"]))
    if len(fin):
        g = agg_seeds(fin, ["k"])
        rows.append(_with_deltas({"condition": "CIFAR-100, 0% noise, no aug", "max_seeds": int(g["n_seeds"].max()),
                     **curve_summary(g["k"], g["train_err_mean"], g["test_err_mean"], g["test_err_std"]),
                     **_paper_cols(load_original("fig7"), g["k"])}, grid=g["k"]))
    if not rows:
        print("T2: no finished runs yet - skipped")
        return
    t = pd.DataFrame(rows)
    save_table(t, "T2_cnn_summary")
    print(t.to_string(index=False))


FIGURES = {"a1": fig_a1, "a2": fig_a2, "a3": fig_a3, "a4": fig_a4, "a5": fig_a5, "a6": fig_a6, "a7": fig_a7,
           "t1": table_t1, "b1": fig_b1, "b2": fig_b2, "b3": fig_b3, "b4": fig_b4, "b5": fig_b5, "b6": fig_b6,
           "b7": fig_b7, "t2": table_t2}
GROUPS = {"resnet": ["a1", "a2", "a3", "a4", "a5", "a6", "a7", "t1"],
          "cnn": ["b1", "b2", "b3", "b4", "b5", "b6", "b7", "t2"]}
GROUPS["all"] = GROUPS["resnet"] + GROUPS["cnn"]


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("which", nargs="+", help="ids (a1 ... t2) or groups: all, resnet, cnn")
    ap.add_argument("--small", type=int, default=None, help="A4: width that never interpolates (default 4)")
    ap.add_argument("--medium", type=int, default=None, help="A4/A5: width that just interpolates (default: auto)")
    ap.add_argument("--large", type=int, default=None, help="A4/A5: large width (default: largest)")
    ap.add_argument("--b3-widths", nargs=3, type=int, default=None, help="B3: small, near-critical, large width")
    ap.add_argument("--b6-width", type=int, default=None, help="B6: width for the over-training curve")
    ap.add_argument("--debug", action="store_true", help="show full tracebacks")
    args = ap.parse_args()
    names = [n for w in args.which for n in GROUPS.get(w, [w])]
    for name in names:
        if name not in FIGURES:
            print(f"unknown figure id {name!r}")
            continue
        print(f"== {name}: {FIGURES[name].__doc__.splitlines()[0]}")
        try:
            FIGURES[name](args)
        except Exception as e:   # one failing figure should not stop the others
            print(f"   {name} failed: {e!r}")
            if args.debug:
                traceback.print_exc()


if __name__ == "__main__":
    main()
