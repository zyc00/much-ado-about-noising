#!/usr/bin/env python3
"""Redesign of fig_widowx_heterogeneous_scale: panels a and c show the pooled point density (grey) with per-task
binned trends (label-RMS octiles -> geometric-mean residual / spread, 95% episode-bootstrap bars) instead of 1,728
overlapping points. Panels b and d reuse the numbers in the summary JSON. Task episode counts come from
bridge_orig_lerobot/meta/episodes.jsonl (53,192 episodes; sweep 963, open drawer 473, close drawer 416)."""
from __future__ import annotations
import json, sys
from pathlib import Path
import numpy as np, matplotlib as mpl
mpl.use("Agg"); import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
sys.path.insert(0, str(Path(__file__).resolve().parent))
from plot_widowx_heterogeneous_scale import load_matched, relative  # noqa: E402

ROOT = Path("analysis/paper/widowx_heterogeneous_scale")
COLORS = ["#0072B2", "#D55E00", "#009E73"]; MARK = ["o", "s", "^"]
EPISODES = {0: 963, 1: 473, 2: 416}  # bridge_orig_lerobot/meta/episodes.jsonl, exact lowercase instruction
LABELS = [f"Sweep into pile ({EPISODES[0]})", f"Open drawer ({EPISODES[1]})", f"Close drawer ({EPISODES[2]})"]
SEED = 20260906

def binned_trend(x, y, episode, nbins=8, boots=2000):
    """geometric-mean y per x-octile with 95% episode-bootstrap CI; x, y already relative (log-scale quantities)."""
    edges = np.quantile(x, np.linspace(0, 1, nbins + 1)); edges[-1] += 1e-9
    b = np.clip(np.searchsorted(edges, x, side="right") - 1, 0, nbins - 1)
    eps = np.unique(episode); rng = np.random.default_rng(SEED)
    lx, ly = np.log(x), np.log(y)
    xm = np.array([lx[b == i].mean() for i in range(nbins)]); ym = np.array([ly[b == i].mean() for i in range(nbins)])
    ep_idx = [np.flatnonzero(episode == e) for e in eps]
    boot = np.empty((boots, nbins))
    for k in range(boots):
        idx = np.concatenate([ep_idx[i] for i in rng.integers(len(eps), size=len(eps))])
        bb, yy = b[idx], ly[idx]
        boot[k] = [yy[bb == i].mean() if np.any(bb == i) else np.nan for i in range(nbins)]
    lo, hi = np.nanquantile(boot, [.025, .975], axis=0)
    return np.exp(xm), np.exp(ym), np.exp(lo), np.exp(hi)

def panel_points(ax, x, y, task, rho, title, ylabel, per_task=120):
    """sparse scatter: a seeded random subset of states per task (small solid dots); rho uses all states."""
    rng = np.random.default_rng(SEED); lx, ly = np.log10(x), np.log10(y)
    for tid in range(3):
        idx = np.flatnonzero(task == tid); sub = rng.choice(idx, size=min(per_task, len(idx)), replace=False)
        ax.scatter(lx[sub], ly[sub], s=4.5, color=COLORS[tid], marker="o", linewidths=0, alpha=.75, zorder=3, rasterized=False)
    ticks = [.25, 1, 4]
    ax.set_xlim(np.log10(.14), np.log10(5.5)); ax.set_ylim(np.log10(.14), np.log10(5.5))
    ax.set_xticks(np.log10(ticks), ["0.25", "1", "4"]); ax.set_yticks(np.log10(ticks), ["0.25", "1", "4"])
    ax.set_xlabel("Label RMS (relative)"); ax.set_ylabel(ylabel)
    ax.set_title(title, loc="left", fontweight="bold", pad=5)
    ax.text(.05, .96, rf"$\rho={rho:.2f}$", transform=ax.transAxes, va="top", fontsize=8,
            bbox=dict(facecolor="white", edgecolor="none", alpha=.85, pad=1))

def main():
    mse, flow, _ = load_matched(ROOT / "raw")
    S = json.load(open(ROOT / "fig_widowx_heterogeneous_scale_summary.json"))
    task, episode = mse["task_id"], mse["episode"]
    label = np.sqrt(np.mean(mse["target"].astype(float) ** 2, axis=(1, 2)))
    error = np.sqrt(np.mean(mse["residual"].astype(float) ** 2, axis=(1, 2)))
    samples = flow["samples"].astype(float)
    spread = np.sqrt(np.mean((samples - samples.mean(axis=1, keepdims=True)) ** 2, axis=(1, 2, 3)))
    xr = relative(label, task)
    mpl.rcParams.update({"font.family": "serif", "font.serif": ["Nimbus Roman", "DejaVu Serif"], "mathtext.fontset": "stix",
        "font.size": 7, "axes.labelsize": 7, "axes.titlesize": 7.5, "xtick.labelsize": 6.5, "ytick.labelsize": 6.5,
        "axes.spines.top": False, "axes.spines.right": False, "axes.linewidth": .7, "pdf.fonttype": 42})
    SQUARE = "--square" in sys.argv
    fig, axs = plt.subplots(2, 2, figsize=(3.5, 3.5)) if SQUARE else plt.subplots(1, 4, figsize=(7.2, 2.35))
    axs = axs.ravel()
    panel_points(axs[0], xr, relative(error, task), task, S["panel_a"]["rho"], "a  General MSE", "Residual RMS (relative)")
    panel_points(axs[2], xr, relative(spread, task), task, S["panel_c"]["rho"], "c  Matched Flow", "Sample spread (relative)")
    ax = axs[1]
    for tid, c in enumerate(S["panel_b"]):
        x, m, ci = np.array(c["x"]), np.array(c["mean"]), np.array(c["ci95"])
        ax.fill_between(x, ci[0], ci[1], color=COLORS[tid], alpha=.12, lw=0)
        ax.plot(x, m, color=COLORS[tid], lw=1.4, marker="o", ms=2.8, mew=.5, mec="white")
    ax.set_xlim(0, 100); ax.set_xticks([0, 50, 100]); ax.set_ylim(bottom=0)
    ax.set_xlabel("Episode progress (%)"); ax.set_ylabel("MSE residual RMS")
    ax.set_title("b  Task progress", loc="left", fontweight="bold", pad=5)
    ax.text(.96, .04, "shading: 95% CI", transform=ax.transAxes, ha="right", fontsize=5.4, color=".4")
    ax = axs[3]; means = np.asarray(S["panel_d"]["means"]); ci = np.asarray(S["panel_d"]["ci95"])
    ax.plot([0, 1], means, color=".75", lw=1.2, zorder=1)
    for i, color in enumerate(["#7A7A7A", "#6B4C9A"]):
        ax.errorbar(i, means[i], yerr=[[means[i] - ci[0, i]], [ci[1, i] - means[i]]], fmt="o", color=color, ms=5.5, capsize=3, elinewidth=1.2, mew=1, zorder=3)
        ax.annotate(f"{means[i]:.3f}", (i, ci[1, i]), xytext=(0, 5), textcoords="offset points", ha="center", fontsize=7)
    ax.set_xlim(-.45, 1.45); ax.set_xticks([0, 1], ["Fixed\nscale", "Input-dep.\nscale"])
    ax.set_ylabel("Test log likelihood (nat/dim)"); ax.set_title("d  Gaussian fit", loc="left", fontweight="bold", pad=5)
    gap = max(ci.max() - ci.min(), .04); ax.set_ylim(ci.min() - .35 * gap, ci.max() + .75 * gap)
    ax.text(.5, .97, rf"$\Delta={S['panel_d']['gain']:+.3f}$", transform=ax.transAxes, ha="center", va="top", fontsize=8)
    ax.text(.5, .04, "higher is better; bars: 95% CI", transform=ax.transAxes, ha="center", fontsize=5.4, color=".4")
    for ax in axs:
        ax.grid(axis="y", color=".92", lw=.5, zorder=0); ax.tick_params(length=2, pad=2)
    handles = [Line2D([0], [0], color=COLORS[t], marker="o", lw=1.3, ms=3.6, mew=.4, mec="white", label=LABELS[t]) for t in range(3)]
    if SQUARE:
        fig.legend(handles=handles, loc="upper center", ncol=3, frameon=False, bbox_to_anchor=(.53, 1.0), fontsize=6.3, columnspacing=1.0, handlelength=1.8, handletextpad=.5)
        fig.subplots_adjust(left=.12, right=.985, bottom=.1, top=.87, wspace=.55, hspace=.62)
        out = ROOT / "fig_widowx_heterogeneous_scale_v5_square"
    else:
        for ax in axs: ax.set_box_aspect(1)   # square panels; the row spans the full two-column width
        fig.legend(handles=handles, loc="upper center", ncol=3, frameon=False, bbox_to_anchor=(.5, 1.0), fontsize=7, columnspacing=1.8, handlelength=2.2)
        fig.subplots_adjust(left=.06, right=.99, bottom=.2, top=.8, wspace=.6)
        out = ROOT / "fig_widowx_heterogeneous_scale_v5"
    fig.savefig(out.with_suffix(".pdf"), bbox_inches="tight", pad_inches=.025)
    fig.savefig(out.with_suffix(".png"), dpi=320, bbox_inches="tight", pad_inches=.025)
    print("saved", out)

if __name__ == "__main__":
    main()
