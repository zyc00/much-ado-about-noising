#!/usr/bin/env python3
"""Four-panel long-tail motivation figure drawn from precomputed curves (analysis/paper/longtail_motivation/longtail_curves.json,
produced on the cluster by compute_longtail_curves.py from the WidowX probe files; 1,728 states, Flow with 1024 draws).
Panels: a HG residual / its own sigma(x); b MSE residual, fixed scale; c Flow samples - mean, fixed scale;
d Flow samples - mean / the Flow model's own conditional scale (RMS spread of its 1024 draws, RMS-normalized space).
Curves: observed residual tail: fraction of coordinates with magnitude > |z|; standard normal; Laplace (same mean |z|); Student-t matched to the tail (fitted nu shown in each panel)."""
import json
from pathlib import Path
import numpy as np, matplotlib as mpl
mpl.use("Agg"); import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from scipy.stats import norm, t as student_t
OUT = Path("analysis/paper/longtail_motivation"); C = json.load(open(OUT / "longtail_curves.json"))
COL = {"data": "#222222", "gauss": "#0072B2", "t": "#D55E00"}
mpl.rcParams.update({"font.family": "serif", "font.serif": ["Nimbus Roman", "DejaVu Serif"], "mathtext.fontset": "stix", "font.size": 7,
    "axes.labelsize": 7, "axes.titlesize": 7.5, "xtick.labelsize": 6.5, "ytick.labelsize": 6.5, "axes.spines.top": False,
    "axes.spines.right": False, "axes.linewidth": .7, "pdf.fonttype": 42})
fig, axs = plt.subplots(1, 4, figsize=(7.2, 2.3))
panels = [("a_hg", r"a  HG residual / its $\sigma(x)$"), ("b_mse", "b  MSE residual (fixed scale)"),
          ("c_flow_fixed", r"c  Flow samples $-$ mean (fixed scale)"), ("d_flow_own", r"d  Flow samples $-$ mean / own $\sigma(x)$")]
for ax, (key, title) in zip(axs, panels):
    r = C[key]; ts, emp = np.array(r["ts"]), np.array(r["emp"]); keep = emp > 0
    ax.plot(ts, 2 * norm.sf(ts), ls=(0, (4, 2)), color=COL["gauss"], lw=1.6, zorder=2)
    ax.plot(ts, 2 * student_t.sf(ts / r["tailfit_scale"], r["tailfit_nu"]), color=COL["t"], lw=1.8, zorder=3)
    ax.plot(ts, np.exp(-ts / r["laplace_b"]), ls=(0, (1, 1.5)), color=".45", lw=1.2, zorder=2)
    ax.plot(ts[keep], emp[keep], color=COL["data"], lw=1.1, zorder=4)
    sel = np.arange(0, len(ts), 8); ok = emp[sel] > 0
    ax.plot(ts[sel][ok], emp[sel][ok], "o", ms=3.6, mfc="white", mec=COL["data"], mew=.9, zorder=5)
    ax.axvline(3, color=".6", lw=.7, ls=":", zorder=1)
    ax.set_yscale("log"); ax.set_ylim(3e-5, 1.2); ax.set_xlim(0.5, 6.5); ax.set_xticks([1, 2, 3, 4, 5, 6])
    ax.set_xlabel(r"standardized residual magnitude $|z|$"); ax.set_ylabel(r"fraction of residuals above $|z|$" if ax.get_subplotspec().is_first_col() else "")
    p3, g3, gain = r["p_gt3"], r["gauss_gt3"], r["heldout_gauss"] - r["heldout_student"]
    ax.set_title(title, loc="left", fontweight="bold", pad=31)
    ax.text(0, 1.03, f"$P(|z|>3)$ = {100*p3:.2f}%, {p3/g3:.1f}$\\times$ Gaussian\nStudent-$t$ fit: $\\nu$ = {r['tailfit_nu']:.1f}\nNLL $\\downarrow$ {gain:.3f} nat/dim (held-out)",
            transform=ax.transAxes, ha="left", va="bottom", fontsize=5.8, color="#8b1a1a", fontweight="bold", linespacing=1.25)
    ax.set_box_aspect(1); ax.grid(axis="y", color=".92", lw=.5, zorder=0); ax.tick_params(length=2, pad=2)
handles = [Line2D([0], [0], color=COL["data"], lw=1.1, marker="o", ms=3.6, mfc="white", mec=COL["data"], mew=.9, label="observed"),
           Line2D([0], [0], color=COL["gauss"], lw=1.6, ls=(0, (4, 2)), label="Gaussian"),
           Line2D([0], [0], color=COL["t"], lw=1.8, label="Student-$t$ tail fit"),
           Line2D([0], [0], color=".45", lw=1.2, ls=(0, (1, 1.5)), label="Laplace")]
fig.legend(handles=handles, loc="upper center", ncol=4, frameon=False, bbox_to_anchor=(.5, 1.0), fontsize=7, columnspacing=2.0)
fig.subplots_adjust(left=.06, right=.99, bottom=.19, top=.68, wspace=.5)
fig.savefig(OUT / "fig_longtail_motivation_v4.pdf", bbox_inches="tight", pad_inches=.025); fig.savefig(OUT / "fig_longtail_motivation_v4.png", dpi=320, bbox_inches="tight", pad_inches=.025)
print("saved", OUT / "fig_longtail_motivation_v4.png")
