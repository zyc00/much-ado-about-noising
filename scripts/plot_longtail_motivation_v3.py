#!/usr/bin/env python3
"""Long-tail motivation figure, per-coordinate |z| design (as data_ht_motivation/fig_heavy_tailed_residuals).
Same 1,728 WidowX states. Each residual coordinate is divided by the HG head's state-local sigma(x) and then by the
coordinate's RMS estimated on the OTHER episode folds (5 folds within task); |z| is compared with the standard normal and
with a Student-t whose (nu, scale) match the empirical tail on |z| in [1, 5] (nu not shown). Annotations: P(|z| > 3)
against the Gaussian 0.27%, and the held-out NLL gain per coordinate of a maximum-likelihood Student-t over a
maximum-likelihood Gaussian, both fitted on the other folds. Panels: a HG residual, b MSE residual, c Flow samples minus
their per-state mean (every one of the 16 samples counted). The flow-sample-minus-label variant is in the summary."""
from __future__ import annotations
import json, sys
from pathlib import Path
import numpy as np, matplotlib as mpl
mpl.use("Agg"); import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from scipy.special import gammaln
from scipy.stats import norm, t as student_t
sys.path.insert(0, str(Path(__file__).resolve().parent))
from plot_student_t_tails import load, t_mle, t_tailfit  # noqa: E402
import glob as _glob
from plot_longtail_motivation_v2 import folds_of  # noqa: E402

OUT = Path("analysis/paper/longtail_motivation"); D = 48
COL = {"data": "#222222", "gauss": "#0072B2", "t": "#D55E00"}

def crossfit_coord(z, fold):
    """z [N, ..., 8, 6]; divide each of the 48 coordinates by its RMS on the other folds."""
    shape = z.shape; zf = z.reshape(shape[0], -1, D); out = np.empty_like(zf)
    for f in np.unique(fold):
        tr = fold != f; c = np.sqrt(np.mean(zf[tr].reshape(-1, D) ** 2, axis=0)); out[fold == f] = zf[fold == f] / c
    return out

def t_nll(z, nu, s): return -(gammaln((nu + 1) / 2) - gammaln(nu / 2) - .5 * np.log(nu * np.pi) - np.log(s)) + (nu + 1) / 2 * np.log1p((z / s) ** 2 / nu)
def heldout_gain(z_by_state, fold):
    g, t = [], []
    for f in np.unique(fold):
        tr, te = z_by_state[fold != f].ravel(), z_by_state[fold == f].ravel()
        s_g = np.sqrt(np.mean(tr ** 2)); nu, s_t = t_mle(tr)
        g.append(np.mean(.5 * np.log(2 * np.pi * s_g ** 2) + te ** 2 / (2 * s_g ** 2))); t.append(np.mean(t_nll(te, nu, s_t)))
    return float(np.mean(g)), float(np.mean(t))

def panel(ax, z_by_state, fold, title, tag):
    z = z_by_state.ravel(); z = z / np.sqrt(np.mean(z ** 2)); a = np.abs(z); ts = np.linspace(0.5, 6.5, 121)
    emp = np.array([(a > t).mean() for t in ts]); keep = emp > 0
    nu, s = t_tailfit(z); gauss = 2 * norm.sf(ts); tt = 2 * student_t.sf(ts / s, nu)
    b = np.mean(np.abs(z)); lap = np.exp(-ts / b)                      # Laplace with the same unit-RMS data (exponential tail)
    g_nll, t_nl = heldout_gain(z_by_state / np.sqrt(np.mean(z_by_state ** 2)), fold)
    ax.plot(ts, gauss, ls=(0, (4, 2)), color=COL["gauss"], lw=1.6, zorder=2); ax.plot(ts, tt, color=COL["t"], lw=1.8, zorder=3)
    ax.plot(ts, lap, ls=(0, (1, 1.5)), color=".45", lw=1.2, zorder=2)
    ax.plot(ts[keep], emp[keep], color=COL["data"], lw=1.1, zorder=4)
    sel = np.arange(0, 121, 8); ok = emp[sel] > 0
    ax.plot(ts[sel][ok], emp[sel][ok], "o", ms=3.6, mfc="white", mec=COL["data"], mew=.9, zorder=5)
    ax.axvline(3, color=".6", lw=.7, ls=":", zorder=1)
    ax.set_yscale("log"); ax.set_ylim(3e-5, 1.2); ax.set_xlim(0.5, 6.5); ax.set_xticks([1, 2, 3, 4, 5, 6])
    ax.set_xlabel(r"standardized residual magnitude $|z|$"); ax.set_ylabel(r"fraction of residuals above $|z|$" if ax.get_subplotspec().is_first_col() else "")
    ax.set_title(title, loc="left", fontweight="bold", pad=5)
    frac3 = float((a > 3).mean()); g3 = float(2 * norm.sf(3))
    ax.text(.04, .05, f"$P(|z|>3)$: {100*frac3:.2f}%\nGaussian: {100*g3:.2f}% ({frac3/g3:.1f}$\\times$ less)\nStudent-$t$ NLL $\\downarrow$ {g_nll - t_nl:.3f} nat/dim",
            transform=ax.transAxes, ha="left", va="bottom", fontsize=6.1, color="#8b1a1a", fontweight="bold",
            bbox=dict(facecolor="white", edgecolor=".8", boxstyle="round,pad=.3", lw=.6), zorder=6)
    return dict(tag=tag, n=int(len(z)), frac_gt3=frac3, gaussian_gt3=g3, excess3=frac3 / g3, kurtosis=float(np.mean(z ** 4) - 3),
                tail_fit_nu=float(nu), tail_fit_scale=float(s), heldout_gaussian_nll_per_dim=g_nll, heldout_student_nll_per_dim=t_nl, heldout_nll_gain_per_dim=g_nll - t_nl)

def main():
    mse, flow, hg = load("mse"), load("flow"), load("hg")
    for k in ("task_id", "episode", "step"): np.testing.assert_array_equal(mse[k], flow[k]); np.testing.assert_array_equal(mse[k], hg[k])
    task, episode = mse["task_id"], mse["episode"]; fold = folds_of(task, episode); sig = hg["sigma"][:, None, None]
    # each model under ITS OWN scale: HG divides by its predicted sigma(x); MSE has no sigma and gets only a fixed
    # per-coordinate scale (the homoscedastic assumption it makes); Flow divides by its own per-state sample spread.
    z_hg = crossfit_coord(hg["residual"].astype(float) / sig, fold); z_mse = crossfit_coord(mse["residual"].astype(float), fold)
    S = flow["samples"].astype(float)
    # Flow with K = 1024 samples per state (raw_k1024/flow_rank*.npz, same states): panel c uses one fixed (global)
    # coordinate scale; panel d divides by the Flow model's OWN conditional scale, the RMS spread of its 1024 samples at
    # that state (a scalar per state; with 1024 draws its estimation noise cannot induce a small-nu artifact).
    K = {}
    for f in sorted(_glob.glob("analysis/paper/widowx_heterogeneous_scale/raw_k1024/flow_rank*.npz")):
        with np.load(f, allow_pickle=False) as z:
            for k in z.files:
                if k != "metadata": K.setdefault(k, []).append(z[k])
    K = {k: np.concatenate(v) for k, v in K.items()}; order = np.lexsort((K["step"], K["episode"], K["task_id"])); K = {k: v[order] for k, v in K.items()}
    for k in ("task_id", "episode", "step"): np.testing.assert_array_equal(K[k], mse[k])
    S1k = K["samples"].astype(np.float32); dev = S1k - S1k.mean(axis=1, keepdims=True)              # [N, 1024, 8, 6]
    # work in the RMS-normalized space: every coordinate divided by its held-out RMS FIRST (cross-fit), then the Flow
    # model's own conditional scale = scalar RMS spread over the 48 normalized coordinates and all 1024 draws at that state.
    dev_rms = crossfit_coord(dev, fold).reshape(len(task), 1024, D)
    s_flow = np.sqrt(np.mean(dev_rms.astype(np.float64) ** 2, axis=(1, 2)))[:, None, None].astype(np.float32)
    # panel c: fixed (global) scale only; panel d: divided by the own conditional scale. Pooled curves use every 16th draw
    # (64 per state, 5.3M values) to bound memory; the per-state scale uses all 1024 draws.
    z_flow_glob = dev_rms[:, ::16]; z_flow = dev_rms[:, ::16] / s_flow
    lab_rms = crossfit_coord(S1k[:, ::16] - flow["target"].astype(np.float32)[:, None], fold).reshape(len(task), 64, D)
    z_f2l = lab_rms / s_flow
    del S1k, dev, dev_rms
    mpl.rcParams.update({"font.family": "serif", "font.serif": ["Nimbus Roman", "DejaVu Serif"], "mathtext.fontset": "stix", "font.size": 7,
        "axes.labelsize": 7, "axes.titlesize": 7.5, "xtick.labelsize": 6.5, "ytick.labelsize": 6.5, "axes.spines.top": False,
        "axes.spines.right": False, "axes.linewidth": .7, "pdf.fonttype": 42})
    fig, axs = plt.subplots(1, 4, figsize=(7.2, 2.0))
    stats = [panel(axs[0], z_hg, fold, "a  HG residual / its $\\sigma(x)$", "hg_residual"),
             panel(axs[1], z_mse, fold, "b  MSE residual (fixed scale)", "mse_residual"),
             panel(axs[2], z_flow_glob, fold, "c  Flow samples $-$ mean (fixed scale)", "flow_spread_fixed_scale"),
             panel(axs[3], z_flow, fold, "d  Flow samples $-$ mean / own $\\sigma(x)$", "flow_spread_own_scale")]
    fig2, ax2 = plt.subplots(); stats.append(panel(ax2, z_f2l, fold, "flow sample - label", "flow_to_label")); plt.close(fig2)
    for ax in axs: ax.set_box_aspect(1); ax.grid(axis="y", color=".92", lw=.5, zorder=0); ax.tick_params(length=2, pad=2)
    handles = [Line2D([0], [0], color=COL["data"], lw=1.1, marker="o", ms=3.6, mfc="white", mec=COL["data"], mew=.9, label="observed"),
               Line2D([0], [0], color=COL["gauss"], lw=1.6, ls=(0, (4, 2)), label="Gaussian"),
               Line2D([0], [0], color=COL["t"], lw=1.8, label="Student-$t$ tail fit"),
               Line2D([0], [0], color=".45", lw=1.2, ls=(0, (1, 1.5)), label="Laplace")]
    fig.legend(handles=handles, loc="upper center", ncol=4, frameon=False, bbox_to_anchor=(.5, 1.0), fontsize=7, columnspacing=2.0)
    fig.subplots_adjust(left=.06, right=.99, bottom=.2, top=.78, wspace=.5)
    fig.savefig(OUT / "fig_longtail_motivation_v3.pdf", bbox_inches="tight", pad_inches=.025); fig.savefig(OUT / "fig_longtail_motivation_v3.png", dpi=320, bbox_inches="tight", pad_inches=.025)
    (OUT / "summary_v3.json").write_text(json.dumps(dict(states=int(len(task)), folds=5, stats={s["tag"]: s for s in stats}), indent=2))
    print(json.dumps([{k: (round(v, 4) if isinstance(v, float) else v) for k, v in s.items()} for s in stats], indent=1))

if __name__ == "__main__":
    main()
