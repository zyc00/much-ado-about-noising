#!/usr/bin/env python3
"""Three tail-probability panels on the same 1,728 WidowX demonstration states (6 continuous action channels x 8 steps):
  a  heteroscedastic Gaussian head: residual / its own per-state sigma(x)
  b  MSE head: residual / the same learned state-local sigma(x)
  c  Flow head: deviation of each of 16 samples from the per-state sample mean / sigma(x)
All divided per coordinate by the coordinate RMS afterwards, then rescaled to unit RMS; the curve is the residual tail P(|z| > t) (fraction of coordinates with magnitude above t) on a log axis against the
standard normal and a Student-t fitted by maximum likelihood (nu, scale). Annotation: excess of 3-sigma events over Gaussian.
Data: analysis/paper/widowx_heterogeneous_scale/raw/{hg,mse,flow}_rank*.npz (probe_widowx_general_scale.py, probe_widowx_hg_scale.py)."""
from __future__ import annotations
import glob, json, sys
from pathlib import Path
import numpy as np, matplotlib as mpl
mpl.use("Agg"); import matplotlib.pyplot as plt
from scipy.special import gammaln
from scipy.stats import norm, t as student_t

RAW = Path("analysis/paper/widowx_heterogeneous_scale/raw"); OUT = Path("analysis/paper/student_t_tails")
COL = {"data": "#222222", "gauss": "#0072B2", "t": "#D55E00"}

def load(mode):
    parts = {}
    for f in sorted(glob.glob(str(RAW / f"{mode}_rank*.npz"))):
        with np.load(f, allow_pickle=False) as z:
            for k in z.files:
                if k != "metadata": parts.setdefault(k, []).append(z[k])
    d = {k: np.concatenate(v) for k, v in parts.items()}
    order = np.lexsort((d["step"], d["episode"], d["task_id"])); return {k: v[order] for k, v in d.items()}

def t_mle(z, grid=np.logspace(np.log10(1.2), np.log10(200), 120)):
    z = z[np.isfinite(z)]; best = (np.nan, -np.inf, np.nan)
    for nu in grid:
        # profile the scale: a few Newton-free iterations of the t-MLE fixed point for the scale at fixed nu
        s2 = np.mean(z ** 2)
        for _ in range(30):
            w = (nu + 1) / (nu + z ** 2 / s2); s2 = np.mean(w * z ** 2)
        s = np.sqrt(s2)
        ll = np.mean(gammaln((nu + 1) / 2) - gammaln(nu / 2) - .5 * np.log(nu * np.pi) - np.log(s) - (nu + 1) / 2 * np.log1p((z / s) ** 2 / nu))
        if ll > best[1]: best = (nu, ll, s)
    return best[0], best[2]

def t_tailfit(z, lo=1.0, hi=5.0):
    """Student-t (nu, scale) that matches the empirical residual tail on |z| in [lo, hi] (least squares in log tail
    probability). The maximum-likelihood fit is dominated by the bulk; the tail fit describes what the figure shows."""
    a = np.abs(z); ts = np.linspace(lo, hi, 41); emp = np.array([(a > t).mean() for t in ts]); keep = emp > 0
    best = (np.nan, np.nan, np.inf)
    for nu in np.logspace(np.log10(1.5), np.log10(300), 160):
        for sc in np.linspace(0.55, 1.3, 76):
            err = np.mean((np.log(emp[keep]) - np.log(2 * student_t.sf(ts[keep] / sc, nu))) ** 2)
            if err < best[2]: best = (nu, sc, err)
    return best[0], best[1]

def panel(ax, z, title, subtitle, tag):
    z = z / np.sqrt(np.mean(z ** 2))                        # unit RMS: compare shapes, not calibration
    a = np.abs(z); n = len(a); ts = np.linspace(0.5, 6.5, 121)
    emp = np.array([(a > t).mean() for t in ts])
    nu_mle, s_mle = t_mle(z); nu, s = t_tailfit(z)
    gauss = 2 * norm.sf(ts); tt = 2 * student_t.sf(ts / s, nu)
    excess3 = (a > 3).mean() / (2 * norm.sf(3))
    keep = emp > 0
    ax.plot(ts, gauss, ls=(0, (4, 2)), color=COL["gauss"], lw=1.6, label="Gaussian", zorder=2)
    ax.plot(ts, tt, color=COL["t"], lw=1.8, label=rf"Student-$t$ ($\nu$={nu:.1f})", zorder=3)
    ax.plot(ts[keep], emp[keep], color=COL["data"], lw=1.1, zorder=4)
    sel = np.arange(0, 121, 8)
    ax.plot(ts[sel][emp[sel] > 0], emp[sel][emp[sel] > 0], "o", ms=3.6, mfc="white", mec=COL["data"], mew=.9, zorder=5, label="observed")
    ax.set_yscale("log"); ax.set_ylim(3e-5, 1.2); ax.set_xlim(0.5, 6.5); ax.set_xticks([1, 2, 3, 4, 5, 6])
    ax.axvline(3, color=".6", lw=.7, ls=":", zorder=1)
    ax.set_xlabel(r"standardized residual $|z|$"); ax.set_ylabel(r"fraction of residuals above $|z|$" if ax.get_subplotspec().is_first_col() else "")
    ax.set_title(title, loc="left", fontweight="bold", pad=5)
    ax.text(.97, .05, f"$P(|z|>3)$: {100*(a>3).mean():.2f}% vs {100*2*norm.sf(3):.2f}% Gaussian\n{excess3:.1f}$\\times$ more", transform=ax.transAxes,
            ha="right", va="bottom", fontsize=6.3, color="#8b1a1a", fontweight="bold",
            bbox=dict(facecolor="white", edgecolor=".8", boxstyle="round,pad=.35", lw=.6))
    ax.text(.97, .96, rf"Student-$t$ tail fit: $\nu$={nu:.1f}", transform=ax.transAxes, ha="right", va="top", fontsize=6.3, color=COL["t"])
    return dict(tag=tag, n=int(n), nu_tailfit=float(nu), scale_tailfit=float(s), nu_mle=float(nu_mle), scale_mle=float(s_mle), frac_gt3=float((a > 3).mean()), gauss_gt3=float(2 * norm.sf(3)),
                excess3=float(excess3), frac_gt4=float((a > 4).mean()), kurtosis=float(np.mean(z ** 4) / np.mean(z ** 2) ** 2 - 3))

def main():
    mse, flow, hg = load("mse"), load("flow"), load("hg")
    for k in ("task_id", "episode", "step"): np.testing.assert_array_equal(mse[k], flow[k]); np.testing.assert_array_equal(mse[k], hg[k])
    np.testing.assert_allclose(mse["target"], hg["target"], atol=1e-5)
    # state-local scale for every panel: the HG head's own per-state sigma(x) (a learned, observation-conditional scale,
    # estimated independently of the MSE residual and of the Flow samples); then each coordinate is divided by its RMS
    # so that coordinate-level scale differences do not masquerade as tails. Same logic as the Tool-Hang data-side figure.
    sig = hg["sigma"][:, None, None]
    percoord = lambda z: z / np.sqrt(np.mean(z ** 2, axis=tuple(range(z.ndim - 1))))
    z_hg = percoord(hg["residual"].astype(float) / sig).ravel()
    z_mse = percoord(mse["residual"].astype(float) / sig).ravel()
    S = flow["samples"].astype(float); z_flow = percoord((S - S.mean(axis=1, keepdims=True)) / sig[:, None]).ravel()
    mpl.rcParams.update({"font.family": "serif", "font.serif": ["Nimbus Roman", "DejaVu Serif"], "mathtext.fontset": "stix", "font.size": 7,
        "axes.labelsize": 7, "axes.titlesize": 7.5, "xtick.labelsize": 6.5, "ytick.labelsize": 6.5, "axes.spines.top": False,
        "axes.spines.right": False, "axes.linewidth": .7, "pdf.fonttype": 42})
    fig, axs = plt.subplots(1, 3, figsize=(7.2, 2.45))
    stats = [panel(axs[0], z_hg, r"a  Heteroscedastic Gaussian:  $r/\sigma(x)$", "", "hg"),
             panel(axs[1], z_mse, r"b  MSE:  $r/\sigma(x)$", "", "mse"),
             panel(axs[2], z_flow, r"c  Flow samples:  $(a_k-\bar a)/\sigma(x)$", "", "flow")]
    for ax in axs:
        ax.set_box_aspect(1); ax.grid(axis="y", color=".92", lw=.5, zorder=0); ax.tick_params(length=2, pad=2)
    from matplotlib.lines import Line2D
    handles = [Line2D([0], [0], color=COL["data"], lw=1.1, marker="o", ms=3.6, mfc="white", mec=COL["data"], mew=.9, label="observed"),
               Line2D([0], [0], color=COL["gauss"], lw=1.6, ls=(0, (4, 2)), label="Gaussian"),
               Line2D([0], [0], color=COL["t"], lw=1.8, label="Student-$t$ (tail fit)")]
    fig.legend(handles=handles, loc="upper center", ncol=3, frameon=False, bbox_to_anchor=(.5, 1.0), fontsize=7, columnspacing=2.0)
    fig.subplots_adjust(left=.07, right=.99, bottom=.17, top=.8, wspace=.5)
    OUT.mkdir(parents=True, exist_ok=True)
    fig.savefig(OUT / "fig_student_t_tails.pdf", bbox_inches="tight", pad_inches=.025); fig.savefig(OUT / "fig_student_t_tails.png", dpi=320, bbox_inches="tight", pad_inches=.025)
    hg_meta = json.loads(str(np.load(sorted(glob.glob(str(RAW / "hg_rank*.npz")))[0], allow_pickle=False)["metadata"]))
    (OUT / "summary.json").write_text(json.dumps(dict(states=int(len(mse["task_id"])), panels=stats, hg_checkpoint=hg_meta["checkpoint"]), indent=2))
    print(json.dumps(stats, indent=1))

if __name__ == "__main__":
    main()
