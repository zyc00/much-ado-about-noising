#!/usr/bin/env python3
"""Long-tail motivation figure, redrawn from the raw WidowX probe data in the paper's current style.
Same 1,728 states (raw/{hg,mse,flow}_rank*.npz). For every 8x6 continuous-action chunk: divide by the HG head's per-state
scale sigma(x); divide each of the 48 coordinates by its RMS estimated on the OTHER episode folds (5 folds within task);
R = ||z||_2 over the 48 coordinates. Panels: a HG residual, b MSE residual, c Flow samples minus their per-state mean
(16 per state). Curves: empirical P(R > r); isotropic Gaussian reference (chi_48); multivariate Student-t whose radial
tail is fitted on r in [sqrt(d), 2.2 sqrt(d)]. Annotation: P(R > 1.4 sqrt(d)) vs Gaussian, and the held-out NLL gain per
dimension of a multivariate Student-t over a Gaussian, both fitted by maximum likelihood on the other folds. nu is not shown."""
from __future__ import annotations
import json, sys
from pathlib import Path
import numpy as np, matplotlib as mpl
mpl.use("Agg"); import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from scipy.special import gammaln
from scipy.stats import chi, f as fdist
sys.path.insert(0, str(Path(__file__).resolve().parent))
from plot_student_t_tails import load  # noqa: E402

OUT = Path("analysis/paper/longtail_motivation"); SEED = 20260906; D = 48
COL = {"data": "#222222", "gauss": "#0072B2", "t": "#D55E00"}

def folds_of(task, episode, k=5):
    rng = np.random.default_rng(SEED); fold = np.empty(len(task), int)
    for tid in np.unique(task):
        eps = rng.permutation(np.unique(episode[task == tid]))
        for i, e in enumerate(eps): fold[episode == e] = i % k
    return fold

def crossfit_whiten(z, fold):
    """z: [N, ..., 48] with leading state axis. Whiten every 48-D vector with the covariance estimated on the OTHER
    episode folds (Cholesky), so that under a Gaussian model the norm is exactly chi_48 despite the strong within-chunk
    correlation of residual coordinates (effective dimension ~25 without whitening)."""
    shape = z.shape; zf = z.reshape(shape[0], -1, D); out = np.empty_like(zf)
    for f in np.unique(fold):
        tr = fold != f; C = np.cov(zf[tr].reshape(-1, D).T) + 1e-8 * np.eye(D); Lc = np.linalg.cholesky(C)
        out[fold == f] = np.linalg.solve(Lc, zf[fold == f].reshape(-1, D).T).T.reshape(zf[fold == f].shape)
    return out.reshape(shape)

def t_nll_per_dim(R2, nu, s):  # multivariate t in D dims, radial form
    return (-(gammaln((nu + D) / 2) - gammaln(nu / 2)) + (D / 2) * np.log(nu * np.pi * s ** 2) + ((nu + D) / 2) * np.log1p(R2 / (nu * s ** 2))) / D
def gauss_nll_per_dim(R2, s): return 0.5 * np.log(2 * np.pi * s ** 2) + R2 / (2 * D * s ** 2)
def t_mle(R2, grid=np.logspace(np.log10(1.05), np.log10(400), 140)):
    best = (np.nan, np.nan, np.inf)
    for nu in grid:
        s2 = np.mean(R2) / D
        for _ in range(40):
            w = (nu + D) / (nu + R2 / s2); s2 = np.mean(w * R2) / D
        nll = np.mean(t_nll_per_dim(R2, nu, np.sqrt(s2)))
        if nll < best[2]: best = (nu, np.sqrt(s2), nll)
    return best[0], best[1]

def heldout_gain(R2, fold):
    g, t = [], []
    for f in np.unique(fold):
        tr, te = fold != f, fold == f
        s_g = np.sqrt(np.mean(R2[tr]) / D); nu, s_t = t_mle(R2[tr])
        g.append(np.mean(gauss_nll_per_dim(R2[te], s_g))); t.append(np.mean(t_nll_per_dim(R2[te], nu, s_t)))
    return float(np.mean(g)), float(np.mean(t))

def tail_fit(R, lo=np.sqrt(D), hi=2.2 * np.sqrt(D)):
    ts = np.linspace(lo, hi, 40); emp = np.array([(R > t).mean() for t in ts]); keep = emp > 0; best = (np.nan, np.nan, np.inf)
    for nu in np.logspace(np.log10(1.5), np.log10(400), 150):
        for s in np.linspace(0.6, 1.25, 66):
            model = fdist.sf(ts[keep] ** 2 / (s ** 2 * D), D, nu)
            err = np.mean((np.log(emp[keep]) - np.log(np.maximum(model, 1e-300))) ** 2)
            if err < best[2]: best = (nu, s, err)
    return best[0], best[1]

def panel(ax, R, fold, title, tag):
    R2 = R ** 2; thr = 1.4 * np.sqrt(D)
    rs = np.linspace(5, 16, 111); emp = np.array([(R > r).mean() for r in rs]); keep = emp > 0
    nu_f, s_f = tail_fit(R); gauss = chi.sf(rs, D); tt = fdist.sf(rs ** 2 / (s_f ** 2 * D), D, nu_f)
    g_nll, t_nll = heldout_gain(R2, fold)
    ax.plot(rs, gauss, ls=(0, (4, 2)), color=COL["gauss"], lw=1.6, zorder=2)
    ax.plot(rs, tt, color=COL["t"], lw=1.8, zorder=3)
    ax.plot(rs[keep], emp[keep], color=COL["data"], lw=1.1, zorder=4)
    sel = np.arange(0, 111, 7); ok = emp[sel] > 0
    ax.plot(rs[sel][ok], emp[sel][ok], "o", ms=3.6, mfc="white", mec=COL["data"], mew=.9, zorder=5)
    ax.axvline(thr, color=".6", lw=.7, ls=":", zorder=1)
    ax.set_yscale("log"); ax.set_ylim(2e-4, 1.3); ax.set_xlim(5, 16); ax.set_xticks([5, 7, 9, 11, 13, 15])
    ax.set_xlabel(r"normalized chunk magnitude $R=\|z\|_2$"); ax.set_ylabel(r"tail probability $P(R>r)$")
    ax.set_title(title, loc="left", fontweight="bold", pad=5)
    emp_thr = float((R > thr).mean()); g_thr = float(chi.sf(thr, D))
    ax.text(.04, .05, f"$P(R>1.4\\sqrt{{d}})$: {100*emp_thr:.1f}% vs {100*g_thr:.3f}% Gaussian\nheld-out Student-$t$ NLL $\\downarrow$ {g_nll - t_nll:.3f} nat/dim",
            transform=ax.transAxes, ha="left", va="bottom", fontsize=6.3, color="#8b1a1a", fontweight="bold",
            bbox=dict(facecolor="white", edgecolor=".8", boxstyle="round,pad=.35", lw=.6), zorder=6)
    return dict(tag=tag, n=int(len(R)), dimension=D, threshold=float(thr), empirical_tail=emp_thr, gaussian_tail=g_thr, tail_ratio=emp_thr / g_thr,
                tail_fit_nu=float(nu_f), tail_fit_scale=float(s_f), heldout_gaussian_nll_per_dim=g_nll, heldout_student_nll_per_dim=t_nll,
                heldout_nll_gain_per_dim=g_nll - t_nll, q90=float(np.quantile(R, .9)), q99=float(np.quantile(R, .99)))

def main():
    mse, flow, hg = load("mse"), load("flow"), load("hg")
    for k in ("task_id", "episode", "step"): np.testing.assert_array_equal(mse[k], flow[k]); np.testing.assert_array_equal(mse[k], hg[k])
    task, episode = mse["task_id"], mse["episode"]; fold = folds_of(task, episode); sig = hg["sigma"][:, None, None]
    z_hg = crossfit_whiten(hg["residual"].astype(float) / sig, fold); z_mse = crossfit_whiten(mse["residual"].astype(float) / sig, fold)
    S = flow["samples"].astype(float); dev = (S - S.mean(axis=1, keepdims=True)) / sig[:, None]
    z_flow = crossfit_whiten(dev, fold); z_f2l = crossfit_whiten((S - flow["target"].astype(float)[:, None]) / sig[:, None], fold)
    R_hg = np.linalg.norm(z_hg.reshape(len(task), -1), axis=1); R_mse = np.linalg.norm(z_mse.reshape(len(task), -1), axis=1)
    R_flow = np.linalg.norm(z_flow.reshape(len(task), 16, -1), axis=2); fold_flow = np.repeat(fold, 16)
    R_f2l = np.linalg.norm(z_f2l.reshape(len(task), 16, -1), axis=2)
    mpl.rcParams.update({"font.family": "serif", "font.serif": ["Nimbus Roman", "DejaVu Serif"], "mathtext.fontset": "stix", "font.size": 7,
        "axes.labelsize": 7, "axes.titlesize": 7.5, "xtick.labelsize": 6.5, "ytick.labelsize": 6.5, "axes.spines.top": False,
        "axes.spines.right": False, "axes.linewidth": .7, "pdf.fonttype": 42})
    fig, axs = plt.subplots(1, 3, figsize=(7.2, 2.45))
    stats = [panel(axs[0], R_hg, fold, "a  HG normalized residual", "hg_residual"),
             panel(axs[1], R_mse, fold, "b  MSE normalized residual", "mse_residual"),
             panel(axs[2], R_flow.ravel(), fold_flow, "c  Flow sample spread", "flow_sample_spread")]
    # flow sample minus the demonstration label (spec item 6b): reported in the summary, not drawn
    fig2, ax2 = plt.subplots(); stats.append(panel(ax2, R_f2l.ravel(), fold_flow, "flow sample - label", "flow_to_label")); plt.close(fig2)
    for ax in axs: ax.set_box_aspect(1); ax.grid(axis="y", color=".92", lw=.5, zorder=0); ax.tick_params(length=2, pad=2)
    handles = [Line2D([0], [0], color=COL["data"], lw=1.1, marker="o", ms=3.6, mfc="white", mec=COL["data"], mew=.9, label="observed"),
               Line2D([0], [0], color=COL["gauss"], lw=1.6, ls=(0, (4, 2)), label=r"Gaussian ($\chi_{48}$)"),
               Line2D([0], [0], color=COL["t"], lw=1.8, label="Student-$t$ tail fit")]
    fig.legend(handles=handles, loc="upper center", ncol=3, frameon=False, bbox_to_anchor=(.5, 1.0), fontsize=7, columnspacing=2.0)
    fig.subplots_adjust(left=.07, right=.99, bottom=.17, top=.8, wspace=.5)
    fig.savefig(OUT / "fig_longtail_motivation_v2.pdf", bbox_inches="tight", pad_inches=.025); fig.savefig(OUT / "fig_longtail_motivation_v2.png", dpi=320, bbox_inches="tight", pad_inches=.025)
    (OUT / "summary_v2.json").write_text(json.dumps(dict(states=int(len(task)), folds=5, stats={s["tag"]: s for s in stats}), indent=2))
    print(json.dumps([{k: (round(v, 4) if isinstance(v, float) else v) for k, v in s.items()} for s in stats], indent=1))

if __name__ == "__main__":
    main()
