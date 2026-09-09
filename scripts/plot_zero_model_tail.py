"""Plot the zero-output baseline and trained MSE using the exact v4 statistics."""
import argparse
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
import numpy as np
from scipy.stats import norm, t

from probe_zero_model_tail import load, make_folds, standardize


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--early-dir', type=Path,
                        help='Replace zero-model panel with a measured early checkpoint; save separate outputs')
    args = parser.parse_args()
    out = Path("analysis/paper/longtail_motivation")
    summary = json.loads((out / "zero_model_tail_summary.json").read_text())
    data = load("analysis/paper/widowx_heterogeneous_scale/raw/mse_rank*.npz")
    fold = make_folds(data["task_id"], data["episode"])
    cases = [
        ("zero_model", r"a  Zero model: $f(o)=0$", -data["target"]),
        ("trained_mse", "b  Trained MSE", data["residual"]),
    ]
    stem = 'zero_model_tail'
    if args.early_dir:
        report = json.loads((args.early_dir / 'early_checkpoint_tail_summary.json').read_text())
        objective = report['objective']
        key = 'early_mse' if objective == 'mse' else 'early_ht'
        early = load(str(args.early_dir / f'{key}_rank0.npz'))
        for key in ('episode', 'task_id', 'step', 'target'):
            np.testing.assert_allclose(early[key], data[key], rtol=0, atol=1e-7)
        key = 'early_mse' if objective == 'mse' else 'early_ht'
        summary[key] = report[key]
        steps = report.get('training_steps', 2000)
        label = f'{steps // 1000}k' if steps % 1000 == 0 else str(steps)
        name = 'MSE' if objective == 'mse' else 'HT'
        cases[0] = (key, f'a  Early {name} checkpoint ({label})', early['residual'])
        cases[1] = ('trained_mse', 'b  Trained MSE (20k)', data['residual'])
        stem = f'mse{steps}_tail' if objective == 'mse' else 'early_checkpoint_tail'
    plt.rcParams.update({
        "font.family": "serif", "font.serif": ["Nimbus Roman", "DejaVu Serif"],
        "mathtext.fontset": "stix", "font.size": 9, "axes.titlesize": 11,
        "axes.labelsize": 9, "xtick.labelsize": 8, "ytick.labelsize": 8,
        "axes.spines.top": False, "axes.spines.right": False,
        "axes.linewidth": .9, "pdf.fonttype": 42,
    })
    colors = dict(observed="#242424", gaussian="#0072B2", tail="#D55E00", mle="#8264A6")
    fig, axes = plt.subplots(1, 2, figsize=(7.2, 3.65), sharex=True, sharey=True)
    grid = np.linspace(.5, 6.5, 481)
    curves = {}
    for ax, (key, title, residual) in zip(axes, cases):
        stats = summary[key]
        z = standardize(residual, fold).ravel()
        magnitudes = np.sort(np.abs(z))
        counts = len(z) - np.searchsorted(magnitudes, grid, side="right")
        survival = counts / len(z)
        np.testing.assert_allclose((np.abs(z) > 3).mean(), stats["p_gt3"])
        gaussian = 2 * norm.sf(grid)
        tail = 2 * t.sf(grid / stats["tailfit_scale"], stats["tailfit_nu"])
        mle = 2 * t.sf(grid / stats["mle_scale"], stats["mle_nu"])
        ax.plot(grid, gaussian, color=colors["gaussian"], lw=1.7, ls=(0, (5, 2.5)))
        ax.plot(grid, mle, color=colors["mle"], lw=1.7, ls=(0, (1.2, 2)))
        ax.plot(grid, tail, color=colors["tail"], lw=2)
        # Empirical zeros are omitted on the log axis, never replaced by a positive floor.
        ax.plot(grid, np.where(counts > 0, survival, np.nan), color=colors["observed"], lw=1.4)
        mark = np.arange(0, len(grid), 24)
        mark = mark[counts[mark] > 0]
        ax.plot(grid[mark], survival[mark], "o", ms=4, mfc="white", mec=colors["observed"], mew=1.1)
        ax.axvline(3, color=".65", lw=.85, ls=":", zorder=0)
        ax.set_yscale("log")
        ax.set_xlim(.5, 6.5)
        ax.set_ylim(1e-5, 1)
        ax.set_xticks(range(1, 7))
        ax.grid(axis="y", color=".90", lw=.65)
        ax.set_axisbelow(True)
        ax.set_title(title, loc="left", fontweight="bold", pad=44)
        tail_text = (r"Tail fit: $\nu=300$ (search limit)" if stats["tailfit_hits_upper_nu_bound"]
                     else rf"Tail fit: $\nu={stats['tailfit_nu']:.2f}$")
        ax.text(0, 1.035, tail_text + "\n" + rf"MLE: $\nu={stats['mle_nu']:.2f}$;  $P(|z|>3)={100*stats['p_gt3']:.2f}\%$",
                transform=ax.transAxes, ha="left", va="bottom", fontsize=8, linespacing=1.45, color=".25")
        ax.set_xlabel(r"Standardized coordinate magnitude $|z|$", labelpad=6)
        if key == "zero_model":
            maximum = stats["max_abs_z"]
            last = np.flatnonzero(counts > 0)[-1]
            ax.annotate("No observed values\nbeyond 4.22", xy=(grid[last], survival[last]), xytext=(5.3, 2.5e-4),
                        ha="center", va="center", fontsize=7.5, color=".35",
                        arrowprops=dict(arrowstyle="->", color=".45", lw=.9))
        curves[key] = dict(grid=grid.tolist(), observed=survival.tolist(), counts=counts.tolist(),
                           gaussian=gaussian.tolist(), student_tail_fit=tail.tolist(), student_mle=mle.tolist())
    axes[0].set_ylabel(r"Tail probability $P(|Z|>t)$", labelpad=6)
    legend = [
        Line2D([], [], color=colors["observed"], lw=1.4, marker="o", ms=4, mfc="white", label="Observed"),
        Line2D([], [], color=colors["gaussian"], lw=1.7, ls=(0, (5, 2.5)), label="Gaussian"),
        Line2D([], [], color=colors["tail"], lw=2, label="Student-$t$ tail fit"),
        Line2D([], [], color=colors["mle"], lw=1.7, ls=(0, (1.2, 2)), label="Student-$t$ MLE"),
    ]
    fig.legend(handles=legend, loc="upper center", bbox_to_anchor=(.51, .995), ncol=4,
               frameon=False, fontsize=8, handlelength=2.3, columnspacing=1.5)
    fig.subplots_adjust(left=.10, right=.99, bottom=.18, top=.69, wspace=.25)
    for suffix in ("png", "pdf"):
        fig.savefig(out / f"fig_{stem}.{suffix}", dpi=300, bbox_inches="tight", pad_inches=.05)
    (out / f"{stem}_curves.json").write_text(json.dumps(curves, indent=2))


if __name__ == "__main__":
    main()
