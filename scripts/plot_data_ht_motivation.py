#!/usr/bin/env python3
"""Create ICLR-width data-motivation figures for heteroscedastic Student-t BC."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
from scipy.stats import norm


TRANSIT = "#D55E00"
PRECISION = "#0072B2"
EMPIRICAL = "#1A1A1A"
GRID = "#D8D8D8"


def set_style():
    mpl.rcParams.update({
        "font.family": "serif",
        "font.serif": ["STIXGeneral", "DejaVu Serif"],
        "mathtext.fontset": "stix",
        "font.size": 9,
        "axes.labelsize": 9,
        "xtick.labelsize": 8.5,
        "ytick.labelsize": 8.5,
        "legend.fontsize": 8.2,
        "axes.linewidth": .8,
        "lines.linewidth": 1.5,
        "pdf.fonttype": 42,
        "ps.fonttype": 42,
        "savefig.bbox": "tight",
        "savefig.pad_inches": .025,
    })


def save_both(fig, stem: Path):
    stem.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(stem.with_suffix(".pdf"))
    fig.savefig(stem.with_suffix(".png"), dpi=300)


def phase_gaussian_mixture_figure(data, summary, out: Path):
    # Every coordinate's center and scale are estimated on fold 0. The plot
    # displays only fold 1, so the component radii are held-out estimates.
    residual = data["residual"].astype(np.float64)
    phase = data["phase"]
    fold = data["fold"]
    fit = summary["phase_gaussian_mixture_fit"][0]
    center = np.asarray(fit["coordinate_mean"])
    scale = np.asarray(fit["coordinate_std"])
    heldout = fold == fit["test_fold"]
    standardized = (residual[heldout] - center) / scale
    heldout_phase = phase[heldout]
    transit = standardized[np.isin(heldout_phase, (0, 2))].reshape(-1)
    precision = standardized[np.isin(heldout_phase, (1, 3))].reshape(-1)
    pooled = np.concatenate((transit, precision))
    sigma_transit = fit["transit_sigma"]
    sigma_precision = fit["precision_sigma"]
    transit_weight = fit["transit_weight"]

    fig, ax = plt.subplots(figsize=(5.5, 2.60))
    edges = np.linspace(-3.25, 3.25, 86)
    bin_width = edges[1] - edges[0]
    ax.hist(pooled, bins=edges,
            weights=np.full(len(pooled), 1/(len(pooled)*bin_width)),
            color="#B7B7B7", edgecolor="white", linewidth=.25, alpha=.58,
            label="Pooled held-out residuals")

    x = np.linspace(edges[0], edges[-1], 500)
    gaussian_transit = norm.pdf(x, scale=sigma_transit)
    gaussian_precision = norm.pdf(x, scale=sigma_precision)
    mixture = (transit_weight*gaussian_transit
               +(1-transit_weight)*gaussian_precision)
    ax.plot(x, gaussian_transit, color=TRANSIT, linestyle="--", linewidth=1.7,
            label=rf"Transit component ($\sigma={sigma_transit:.2f}$)")
    ax.plot(x, gaussian_precision, color=PRECISION, linestyle="-.", linewidth=1.7,
            label=rf"Precision component ($\sigma={sigma_precision:.2f}$)")
    ax.plot(x, mixture, color=EMPIRICAL, linestyle="-", linewidth=1.7,
            label="Gaussian scale mixture")
    ax.fill_between(x, gaussian_transit, where=np.abs(x)<=sigma_transit,
                    color=TRANSIT, alpha=.055)
    ax.fill_between(x, gaussian_precision, where=np.abs(x)<=sigma_precision,
                    color=PRECISION, alpha=.07)

    radius_y = .065
    ax.annotate("", xy=(sigma_precision, radius_y), xytext=(0, radius_y),
                arrowprops={"arrowstyle": "<->", "color": PRECISION,
                            "linewidth": 1.0})
    ax.text(sigma_precision/2, radius_y+.018, "smaller radius", color=PRECISION,
            ha="center", va="bottom", fontsize=7.8, fontweight="semibold")
    ax.annotate("", xy=(-sigma_transit, radius_y), xytext=(0, radius_y),
                arrowprops={"arrowstyle": "<->", "color": TRANSIT,
                            "linewidth": 1.0})
    ax.text(-sigma_transit/2, radius_y+.018, "larger radius", color=TRANSIT,
            ha="center", va="bottom", fontsize=7.8, fontweight="semibold")

    ax.text(.5, .965,
            "Same mean, different phase-dependent radii   "
            + rf"({fit['precision_percent_smaller']:.0f}\% smaller in precision phases)",
            transform=ax.transAxes, ha="center", va="top", fontsize=8.7,
            fontweight="semibold")
    ax.set_xlabel("Continuous-action residual (coordinate-normalized)")
    ax.set_ylabel("Density")
    ax.set_xlim(edges[0], edges[-1])
    ax.set_ylim(0, .62)
    ax.set_yticks([0, .2, .4, .6])
    ax.grid(axis="y", color=GRID, linewidth=.55)
    ax.set_axisbelow(True)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    handles, labels = ax.get_legend_handles_labels()
    order = [0, 3, 1, 2]
    ax.legend([handles[i] for i in order], [labels[i] for i in order],
              frameon=False, loc="upper right", bbox_to_anchor=(.995, .88),
              handlelength=2.7, borderaxespad=0)
    fig.subplots_adjust(left=.105, right=.99, bottom=.21, top=.98)
    save_both(fig, out / "fig_phase_gaussian_mixture")
    plt.close(fig)


def phase_scale_figure(data, summary, out: Path):
    """Cross-fitted local-scale distribution and held-out calibration."""
    calibration = summary["local_scale_calibration"][0]
    query_fold = calibration["query_fold"]
    keep = data["fold"] == query_fold
    local_scale = data["local_scale"][keep].astype(np.float64)
    groups = calibration["groups"]
    predicted = np.array([g["predicted_radius_median"] for g in groups])
    observed = np.array([g["heldout_residual_rms"] for g in groups])
    ci = np.array([g["episode_bootstrap_95_ci"] for g in groups])
    cuts = np.asarray(calibration["local_radius_quantile_edges"])

    fig, (ax0, ax1) = plt.subplots(
        1, 2, figsize=(5.5, 2.34), gridspec_kw={"width_ratios": [1.08, 1]})

    # Panel A: the local scale itself varies substantially over states.
    edges = np.geomspace(.03, .31, 30)
    ax0.axvspan(cuts[0], cuts[1], color=PRECISION, alpha=.13, linewidth=0)
    ax0.axvspan(cuts[-2], cuts[-1], color=TRANSIT, alpha=.13, linewidth=0)
    ax0.hist(local_scale, bins=edges, color="#9F9F9F", edgecolor="white",
             linewidth=.35)
    ax0.set_xscale("log")
    ax0.set_xlim(.03, .31)
    ax0.set_xticks([.04, .06, .1, .15, .25])
    ax0.get_xaxis().set_major_formatter(mpl.ticker.ScalarFormatter())
    ax0.ticklabel_format(axis="x", style="plain")
    ymax = ax0.get_ylim()[1]
    q10, q90 = np.quantile(local_scale, [.1, .9])
    arrow_y = .82*ymax
    ax0.annotate("", xy=(q90, arrow_y), xytext=(q10, arrow_y),
                 arrowprops={"arrowstyle": "<->", "linewidth": 1.0,
                             "color": "#222222"})
    ax0.text(np.sqrt(q10*q90), arrow_y+.045*ymax,
             rf"central 80\% spans {q90/q10:.2f}$\times$",
             ha="center", va="bottom", fontsize=7.7, fontweight="semibold")
    ax0.text(np.sqrt(cuts[0]*cuts[1]), .08*ymax, "low-scale\nstates",
             color=PRECISION, ha="center", va="bottom", fontsize=7.7,
             fontweight="semibold")
    ax0.text(np.sqrt(cuts[-2]*cuts[-1]), .08*ymax, "high-scale\nstates",
             color=TRANSIT, ha="center", va="bottom", fontsize=7.7,
             fontweight="semibold")
    ax0.set_xlabel(r"Estimated local radius $\hat\sigma(o)$")
    ax0.set_ylabel("Number of states")
    ax0.set_title("a   Local radius varies across states", loc="left",
                  fontsize=8.7, fontweight="semibold", pad=5)

    # Panel B: a radius estimated without the query action predicts its error.
    ax1.plot([.05, .16], [.05, .16], color="#888888", linestyle="--",
             linewidth=1.0, label="perfect calibration")
    ax1.plot(predicted, observed, color=EMPIRICAL, linewidth=1.35, zorder=2)
    point_colors = [PRECISION, "#56A0C9", "#777777", "#DE8B45", TRANSIT]
    for i, color in enumerate(point_colors):
        ax1.errorbar(predicted[i], observed[i],
                     yerr=[[observed[i]-ci[i,0]], [ci[i,1]-observed[i]]],
                     fmt="o", color=color, markeredgecolor="white",
                     markeredgewidth=.65, markersize=5.3, capsize=2,
                     linewidth=1.0, zorder=3)
        ax1.text(predicted[i], observed[i]+.0065, f"Q{i+1}", color=color,
                 ha="center", va="bottom", fontsize=7.2, fontweight="semibold")
    ratio_ci = calibration["high_over_low_bootstrap_95_ci"]
    ax1.text(.97, .06,
             rf"Q5 / Q1 = {calibration['high_over_low_rms']:.2f}$\times$"
             "\n" + rf"95\% CI [{ratio_ci[0]:.2f}, {ratio_ci[1]:.2f}]"
             "\n" + r"within-phase perm. $p<2.5\!\times\!10^{-4}$",
             transform=ax1.transAxes, ha="right", va="bottom", fontsize=7.8,
             bbox={"boxstyle": "round,pad=.22", "facecolor": "white",
                   "edgecolor": "#BBBBBB", "linewidth": .65, "alpha": .95})
    ax1.set_xlim(.052, .155)
    ax1.set_ylim(.058, .158)
    ax1.set_xlabel(r"Estimated radius $\hat\sigma(o)$")
    ax1.set_ylabel("Held-out residual RMS")
    ax1.set_title("b   Radius predicts unseen spread", loc="left",
                  fontsize=8.7, fontweight="semibold", pad=5)

    for ax in (ax0, ax1):
        ax.grid(axis="y", color=GRID, linewidth=.5)
        ax.set_axisbelow(True)
        for side in ("top", "right"):
            ax.spines[side].set_visible(False)
    fig.subplots_adjust(left=.105, right=.99, bottom=.22, top=.91, wspace=.36)
    save_both(fig, out / "fig_phase_dependent_scale")
    plt.close(fig)


def tail_figure(summary, out: Path):
    tail = summary["tail_plot"]
    fit = summary["coordinate_controlled_student_fit"][0]
    x = np.asarray(tail["grid"])
    empirical = np.asarray(tail["empirical"])
    gaussian = np.asarray(tail["gaussian"])
    student = np.asarray(tail["student"])
    at_three = tail["threshold_survival"]["3"]

    fig, ax = plt.subplots(figsize=(5.5, 2.52))
    ax.semilogy(x, gaussian, color=PRECISION, linestyle="--", linewidth=1.55,
                label="Gaussian fit")
    ax.semilogy(x, student, color=TRANSIT, linestyle="-", linewidth=1.75,
                label=rf"Student-$t$ fit ($\nu={fit['student_df']:.1f}$)")
    ids = np.arange(0, len(x), 4)
    ax.semilogy(x, empirical, color=EMPIRICAL, linewidth=1.25)
    ax.semilogy(x[ids], empirical[ids], linestyle="none", marker="o",
                markerfacecolor="white", markeredgecolor=EMPIRICAL,
                markeredgewidth=.75, markersize=3.0, label="Held-out data")

    ax.axvline(3, color="#777777", linestyle=":", linewidth=.8)
    ax.text(3.10, .17,
            rf"$P(|z|>3)$: {100*at_three['empirical']:.2f}\% data"
            "\n" + rf"vs. {100*at_three['gaussian']:.2f}\% Gaussian"
            "\n" + rf"({at_three['empirical_over_gaussian']:.1f}$\times$ more)",
            fontsize=8.2, ha="left", va="top",
            bbox={"boxstyle": "round,pad=.24", "facecolor": "white",
                  "edgecolor": "#BBBBBB", "linewidth": .7, "alpha": .94})
    ax.text(.985, .035,
            rf"held-out NLL $\downarrow$ {fit['heldout_student_nll_gain']:.3f} nat / action dim",
            transform=ax.transAxes, ha="right", va="bottom", fontsize=8,
            color="#333333")

    ax.set_xlabel(r"Standardized residual magnitude $|z|$")
    ax.set_ylabel("Two-sided tail probability")
    ax.set_xlim(.45, 6.3)
    ax.set_ylim(1e-5, .55)
    ax.set_yticks([1e-1, 1e-2, 1e-3, 1e-4, 1e-5])
    ax.grid(axis="y", which="major", color=GRID, linewidth=.55)
    ax.set_axisbelow(True)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    ax.legend(frameon=False, loc="lower left", bbox_to_anchor=(0, .08),
              handlelength=2.5, borderaxespad=0)
    fig.subplots_adjust(left=.13, right=.99, bottom=.22, top=.97)
    save_both(fig, out / "fig_heavy_tailed_residuals")
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=Path,
                        default=Path("analysis/paper/data_ht_motivation"))
    parser.add_argument("--out", type=Path, default=None)
    args = parser.parse_args()
    out = args.out or args.input
    set_style()
    with (args.input / "summary.json").open() as f:
        summary = json.load(f)
    data = np.load(args.input / "probe.npz")
    phase_gaussian_mixture_figure(data, summary, out)
    phase_scale_figure(data, summary, out)
    tail_figure(summary, out)


if __name__ == "__main__":
    main()
