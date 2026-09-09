"""Diagnostics used by sigma_gradient_note.tex.

Panels (a) uses frozen real GR00T/Fractal features and heads from the existing
gradient audit. Panels (b--d) are analytic or deterministic Monte-Carlo
illustrations of the losses, not robot success measurements.
"""
from __future__ import annotations

import json
import math
from pathlib import Path

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
from scipy.special import gammaln


ROOT = Path(__file__).resolve().parents[3]
OUT = Path(__file__).resolve().parent
DATA = ROOT / "analysis/diagnostics/nu_recipe_debug/shared_feature_gradients"


def kappa(d: int, nu: float) -> float:
    return float(np.exp(
        0.5 * math.log(nu)
        + gammaln((d + 1) / 2) - gammaln(d / 2)
        + gammaln(nu - 0.5) - gammaln(nu - 1)
        + 2 * gammaln((nu - 1) / 2) - 2 * gammaln(nu / 2)
    ))


def setup() -> None:
    mpl.rcParams.update({
        "font.family": "DejaVu Sans",
        "font.size": 9,
        "axes.titlesize": 10.5,
        "axes.labelsize": 9,
        "legend.fontsize": 8,
        "axes.spines.top": False,
        "axes.spines.right": False,
        "axes.linewidth": 0.9,
        "grid.color": "#D8DCE3",
        "grid.linewidth": 0.7,
        "grid.alpha": 0.75,
        "pdf.fonttype": 42,
        "ps.fonttype": 42,
    })


def main() -> None:
    setup()
    blue, orange, green, gray = "#1678B4", "#D95F02", "#009E73", "#555555"
    fig, axes = plt.subplots(1, 4, figsize=(13.0, 2.75))

    # (a) Measured branch gradients on frozen, shared action-head features.
    ax = axes[0]
    nus = np.array([7, 14, 32, 64, 128, 224, 448, 1024], dtype=float)
    for checkpoint, color, marker in [(10000, blue, "o"), (12000, orange, "s")]:
        obj = json.loads((DATA / f"checkpoint_{checkpoint}.json").read_text())
        ratios = np.array([
            obj["settings"][str(int(nu))]["sigma_over_mu_norm_median"]
            for nu in nus
        ])
        ax.plot(nus / 56.0, ratios, color=color, marker=marker, ms=4.3,
                lw=2.0, label=f"checkpoint {checkpoint // 1000}k")
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlabel(r"robustness ratio $\nu/d$")
    ax.set_ylabel(r"median $\|g_x^\sigma\|/\|g_x^\mu\|$")
    ax.grid(True, which="both", axis="both")
    ax.legend(frameon=False, loc="lower right")
    ax.set_title("a  Measured shared-feature gradients", loc="left", fontweight="bold")

    # (b) Analytic log-scale score as a function of standardized energy.
    ax = axes[1]
    ratio = np.linspace(0.0, 8.0, 600)  # q/d
    hg = 1.0 - ratio
    ax.plot(ratio, hg, color=gray, lw=2.0, label="Gaussian")
    for c, color, label in [(4.0, blue, r"Student-$t$: $\nu/d=4$"),
                            (0.125, orange, r"Student-$t$: $\nu/d=0.125$")]:
        gt = c * (1.0 - ratio) / (c + ratio)
        ax.plot(ratio, gt, color=color, lw=2.2, label=label)
    ax.axhline(0, color="black", lw=0.7)
    ax.axvline(1, color="#9A9A9A", lw=1.0, ls=":")
    ax.set_ylim(-3.3, 1.2)
    ax.set_xlabel(r"standardized energy $q/d$")
    ax.set_ylabel(r"$\partial\ell/\partial\log\sigma$")
    ax.grid(True, axis="y")
    ax.legend(frameon=False, loc="lower left")
    ax.set_title("b  Small nu caps scale correction", loc="left", fontweight="bold")

    # (c) Existing softplus raw-head chain factor versus direct log-scale.
    ax = axes[2]
    sigma = np.geomspace(0.01, 10.0, 500)
    eps = 1e-3
    x = np.maximum(sigma - eps, 1e-12)
    softplus_factor = (1.0 - np.exp(-x)) / sigma
    ax.plot(sigma, np.ones_like(sigma), color=green, lw=2.2,
            label=r"predict $s=\log\sigma$")
    ax.plot(sigma, softplus_factor, color=orange, lw=2.2,
            label=r"$\sigma=\mathrm{softplus}(u)+10^{-3}$")
    for value in (1.0, 5.0):
        f = (1.0 - math.exp(-(value - eps))) / value
        ax.scatter([value], [f], color=orange, s=30, zorder=3)
        ax.annotate(f"{f:.2f}", (value, f), xytext=(4, 5),
                    textcoords="offset points", color=orange, fontsize=8)
    ax.set_xscale("log")
    ax.set_ylim(0, 1.08)
    ax.set_xlabel(r"predicted scale $\sigma$")
    ax.set_ylabel(r"chain factor $\partial\log\sigma/\partial u$")
    ax.grid(True, which="both", axis="both")
    ax.legend(frameon=False, loc="lower left")
    ax.set_title("c  Raw-scale attenuation", loc="left", fontweight="bold")

    # (d) Influence of the prediction residual. All curves are sigma*||grad_mu||.
    ax = axes[3]
    d, nu = 56, 14.0
    radius = np.linspace(0.0, 20.0, 161)
    gaussian = radius / d
    student = (nu + d) * radius / (d * (nu + radius**2))
    # Deterministic MC of the expected unit direction for z = radius * e_1.
    rng = np.random.default_rng(20260908)
    count = 40_000
    base = rng.normal(size=(count, d)) / np.sqrt(
        rng.chisquare(nu, size=(count, 1)) / nu
    )
    energy = []
    kap = kappa(d, nu)
    for start in range(0, len(radius), 8):
        rr = radius[start:start + 8]
        diff = -base[None, :, :]
        diff = np.broadcast_to(diff, (len(rr), count, d)).copy()
        diff[:, :, 0] += rr[:, None]
        unit_mean = (diff / np.linalg.norm(diff, axis=-1, keepdims=True)).mean(axis=1)
        energy.extend((2.0 / kap * np.linalg.norm(unit_mean, axis=1)).tolist())
    ax.plot(radius, gaussian, color=gray, lw=2.0, label="Gaussian NLL")
    ax.plot(radius, student, color=blue, lw=2.2, label=r"Student-$t$ NLL")
    ax.plot(radius, energy, color=orange, lw=2.2, label="scaled energy")
    ax.set_xlabel(r"standardized residual $\|r\|/\sigma$")
    ax.set_ylabel(r"mean influence $\sigma\|\nabla_\mu\ell\|$")
    ax.grid(True, axis="both")
    ax.legend(frameon=False, loc="upper left")
    ax.set_title("d  Tail influence", loc="left", fontweight="bold")

    fig.tight_layout(w_pad=2.0)
    fig.savefig(OUT / "sigma_gradient_diagnostics.pdf", bbox_inches="tight", pad_inches=0.025)
    fig.savefig(OUT / "sigma_gradient_diagnostics.png", dpi=240,
                bbox_inches="tight", pad_inches=0.025)


if __name__ == "__main__":
    main()
