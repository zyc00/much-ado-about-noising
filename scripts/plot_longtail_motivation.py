#!/usr/bin/env python3
"""Paper figure: chunk-level heavy tails after state-local scale normalization.

The primary two-panel figure compares (a) an HG policy residual and (b) the
spread of *every* Flow sample around its state-wise sample mean.  The latter
uses one 48-D norm per sample, rather than an RMS aggregated over samples.
MSE residuals and Flow-to-demonstration errors are rendered separately as
diagnostics, so the main figure carries one simple message.
"""
from __future__ import annotations

import glob
import json
from pathlib import Path

import matplotlib as mpl
mpl.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
import numpy as np
from scipy.optimize import minimize, minimize_scalar
from scipy.special import gammaln
from scipy.stats import chi, f

ROOT = Path("analysis/paper/widowx_heterogeneous_scale/raw")
OUT = Path("analysis/paper/longtail_motivation")
D = 48  # eight action steps times six continuous WidowX channels
COL = {"data": "#242424", "gauss": "#0072B2", "t": "#D55E00", "grey": "#7A7A7A"}


def load(mode: str) -> dict[str, np.ndarray]:
    parts: dict[str, list[np.ndarray]] = {}
    for file in sorted(glob.glob(str(ROOT / f"{mode}_rank*.npz"))):
        with np.load(file, allow_pickle=False) as archive:
            for key in archive.files:
                if key != "metadata":
                    parts.setdefault(key, []).append(archive[key])
    data = {key: np.concatenate(values) for key, values in parts.items()}
    order = np.lexsort((data["step"], data["episode"], data["task_id"]))
    return {key: value[order] for key, value in data.items()}


def flatten_chunks(values: np.ndarray) -> np.ndarray:
    """Keep each action chunk (or each Flow draw) as one 48-D observation."""
    return values.reshape(-1, D).astype(np.float64)


def split_standardize(values: np.ndarray, episode_ids: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Coordinate normalization estimated on even episodes, then applied to all."""
    vectors = flatten_chunks(values)
    ids = np.repeat(episode_ids, values.shape[1]) if values.ndim == 4 else episode_ids
    train = (ids % 2) == 0
    coordinate_rms = np.sqrt(np.mean(vectors[train] ** 2, axis=0, keepdims=True))
    return vectors / coordinate_rms, train


def gaussian_nll(vectors: np.ndarray, log_scale: float) -> np.ndarray:
    scale = np.exp(log_scale)
    sq = np.sum(vectors ** 2, axis=1)
    return 0.5 * D * np.log(2 * np.pi * scale ** 2) + sq / (2 * scale ** 2)


def student_nll(vectors: np.ndarray, params: np.ndarray) -> np.ndarray:
    # Multivariate isotropic Student-t, parameterized with nu > 2.
    nu = 2.0 + np.exp(params[0])
    scale = np.exp(params[1])
    sq = np.sum(vectors ** 2, axis=1)
    logp = (
        gammaln((nu + D) / 2) - gammaln(nu / 2)
        - 0.5 * D * np.log(nu * np.pi) - D * np.log(scale)
        - 0.5 * (nu + D) * np.log1p(sq / (nu * scale ** 2))
    )
    return -logp


def fit_crossfit(vectors: np.ndarray, train: np.ndarray) -> dict[str, float]:
    log_scale = 0.5 * np.log(np.mean(vectors[train] ** 2))
    fit = minimize(
        lambda x: float(np.mean(student_nll(vectors[train], x))),
        np.array([np.log(8.0), log_scale]), method="Nelder-Mead",
        options={"maxiter": 2000},
    )
    gaussian = float(np.mean(gaussian_nll(vectors[~train], log_scale)) / D)
    student = float(np.mean(student_nll(vectors[~train], fit.x)) / D)
    return {
        "nu_mle": float(2.0 + np.exp(fit.x[0])),
        "scale_mle": float(np.exp(fit.x[1])),
        "heldout_gaussian_nll_per_dim": gaussian,
        "heldout_student_nll_per_dim": student,
        "heldout_nll_gain_per_dim": gaussian - student,
    }


def tail_fit(radii: np.ndarray) -> tuple[float, float]:
    """A radial t fit for a readable survival-curve overlay (not the NLL fit)."""
    grid = np.linspace(5.7, 14.5, 80)
    empirical = np.array([(radii > radius).mean() for radius in grid])
    keep = empirical > 1.0 / len(radii)

    def objective(x: np.ndarray) -> float:
        nu = 1.2 + np.exp(x[0])
        scale = np.exp(x[1])
        model = f.sf(grid[keep] ** 2 / (D * scale ** 2), D, nu)
        return float(np.mean((np.log(empirical[keep]) - np.log(model)) ** 2))

    result = minimize(objective, np.array([np.log(6.0), np.log(0.85)]), method="Nelder-Mead")
    return float(1.2 + np.exp(result.x[0])), float(np.exp(result.x[1]))


def summarize(vectors: np.ndarray, train: np.ndarray) -> dict[str, object]:
    radii = np.linalg.norm(vectors, axis=1)
    threshold = 1.4 * np.sqrt(D)
    gaussian_tail = float(chi.sf(threshold, D))
    tail_nu, tail_scale = tail_fit(radii)
    result: dict[str, object] = fit_crossfit(vectors, train)
    result.update({
        "n_vectors": int(len(radii)),
        "dimension": D,
        "threshold": float(threshold),
        "empirical_tail": float(np.mean(radii > threshold)),
        "gaussian_tail": gaussian_tail,
        "tail_ratio": float(np.mean(radii > threshold) / gaussian_tail),
        "tail_fit_nu": tail_nu,
        "tail_fit_scale": tail_scale,
        "q90": float(np.quantile(radii, .90)),
        "q99": float(np.quantile(radii, .99)),
    })
    return result


def plot_panel(ax: plt.Axes, vectors: np.ndarray, stats: dict[str, object], title: str, subtitle: str) -> None:
    radii = np.linalg.norm(vectors, axis=1)
    grid = np.linspace(4.8, 16.0, 160)
    empirical = np.array([(radii > radius).mean() for radius in grid])
    keep = empirical > 0
    fitted_t = f.sf(grid ** 2 / (D * float(stats["tail_fit_scale"]) ** 2), D, float(stats["tail_fit_nu"]))
    ax.plot(grid, chi.sf(grid, D), color=COL["gauss"], lw=2.1, ls=(0, (5, 2)), zorder=2)
    ax.plot(grid, fitted_t, color=COL["t"], lw=2.2, zorder=3)
    ax.plot(grid[keep], empirical[keep], color=COL["data"], lw=1.45, zorder=4)
    points = np.arange(0, len(grid), 9)
    points = points[empirical[points] > 0]
    ax.plot(grid[points], empirical[points], "o", ms=4.5, mfc="white", mec=COL["data"], mew=1.25, zorder=5)
    threshold = float(stats["threshold"])
    ax.axvline(threshold, color="#909090", lw=1.1, ls=":", zorder=1)
    ax.set_yscale("log")
    ax.set_xlim(4.9, 15.8)
    ax.set_ylim(2e-4, 1.1)
    ax.set_xticks([5, 7, 9, 11, 13, 15])
    ax.grid(axis="y", color="#d8d8d8", lw=.7, zorder=0)
    ax.tick_params(length=3, pad=2)
    ax.set_title(title, loc="left", fontweight="bold", pad=6)
    text = (
        rf"$P(R > 1.4\sqrt{{d}})$ = {100 * float(stats['empirical_tail']):.1f}\%"
        "\n"
        rf"vs. {100 * float(stats['gaussian_tail']):.3f}\% Gaussian"
        "\n"
        rf"cross-fit Student-$t$ NLL $\downarrow$ {float(stats['heldout_nll_gain_per_dim']):.3f} nat / dim"
    )
    ax.text(.965, .95, text, transform=ax.transAxes, ha="right", va="top", fontsize=7.1,
            color="#8b2d24", fontweight="bold",
            bbox=dict(facecolor="white", edgecolor="#c8c8c8", boxstyle="round,pad=.34", lw=.7))


def make_figure(panels: list[tuple[np.ndarray, dict[str, object], str, str]], output: Path) -> None:
    mpl.rcParams.update({
        "font.family": "serif", "font.serif": ["Nimbus Roman", "DejaVu Serif"], "mathtext.fontset": "stix",
        "font.size": 8, "axes.labelsize": 8.5, "axes.titlesize": 9.0, "xtick.labelsize": 7.2,
        "ytick.labelsize": 7.2, "axes.spines.top": False, "axes.spines.right": False,
        "axes.linewidth": .85, "pdf.fonttype": 42,
    })
    fig, axes = plt.subplots(1, len(panels), figsize=(7.2, 2.55), sharey=True)
    if len(panels) == 1:
        axes = [axes]
    for axis, (vectors, stats, title, subtitle) in zip(axes, panels):
        plot_panel(axis, vectors, stats, title, subtitle)
        axis.set_xlabel(r"normalized chunk magnitude $R = \|z\|_2$")
    axes[0].set_ylabel(r"tail probability $P(R > r)$")
    handles = [
        Line2D([0], [0], color=COL["data"], lw=1.45, marker="o", ms=4.5, mfc="white", mec=COL["data"], mew=1.25, label="observed samples"),
        Line2D([0], [0], color=COL["gauss"], lw=2.1, ls=(0, (5, 2)), label=rf"Gaussian ($\chi_{{{D}}}$)"),
        Line2D([0], [0], color=COL["t"], lw=2.2, label="Student-$t$ tail fit"),
    ]
    fig.legend(handles=handles, loc="upper center", ncol=3, frameon=False, bbox_to_anchor=(.5, 1.04), columnspacing=2.2, fontsize=8)
    fig.subplots_adjust(left=.075, right=.995, bottom=.22, top=.78, wspace=.28)
    fig.savefig(output.with_suffix(".pdf"), bbox_inches="tight", pad_inches=.025)
    fig.savefig(output.with_suffix(".png"), dpi=320, bbox_inches="tight", pad_inches=.025)


def main() -> None:
    hg, mse, flow = load("hg"), load("mse"), load("flow")
    for key in ("task_id", "episode", "step"):
        np.testing.assert_array_equal(hg[key], mse[key])
        np.testing.assert_array_equal(hg[key], flow[key])
    sigma = hg["sigma"][:, None, None]
    # sigma(x) comes from the independently trained HG scale head.  It removes
    # input-dependent scale before testing the remaining radial residual shape.
    hg_vectors, hg_train = split_standardize(hg["residual"] / sigma, hg["episode"])
    mse_vectors, mse_train = split_standardize(mse["residual"] / sigma, mse["episode"])
    flow_mean_vectors, flow_mean_train = split_standardize(
        (flow["samples"] - flow["samples"].mean(axis=1, keepdims=True)) / sigma[:, None], flow["episode"]
    )
    flow_label_vectors, flow_label_train = split_standardize(
        (flow["samples"] - flow["target"][:, None]) / sigma[:, None], flow["episode"]
    )
    stats = {
        "hg_residual": summarize(hg_vectors, hg_train),
        "mse_residual": summarize(mse_vectors, mse_train),
        "flow_sample_spread": summarize(flow_mean_vectors, flow_mean_train),
        "flow_to_label": summarize(flow_label_vectors, flow_label_train),
    }
    OUT.mkdir(parents=True, exist_ok=True)
    make_figure([
        (hg_vectors, stats["hg_residual"], r"a  HG normalized residual", r"BridgeData V2 / WidowX; one 48-D chunk per state"),
        (mse_vectors, stats["mse_residual"], r"b  MSE normalized residual", r"Same states and state-local normalization"),
        (flow_mean_vectors, stats["flow_sample_spread"], r"c  Flow sample spread", r"16 draws/state; each $\|a_k - \bar a\|_2$ counted separately"),
    ], OUT / "fig_longtail_motivation")
    make_figure([
        (mse_vectors, stats["mse_residual"], r"a  MSE normalized residual", r"Same states and state-local normalization"),
        (flow_label_vectors, stats["flow_to_label"], r"b  Flow-to-label error", r"Each sampled chunk compared with its demonstration label"),
    ], OUT / "fig_longtail_diagnostics")
    (OUT / "summary.json").write_text(json.dumps({"dataset": "BridgeData V2 / WidowX", "states": int(len(hg["episode"])), "stats": stats}, indent=2))


if __name__ == "__main__":
    main()
