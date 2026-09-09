#!/usr/bin/env python3
"""Plot the held-out action-magnitude specialist probe."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np


BIN_COLORS = {0: "#2C7BB6", 1: "#74ADD1", 2: "#A8A8A8", 3: "#F4A582", 4: "#D6604D"}
STACK_STYLE = {
    "gr1": ("GR00T / GR1", "#6A51A3", "s"),
    "pi05": (r"$\pi_{0.5}$ / LIBERO", "#009E73", "^"),
    "widowx": ("GR00T / WidowX", "#D55E00", "o"),
}
TAIL_GAUSSIAN = "#2C7BB6"
TAIL_STUDENT = "#D55E00"
TAIL_EMPIRICAL = "#222222"


def configure() -> None:
    mpl.rcParams.update(
        {
            "font.family": "serif",
            "font.serif": ["Nimbus Roman", "Times New Roman", "Times", "DejaVu Serif"],
            "mathtext.fontset": "stix",
            "font.size": 8.5,
            "axes.titlesize": 9.2,
            "axes.labelsize": 8.6,
            "xtick.labelsize": 8.0,
            "ytick.labelsize": 8.0,
            "legend.fontsize": 7.6,
            "axes.linewidth": 0.85,
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
            "savefig.bbox": "tight",
            "savefig.pad_inches": 0.025,
        }
    )


def load_result(path: Path) -> dict:
    with np.load(path) as data:
        return {key: data[key] for key in data.files if key != "metadata"}


def rms(residual: np.ndarray) -> float:
    return float(np.sqrt(np.mean(np.square(residual, dtype=np.float64))))


def cluster_bootstrap_rms(data: dict, stack: str, seed: int, draws: int = 2000) -> np.ndarray:
    residual = np.asarray(data["residual"], dtype=np.float64)
    clusters = np.asarray(data["episode"] if stack == "widowx" else data["task"])
    unique, inverse = np.unique(clusters, return_inverse=True)
    sample_ss = np.sum(np.square(residual), axis=tuple(range(1, residual.ndim)))
    sample_n = np.full(len(residual), np.prod(residual.shape[1:]), dtype=np.int64)
    cluster_ss = np.bincount(inverse, weights=sample_ss, minlength=len(unique))
    cluster_n = np.bincount(inverse, weights=sample_n, minlength=len(unique))
    rng = np.random.default_rng(seed)
    output = np.empty(draws, dtype=np.float64)
    for start in range(0, draws, 200):
        count = min(200, draws - start)
        sampled = rng.integers(0, len(unique), size=(count, len(unique)))
        output[start : start + count] = np.sqrt(
            cluster_ss[sampled].sum(axis=1) / cluster_n[sampled].sum(axis=1)
        )
    return output


def available(results: Path, stack: str) -> dict[int, dict]:
    found = {}
    for q in range(5):
        path = results / f"{stack}_q{q}.npz"
        if path.exists():
            found[q] = load_result(path)
    return found


def plot_density_panel(ax: plt.Axes, widowx: dict[int, dict]) -> dict:
    selected = [0, 2, 4]
    missing = [q for q in selected if q not in widowx]
    if missing:
        raise FileNotFoundError(f"WidowX density panel is missing bins {missing}")

    centered = {}
    sigmas = {}
    for q in selected:
        residual = np.asarray(widowx[q]["residual"], dtype=np.float64)
        residual = residual - residual.mean(axis=0, keepdims=True)
        values = residual.reshape(-1)
        centered[q] = values
        sigmas[q] = float(np.sqrt(np.mean(values**2)))

    extent = max(np.quantile(np.abs(centered[q]), 0.995) for q in selected)
    bins = np.linspace(-extent, extent, 95)
    x = np.linspace(-extent, extent, 500)
    for q in selected:
        color = BIN_COLORS[q]
        values = centered[q]
        ax.hist(
            values,
            bins=bins,
            density=True,
            histtype="stepfilled",
            color=color,
            alpha=0.15,
            linewidth=0,
        )
        sigma = sigmas[q]
        gaussian = np.exp(-0.5 * np.square(x / sigma)) / (np.sqrt(2 * np.pi) * sigma)
        ax.plot(
            x,
            gaussian,
            color=color,
            linewidth=2.1,
            label=rf"Q{q + 1}: $\hat{{\sigma}}={sigma:.3f}$",
        )

    ax.axvline(0, color="0.35", linewidth=0.8, linestyle=(0, (2, 2)), zorder=0)
    ax.set_title(r"a  Residual scale varies", loc="left", fontweight="bold")
    ax.text(
        0.02,
        0.98,
        "BridgeData V2 / WidowX (all tasks pooled)",
        transform=ax.transAxes,
        ha="left",
        va="top",
        fontsize=7.1,
        color="0.30",
    )
    ax.set_xlabel("Centered continuous residual")
    ax.set_ylabel("Density")
    ax.set_yticks([])
    ax.set_xlim(-extent, extent)
    ax.grid(axis="x", color="0.92", linewidth=0.6, zorder=0)
    ax.spines[["top", "right", "left"]].set_visible(False)
    ax.legend(frameon=False, loc="upper left", bbox_to_anchor=(0.0, 0.77), handlelength=2.0)
    ratio = rms(widowx[4]["residual"]) / rms(widowx[0]["residual"])
    ax.text(
        0.98,
        0.88,
        rf"Q5 / Q1 = {ratio:.2f}$\times$",
        transform=ax.transAxes,
        ha="right",
        va="top",
        color="#9B2F22",
        fontsize=8.0,
        fontweight="bold",
    )
    return {"centered_sigma": {f"q{q + 1}": sigmas[q] for q in selected}, "ratio": ratio}


def plot_cross_stack_panel(ax: plt.Axes, stacks: dict[str, dict[int, dict]]) -> dict:
    summary = {}
    max_upper = 1.0
    for stack in ("gr1", "pi05", "widowx"):
        values = stacks.get(stack, {})
        if 0 not in values or len(values) < 2:
            continue
        label, color, marker = STACK_STYLE[stack]
        base = rms(values[0]["residual"])
        base_boot = cluster_bootstrap_rms(values[0], stack, seed=1000)
        qs, ratios, lower, upper = [], [], [], []
        stack_rows = []
        for q, data in sorted(values.items()):
            value = rms(data["residual"])
            boot = cluster_bootstrap_rms(data, stack, seed=1000 + 17 * q)
            ratio_boot = boot / base_boot
            ratio = value / base
            lo, hi = np.quantile(ratio_boot, [0.025, 0.975])
            if q == 0:
                lo = hi = 1.0
            qs.append(q + 1)
            ratios.append(ratio)
            lower.append(ratio - lo)
            upper.append(hi - ratio)
            max_upper = max(max_upper, hi)
            stack_rows.append(
                {
                    "quintile": q + 1,
                    "rms": value,
                    "relative_to_q1": ratio,
                    "ci95": [float(lo), float(hi)],
                    "samples": int(len(data["residual"])),
                }
            )
        ax.errorbar(
            qs,
            ratios,
            yerr=np.asarray([lower, upper]),
            color=color,
            marker=marker,
            markersize=5.8,
            markeredgecolor="white",
            markeredgewidth=0.65,
            linewidth=2.0,
            capsize=2.2,
            label=label,
            zorder=3,
        )
        if 4 in values:
            ratio = rms(values[4]["residual"]) / base
            ax.text(5.10, ratio, f"{ratio:.2f}$\\times$", color=color, va="center", fontsize=7.8)
        summary[stack] = stack_rows

    ax.axhline(1, color="0.55", linewidth=0.9, linestyle=(0, (3, 2)), zorder=0)
    ax.set_title("b  Scale variation repeats", loc="left", fontweight="bold")
    ax.text(
        0.02,
        0.96,
        r"split by $\Vert a\Vert$ $\rightarrow$ fit specialists $\rightarrow$ test held out",
        transform=ax.transAxes,
        ha="left",
        va="top",
        fontsize=6.8,
        color="0.28",
    )
    ax.set_xlabel("Action-magnitude quintile")
    ax.set_ylabel("Held-out RMS / Q1")
    ax.set_xticks(range(1, 6), [f"Q{q}" for q in range(1, 6)])
    ax.set_xlim(0.75, 5.58)
    ax.set_ylim(0.8, max(3.0, max_upper * 1.15))
    ax.grid(axis="y", color="0.90", linewidth=0.6, zorder=0)
    ax.spines[["top", "right"]].set_visible(False)
    ax.legend(frameon=False, loc="upper left", bbox_to_anchor=(0.0, 0.86))
    return summary


def plot_tail_panel(ax: plt.Axes, tail_summary: dict) -> dict:
    """Held-out survival plot after local- and coordinate-scale normalization."""
    tail = tail_summary["tail_plot"]
    fit = tail_summary["coordinate_controlled_student_fit"][0]
    x = np.asarray(tail["grid"], dtype=np.float64)
    empirical = np.asarray(tail["empirical"], dtype=np.float64)
    gaussian = np.asarray(tail["gaussian"], dtype=np.float64)
    student = np.asarray(tail["student"], dtype=np.float64)
    at_three = tail["threshold_survival"]["3"]

    ax.semilogy(
        x,
        gaussian,
        color=TAIL_GAUSSIAN,
        linestyle=(0, (4, 2.4)),
        linewidth=1.8,
        label="Gaussian",
        zorder=1,
    )
    ax.semilogy(
        x,
        student,
        color=TAIL_STUDENT,
        linewidth=2.0,
        label=rf"Student-$t$ ($\nu={fit['student_df']:.1f}$)",
        zorder=2,
    )
    ids = np.arange(0, len(x), 5)
    ax.semilogy(x, empirical, color=TAIL_EMPIRICAL, linewidth=1.4, zorder=3)
    ax.semilogy(
        x[ids],
        empirical[ids],
        linestyle="none",
        marker="o",
        markerfacecolor="white",
        markeredgecolor=TAIL_EMPIRICAL,
        markeredgewidth=0.75,
        markersize=3.2,
        label="Held-out data",
        zorder=4,
    )

    ax.axvline(3, color="0.48", linestyle=(0, (1.5, 2.2)), linewidth=0.9, zorder=0)
    ax.text(
        3.12,
        7.2e-2,
        rf"$3.6\times$ more $3\sigma$ events"
        "\n" + rf"{100 * at_three['empirical']:.2f}\% data vs. "
        + rf"{100 * at_three['gaussian']:.2f}\% Gaussian",
        ha="left",
        va="top",
        fontsize=7.4,
        color="#8F2D20",
        fontweight="bold",
    )
    ax.text(
        0.98,
        0.04,
        rf"held-out NLL $\downarrow$ {fit['heldout_student_nll_gain']:.3f} nat / dim",
        transform=ax.transAxes,
        ha="right",
        va="bottom",
        fontsize=7.2,
        color="0.25",
    )
    ax.set_title("c  Residuals stay heavy-tailed", loc="left", fontweight="bold")
    ax.text(
        0.02,
        0.98,
        "RoboMimic Tool-Hang (human demonstrations)",
        transform=ax.transAxes,
        ha="left",
        va="top",
        fontsize=7.1,
        color="0.30",
    )
    ax.set_xlabel(r"Standardized residual $|z|$")
    ax.set_ylabel("Two-sided tail probability")
    ax.set_xlim(0.55, 6.25)
    ax.set_ylim(1e-5, 0.55)
    ax.set_yticks([1e-1, 1e-2, 1e-3, 1e-4, 1e-5])
    ax.grid(axis="y", color="0.90", linewidth=0.6, zorder=0)
    ax.spines[["top", "right"]].set_visible(False)
    handles, labels = ax.get_legend_handles_labels()
    order = [2, 0, 1]
    ax.legend(
        [handles[i] for i in order],
        [labels[i] for i in order],
        frameon=False,
        loc="lower left",
        bbox_to_anchor=(0.0, 0.08),
        handlelength=2.2,
    )
    return {
        "dataset": "RoboMimic Tool-Hang human demonstrations",
        "student_df": fit["student_df"],
        "heldout_student_nll_gain": fit["heldout_student_nll_gain"],
        "p_abs_z_gt_3_data": at_three["empirical"],
        "p_abs_z_gt_3_gaussian": at_three["gaussian"],
        "three_sigma_ratio": at_three["empirical_over_gaussian"],
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--results", type=Path, required=True)
    parser.add_argument("--tail-summary", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    configure()
    args.output.mkdir(parents=True, exist_ok=True)

    stacks = {stack: available(args.results, stack) for stack in STACK_STYLE}
    with args.tail_summary.open() as handle:
        tail_summary = json.load(handle)
    fig, (ax_a, ax_b, ax_c) = plt.subplots(
        1,
        3,
        figsize=(7.0, 2.55),
        gridspec_kw={"width_ratios": [1.12, 1.0, 1.05], "wspace": 0.41},
    )
    density_summary = plot_density_panel(ax_a, stacks["widowx"])
    cross_summary = plot_cross_stack_panel(ax_b, stacks)
    tail_panel_summary = plot_tail_panel(ax_c, tail_summary)
    fig.subplots_adjust(left=0.065, right=0.995, bottom=0.21, top=0.88)

    for suffix in ("png", "pdf"):
        path = args.output / f"fig_split_fit_scale.{suffix}"
        fig.savefig(path, dpi=300 if suffix == "png" else None)
        print(f"saved {path}")
    plt.close(fig)
    (args.output / "summary.json").write_text(
        json.dumps(
            {
                "density_panel": density_summary,
                "cross_stack": cross_summary,
                "tail_panel": tail_panel_summary,
            },
            indent=2,
        )
        + "\n"
    )


if __name__ == "__main__":
    main()
