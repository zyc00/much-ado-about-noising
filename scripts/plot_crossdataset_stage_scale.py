#!/usr/bin/env python3
"""ICLR-ready evidence for input- and stage-dependent regression scale."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib.ticker import NullLocator
import numpy as np


MSE_SETTINGS = [
    ("gr1", "GR1 / RoboCasa", "#CC79A7"),
    ("pi05", r"$\pi_{0.5}$ / LIBERO", "#009E73"),
    ("tool_hang", "Tool-Hang", "#0072B2"),
    ("transport", "Transport", "#56B4E9"),
]
CROSS_COLORS = ["#0072B2", "#009E73", "#CC79A7", "#D55E00", "#E69F00"]


def configure() -> None:
    mpl.rcParams.update({
        "font.family": "serif",
        "font.serif": ["Times New Roman", "Times", "DejaVu Serif"],
        "mathtext.fontset": "stix",
        "font.size": 9.0,
        "axes.titlesize": 10.0,
        "axes.labelsize": 9.2,
        "xtick.labelsize": 8.2,
        "ytick.labelsize": 8.2,
        "axes.linewidth": 0.75,
        "pdf.fonttype": 42,
        "ps.fonttype": 42,
    })


def weighted_quantile(values: np.ndarray, weights: np.ndarray, probabilities) -> np.ndarray:
    order = np.argsort(values)
    values = values[order]
    weights = weights[order]
    cumulative = np.cumsum(weights) - 0.5 * weights
    cumulative /= weights.sum()
    return np.interp(probabilities, cumulative, values)


def load_mse_residuals(path: Path) -> dict:
    archive = np.load(path, allow_pickle=True)
    metadata = json.loads(str(archive["metadata"].item()))
    start = int(metadata["executed_start"])
    horizon = min(8, int(metadata["executed_horizon"]))
    channels = np.asarray(metadata["continuous_channels"], dtype=int)
    valid = archive["valid_time"][:, start : start + horizon].all(axis=1)
    residual = (
        archive["pred_final"][valid, start : start + horizon][:, :, channels]
        - archive["gt"][valid, start : start + horizon][:, :, channels]
    ).astype(np.float64)
    rms = np.sqrt(np.mean(residual ** 2, axis=(1, 2)))
    tasks = archive["task"][valid].astype(str)
    relative = np.empty_like(rms)
    weights = np.empty_like(rms)
    names = np.unique(tasks)
    for task in names:
        keep = tasks == task
        relative[keep] = rms[keep] / np.median(rms[keep])
        weights[keep] = 1.0 / (len(names) * keep.sum())
    quantiles = weighted_quantile(relative, weights, [0.01, 0.1, 0.5, 0.9, 0.99])
    return {
        "relative": relative,
        "weights": weights,
        "quantiles": quantiles,
        "central_ratio": float(quantiles[3] / quantiles[1]),
        "full_ratio": float(relative.max() / relative.min()),
    }


def draw_residual_spectra(axis, mse_root: Path) -> None:
    datasets = [(key, label, color, load_mse_residuals(mse_root / f"{key}.npz"))
                for key, label, color in MSE_SETTINGS]
    edges = np.geomspace(0.15, 18.0, 55)
    bases = np.arange(len(datasets))[::-1]
    for base, (_, _, color, data) in zip(bases, datasets):
        histogram, _ = np.histogram(
            data["relative"], bins=edges, weights=data["weights"]
        )
        height = 0.58 * histogram / histogram.max()
        axis.stairs(base + height, edges, baseline=base, fill=True,
                    color=color, alpha=0.30, linewidth=0)
        axis.stairs(base + height, edges, baseline=base,
                    color=color, linewidth=1.05)
        q01, q10, median, q90, q99 = data["quantiles"]
        axis.plot([data["relative"].min(), data["relative"].max()],
                  [base - 0.10] * 2, color="#B6B6B6", linewidth=0.8)
        axis.plot([q10, q90], [base - 0.10] * 2, color=color, linewidth=3.0,
                  solid_capstyle="butt")
        axis.scatter([median], [base - 0.10], s=18, color="white",
                     edgecolor="#202020", linewidth=0.75, zorder=5)
        axis.text(
            17.3, base + 0.20,
            f"{data['central_ratio']:.1f}$\\times$  ({data['full_ratio']:.0f}$\\times$)",
            ha="right", va="center", fontsize=7.6, color="#282828"
        )
    axis.axvline(1.0, color="#777777", linewidth=0.8,
                 linestyle=(0, (3, 2)), zorder=0)
    axis.set_xscale("log", base=2)
    axis.set_xlim(edges[0], edges[-1])
    axis.set_ylim(-0.35, 3.82)
    axis.set_xticks([0.25, 0.5, 1, 2, 4, 8, 16],
                    ["0.25", "0.5", "1", "2", "4", "8", "16"])
    axis.xaxis.set_minor_locator(NullLocator())
    axis.set_yticks(bases + 0.20, [label for _, label, _ in MSE_SETTINGS])
    axis.tick_params(axis="y", length=0, pad=5)
    axis.set_xlabel("Per-input residual RMS / task median")
    axis.set_title("a   Residual scale varies across inputs", loc="left", fontweight="bold")
    axis.text(
        17.3, 3.68, "10--90%  (full range)", ha="right", va="bottom",
        fontsize=7.0, color="#555555"
    )


def draw_cross_dataset_validation(axis, summary: dict) -> None:
    datasets = summary["datasets"]
    positions = np.arange(len(datasets))[::-1]
    labels = {
        "RoboMimic": "RoboMimic (5)",
        "LIBERO": "LIBERO (40)",
        "RoboCasa / GR1": "RoboCasa / GR1 (24)",
        "Bridge / WidowX": "Bridge / WidowX (40)",
        "Fractal": "Fractal (40)",
    }
    for y, dataset, color in zip(positions, datasets, CROSS_COLORS):
        rms_ratio = float(dataset["q5_over_q1_heldout_rms"])
        variance_ratio = rms_ratio ** 2
        lo, hi = np.asarray(dataset["task_bootstrap_95_ci"], dtype=float) ** 2
        axis.plot([lo, hi], [y, y], color=color, linewidth=2.0,
                  solid_capstyle="round")
        axis.scatter([variance_ratio], [y], s=38, color=color, edgecolor="white",
                     linewidth=0.75, zorder=4)
        axis.text(hi * 1.055, y, f"{variance_ratio:.1f}$\\times$", color=color,
                  va="center", ha="left", fontsize=8.0)
    axis.axvline(1.0, color="#777777", linewidth=0.9,
                 linestyle=(0, (3, 2)), zorder=0)
    axis.set_xscale("log", base=2)
    axis.set_xlim(0.92, 10.2)
    axis.set_ylim(-0.45, len(datasets) - 0.45)
    axis.set_xticks([1, 2, 4, 8], ["1", "2", "4", "8"])
    axis.xaxis.set_minor_locator(NullLocator())
    axis.set_yticks(positions, [labels[item["dataset"]] for item in datasets])
    axis.tick_params(axis="y", length=0, pad=5)
    axis.set_xlabel("Variance ratio (high / low stage)")
    axis.set_title("b   Held-out stage variance", loc="left", fontweight="bold")
    axis.text(1.03, len(datasets) - 0.72, "95% task-bootstrap CI",
              fontsize=7.0, color="#555555", va="top")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--mse-root", type=Path, default=Path("analysis/paper/mse_scale"))
    parser.add_argument(
        "--summary", type=Path,
        default=Path("analysis/paper/data_ht_motivation/cross_dataset/cross_dataset_scale_summary.json")
    )
    parser.add_argument(
        "--out", type=Path,
        default=Path("analysis/paper/data_ht_motivation/fig_crossdataset_stage_scale")
    )
    args = parser.parse_args()
    configure()
    summary = json.loads(args.summary.read_text())

    figure = plt.figure(figsize=(7.05, 2.70))
    left = figure.add_axes([0.15, 0.22, 0.36, 0.63])
    right = figure.add_axes([0.70, 0.22, 0.285, 0.63])
    draw_residual_spectra(left, args.mse_root)
    draw_cross_dataset_validation(right, summary)
    for axis in (left, right):
        axis.grid(axis="x", color="#D9D9D9", linewidth=0.6, zorder=0)
        axis.spines["top"].set_visible(False)
        axis.spines["right"].set_visible(False)
        axis.tick_params(axis="x", length=3, width=0.75)
    for suffix in ("pdf", "png"):
        figure.savefig(args.out.with_suffix(f".{suffix}"),
                       dpi=320 if suffix == "png" else None, pad_inches=0.02)
    plt.close(figure)


if __name__ == "__main__":
    main()
