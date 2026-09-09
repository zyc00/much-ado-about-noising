#!/usr/bin/env python3
"""Plot matched gradient allocation across residual scales for the paper."""

import json
from pathlib import Path

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np


OUT_DIR = Path("analysis/paper/gradient_distribution")
ROBOMIMIC_RESULTS = OUT_DIR / "robomimic_gradient_probes.json"

# Percentages from the converged-checkpoint E8 tables in
# analysis/paper/vla_results_table.md (E8a for GR1, E8b-corrected for pi0.5).
# Rows give data frequency and the share of total squared output-gradient mass
# assigned by each objective on the same samples.
GR1 = {
    "Samples": [0.0, 75.9, 23.4, 0.6],
    "MSE": [0.0, 57.2, 40.0, 2.8],
    "Flow": [0.0, 70.7, 28.0, 1.2],
    "HT": [0.0, 72.4, 27.0, 0.6],
}

PI05 = {
    # corrected probe 2026-09-06 (resid_flow_pi05_v2.py on PFS): Gaussian x0 in the 8 (t, x0)
    # loss draws and the deployed stochastic 10-step sampler for the residual; 600 samples.
    # The >4x bin holds a single sample (0.2%) and is not drawn.
    "Samples": [24.8, 39.8, 32.7, 2.5, 0.2],
    "MSE": [3.2, 22.3, 55.7, 14.5, 4.2],
    "Flow": [26.5, 38.5, 33.2, 1.6, 0.1],
    "HT": [8.4, 40.0, 48.6, 3.0, 0.1],
}

BIN_LABELS = [
    r"$[0,0.5)$",
    r"$[0.5,1)$",
    r"$[1,2)$",
    r"$[2,4)$",
    r"$[4,\infty)$",
]
METHODS = ("MSE", "Flow", "HT")
# Okabe--Ito-inspired, colorblind-safe palette. Marker shape preserves the
# distinction when the paper is printed in grayscale.
METHOD_COLORS = {"MSE": "#D55E00", "Flow": "#0072B2", "HT": "#009E73"}
METHOD_MARKERS = {"MSE": "o", "Flow": "s", "HT": "D"}


def configure_matplotlib() -> None:
    mpl.rcParams.update(
        {
            "font.family": "serif",
            "font.serif": ["Nimbus Roman", "Times New Roman", "Times", "DejaVu Serif"],
            "mathtext.fontset": "stix",
            "font.size": 7.6,
            "axes.titlesize": 8.5,
            "axes.labelsize": 7.9,
            "xtick.labelsize": 6.2,
            "ytick.labelsize": 6.9,
            "legend.fontsize": 7.3,
            "axes.linewidth": 0.8,
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
            "savefig.bbox": "tight",
            "savefig.pad_inches": 0.025,
        }
    )


def load_robomimic(task: str) -> dict[str, list[float]]:
    payload = json.loads(ROBOMIMIC_RESULTS.read_text())
    result = payload["tasks"][task]
    return {
        "Samples": result["sample_percent"],
        "MSE": result["mse_gradient_percent"],
        "Flow": result["flow_gradient_percent"],
        "HT": result["ht_gradient_percent"],
    }


def amplification_panel(
    ax: plt.Axes,
    data: dict[str, list[float]],
    panel: str,
    title: str,
    visible_bins: list[int],
) -> None:
    """Plot gradient-share/data-share for ordered residual-size bins."""
    sample_share = np.asarray(data["Samples"], dtype=float)[visible_bins]
    x = np.arange(len(visible_bins), dtype=float)

    for method in METHODS:
        grad_share = np.asarray(data[method], dtype=float)[visible_bins]
        amplification = np.divide(
            grad_share,
            sample_share,
            out=np.full_like(grad_share, np.nan),
            where=sample_share > 0,
        )
        ax.plot(
            x,
            amplification,
            color=METHOD_COLORS[method],
            marker=METHOD_MARKERS[method],
            markersize=5.1,
            markeredgecolor="white",
            markeredgewidth=0.55,
            linewidth=1.8,
            linestyle="-",
            label=method,
            zorder=3,
        )

        # Annotate only MSE at the tail. The Flow and HT endpoints remain easy
        # to read against the shared log-scale ticks, without crowding a 1x4 row.
        if method == "MSE":
            endpoint = amplification[-1]
            endpoint_text = f"{endpoint:.0f}$\\times$" if endpoint >= 10 else f"{endpoint:.1f}$\\times$"
            ax.annotate(
                endpoint_text,
                (x[-1], endpoint),
                xytext=(-2, 8),
                textcoords="offset points",
                ha="center",
                va="bottom",
                fontsize=6.7,
                fontweight="bold",
                color=METHOD_COLORS[method],
                clip_on=False,
            )

    ax.axhline(1.0, color="0.48", linestyle=(0, (3, 2)), linewidth=0.85, zorder=1)

    ax.set_yscale("log", base=2)
    ax.set_ylim(0.125, 64)
    ticks = [0.125, 0.25, 0.5, 1, 2, 4, 8, 16, 32, 64]
    ax.set_yticks(ticks, ["0.125", "0.25", "0.5", "1", "2", "4", "8", "16", "32", "64"])
    ax.set_xlim(-0.18, len(visible_bins) - 0.82)
    ax.set_xticks(x, [BIN_LABELS[i] for i in visible_bins])
    ax.set_title(rf"$\bf{{{panel}}}$  {title}", loc="left", fontweight="bold", pad=4)
    ax.grid(axis="y", which="major", color="0.89", linewidth=0.58, zorder=0)
    ax.tick_params(axis="x", length=0, pad=3)
    ax.tick_params(axis="y", length=2.5, pad=2)
    ax.spines[["top", "right"]].set_visible(False)


def validate_table(name: str, table: dict[str, list[float]]) -> None:
    for row_name, values in table.items():
        total = sum(values)
        if not np.isclose(total, 100.0, atol=0.15):
            raise ValueError(f"{name}/{row_name} sums to {total:.2f}, not 100%")


def main() -> None:
    configure_matplotlib()
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    tool_hang = load_robomimic("Tool-Hang")
    square_ph = load_robomimic("Square-PH")
    for name, table in (
        ("GR1", GR1),
        ("pi0.5", PI05),
        ("Tool-Hang", tool_hang),
        ("Square-PH", square_ph),
    ):
        validate_table(name, table)

    fig, axes = plt.subplots(1, 4, figsize=(7.08, 2.28), sharey=True)
    amplification_panel(axes[0], GR1, "a", "GR1 (GR00T)", [1, 2, 3])
    amplification_panel(axes[1], PI05, "b", r"LIBERO ($\pi_{0.5}$)", [0, 1, 2, 3])
    amplification_panel(axes[2], tool_hang, "c", "Tool-Hang", [0, 1, 2, 3, 4])
    amplification_panel(axes[3], square_ph, "d", "Square-PH", [0, 1, 2, 3, 4])
    axes[0].set_ylabel("Gradient amplification ratio", labelpad=3)

    handles = [
        mpl.lines.Line2D(
            [],
            [],
            color=METHOD_COLORS[method],
            marker=METHOD_MARKERS[method],
            markeredgecolor="white",
            linewidth=1.8,
            linestyle="-",
            label=method,
        )
        for method in METHODS
    ]
    fig.legend(
        handles=handles,
        loc="upper center",
        bbox_to_anchor=(0.52, 1.01),
        ncol=3,
        frameon=False,
        columnspacing=1.8,
        handlelength=2.1,
        handletextpad=0.5,
    )
    fig.supxlabel("Normalized residual magnitude", y=0.065, fontsize=8.1)
    fig.subplots_adjust(left=0.083, right=0.995, bottom=0.24, top=0.81, wspace=0.24)

    png_path = OUT_DIR / "fig_gradient_distribution.png"
    pdf_path = OUT_DIR / "fig_gradient_distribution.pdf"
    fig.savefig(png_path, dpi=300)
    fig.savefig(pdf_path)
    plt.close(fig)
    print(f"saved {png_path}")
    print(f"saved {pdf_path}")


if __name__ == "__main__":
    main()
