#!/usr/bin/env python3
"""Plot measured GR00T/GR1 success rates over the 60k-step schedule."""

from pathlib import Path

import matplotlib as mpl
import matplotlib.pyplot as plt
import pandas as pd


OUT_DIR = Path("analysis/paper/convergence_curve")
RESULTS = OUT_DIR / "results.csv"
STYLE = {
    "MSE": {"color": "#D55E00", "marker": "o", "label": "MSE"},
    "Flow": {"color": "#0072B2", "marker": "s", "label": "Flow"},
    "HT": {
        "color": "#009E73",
        "marker": "D",
        "label": "HT (ours)",
    },
}


def configure_matplotlib() -> None:
    mpl.rcParams.update(
        {
            "font.family": "serif",
            "font.serif": ["Nimbus Roman", "Times New Roman", "Times", "DejaVu Serif"],
            "mathtext.fontset": "stix",
            "font.size": 8.0,
            "axes.titlesize": 9.0,
            "axes.labelsize": 8.2,
            "xtick.labelsize": 7.2,
            "ytick.labelsize": 7.2,
            "legend.fontsize": 7.2,
            "axes.linewidth": 0.8,
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
            "savefig.bbox": "tight",
            "savefig.pad_inches": 0.025,
        }
    )


def load_results() -> pd.DataFrame:
    frame = pd.read_csv(RESULTS)
    required = {
        "MSE": {2, 16, 32, 60},
        "Flow": {2, 16, 60},
        "HT (nu=2)": {2, 16, 32},
        "HT (nu=4d)": {60},
    }
    for method, steps in required.items():
        observed = set(frame.loc[frame["method"] == method, "steps_k"])
        missing = steps - observed
        if missing:
            raise ValueError(f"{method} is missing checkpoints: {sorted(missing)}")
    if not (frame["n_tasks"] == 24).all():
        raise ValueError("Every result must contain all 24 GR1 tasks")

    mse = frame[(frame["method"] == "MSE") & frame["steps_k"].isin([2, 16, 32, 60])].copy()
    flow = frame[(frame["method"] == "Flow") & frame["steps_k"].isin([2, 16, 60])].copy()
    ht_early = frame[
        (frame["method"] == "HT (nu=2)") & frame["steps_k"].isin([2, 16, 32])
    ].copy()
    ht_final = frame[
        (frame["method"] == "HT (nu=4d)") & (frame["steps_k"] == 60)
    ].copy()
    ht = pd.concat([ht_early, ht_final], ignore_index=True)
    ht["method"] = "HT"
    return pd.concat([mse, flow, ht], ignore_index=True)


def main() -> None:
    configure_matplotlib()
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    frame = load_results()

    fig, ax = plt.subplots(figsize=(3.35, 2.55))

    # Lines connect evaluations and are not fitted or smoothed.
    for name in ("MSE", "Flow", "HT"):
        values = frame[frame["method"] == name].sort_values("steps_k")
        style = STYLE[name]
        ax.plot(
            values["steps_k"],
            values["success_pct"],
            color=style["color"],
            marker=style["marker"],
            markersize=5.5 if name == "HT" else 5.0,
            markeredgecolor="white",
            markeredgewidth=0.65,
            linewidth=2.2 if name == "HT" else 1.8,
            label=style["label"],
            zorder=4 if name == "HT" else 3,
        )

    ax.annotate(
        r"43.0\% at 32k",
        xy=(32, 42.976),
        xytext=(20.5, 49.0),
        color=STYLE["HT"]["color"],
        fontsize=7.0,
        fontweight="bold",
        arrowprops={
            "arrowstyle": "->",
            "color": STYLE["HT"]["color"],
            "lw": 0.9,
        },
    )

    # Label the final measured results without crowding every marker.
    for name in ("HT", "Flow", "MSE"):
        value = frame[(frame["method"] == name) & (frame["steps_k"] == 60)][
            "success_pct"
        ].item()
        ax.text(
            61.25,
            value,
            f"{value:.1f}",
            color=STYLE[name]["color"],
            fontsize=7.1,
            fontweight="bold" if name == "HT" else "normal",
            va="center",
        )

    ax.set_title("GR00T N1.7 on RoboCasa-GR1", loc="left", fontweight="bold", pad=5)
    ax.set_xlabel(r"Training steps ($\times 10^3$)")
    ax.set_ylabel("Mean success rate (%)")
    ax.set_xlim(0, 68)
    ax.set_ylim(0, 56)
    ax.set_xticks([2, 16, 32, 60])
    ax.set_yticks([0, 10, 20, 30, 40, 50])
    ax.grid(axis="both", color="0.89", linewidth=0.6, zorder=0)
    ax.tick_params(length=2.8, pad=2)
    ax.spines[["top", "right"]].set_visible(False)
    ax.legend(
        loc="upper left",
        ncol=3,
        frameon=False,
        bbox_to_anchor=(0.01, 0.99),
        columnspacing=1.0,
        handlelength=1.7,
        handletextpad=0.35,
        borderaxespad=0.0,
    )
    fig.subplots_adjust(left=0.16, right=0.90, bottom=0.18, top=0.89)

    png_path = OUT_DIR / "fig_convergence_curve.png"
    pdf_path = OUT_DIR / "fig_convergence_curve.pdf"
    fig.savefig(png_path, dpi=300)
    fig.savefig(pdf_path)
    plt.close(fig)
    print(f"saved {png_path}")
    print(f"saved {pdf_path}")


if __name__ == "__main__":
    main()
