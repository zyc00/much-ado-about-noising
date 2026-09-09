#!/usr/bin/env python3
"""
Plot simulation and real-world benchmark success rates.

Requires:
    python -m pip install matplotlib

Run:
    python simulation_realworld_benchmarks.py

Edit PANELS below to change:
    - data
    - labels
    - y-axis limits
    - ticks

Notes:
    - None leaves a method empty.
    - Simulation panels show "n/a" for missing methods.
    - Empty real-world panels remain visually empty.
    - PNG output is exactly 3575 x 860 px at 500 dpi.
"""

from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib.patches import Patch


# ============================================================
# EDIT ONLY THIS BLOCK
# ============================================================

PANELS = [

    # --------------------------------------------------------
    # Simulation
    # --------------------------------------------------------

    {
        "title": "RoboMimic",
        "subtitle": "avg. of 4 configs",
        "labels": ["DP", "MSE", "HT"],
        "values": [94.1, 79.6, 93.5],
        "ylim": (0, 120),
        "yticks": [0, 25, 50, 75, 100],
    },

    {
        "title": "RoboCasa-GR1",
        "subtitle": "GR00T N1.7",
        "labels": ["Flow", "MSE", "HT"],
        "values": [44.5, 37.8, 51.5],
        "ylim": (0, 120),
        "yticks": [0, 25, 50, 75, 100],
    },

    {
        "title": "SIMPLER",
        "subtitle": "GR00T N1.7 · 2-robot avg.",
        "labels": ["Flow", "MSE", "HT"],
        "values": [67.4, None, 65.5],
        "ylim": (0, 120),
        "yticks": [0, 25, 50, 75, 100],
    },

    {
        "title": "LIBERO",
        "subtitle": "π0.5 · 4-suite avg.",
        "labels": ["Flow", "MSE", "HT"],
        "values": [96.9, 96.8, 97.6],
        "ylim": (0, 120),
        "yticks": [0, 25, 50, 75, 100],
    },

    {
        "title": "LIBERO-10",
        "subtitle": "Cosmos 3",
        "labels": ["Flow", "MSE", "HT"],
        "values": [95.2, None, 96.0],
        "ylim": (0, 120),
        "yticks": [0, 25, 50, 75, 100],
    },

    # --------------------------------------------------------
    # Real world
    # --------------------------------------------------------

    {
        "title": "Push-T",
        "subtitle": "",
        "labels": ["Flow", "MSE", "HT"],
        "values": [86.0, 54.0, 88.0],
        "ylim": (0, 120),
        "yticks": [0, 25, 50, 75, 100],
    },

    {
        "title": "Hang Mug",
        "subtitle": "",
        "labels": ["Flow", "MSE", "HT"],
        "values": [None, None, None],
        "ylim": (0, 120),
        "yticks": [0, 25, 50, 75, 100],
    },

    {
        "title": "T-Insertion",
        "subtitle": "",
        "labels": ["Flow", "MSE", "HT"],
        "values": [None, None, None],
        "ylim": (0, 120),
        "yticks": [0, 25, 50, 75, 100],
    },
]


# ============================================================
# COLORS
# ============================================================

FLOW_COLOR = "#AEC1DC"
MSE_COLOR = "#4F7FB7"
HT_COLOR = "#F39749"

TEXT_COLOR = "#252C34"
VALUE_COLOR = "#3E4C59"

MUTED_COLOR = "#6B737C"
AXIS_COLOR = "#A7AFB7"
EMPTY_COLOR = "#CBD2D9"

HT_TEXT_COLOR = "#C96D24"

COLORS = [
    FLOW_COLOR,
    MSE_COLOR,
    HT_COLOR,
]


# ============================================================
# TYPOGRAPHY
# ============================================================

# Section:
# Simulation / Real world
SECTION_FS = 6.5

# Group:
# Trained from scratch / Mid-size VLA / ...
GROUP_FS = 4.3

# Task:
# RoboMimic / RoboCasa-GR1 / ...
TASK_FS = 5.6

# GR00T N1.7 / Cosmos 3 / ...
SUBTITLE_FS = 4.0

# Axes
XTICK_FS = 5.1
YTICK_FS = 4.7
YLABEL_FS = 5.0

# Values
VALUE_FS = 5.2
DELTA_FS = 5.3
NA_FS = 4.2

# Legend
LEGEND_FS = 4.5


# ============================================================
# LINE WEIGHTS
# ============================================================

SPINE_LW = 0.36
RULE_LW = 0.30
DIVIDER_LW = 0.43


# ============================================================
# FONT WEIGHTS
# ============================================================

SECTION_WEIGHT = 600
TASK_WEIGHT = 600

# Keep HT annotations stronger.
HT_WEIGHT = "bold"


# ============================================================
# DELTA ANNOTATION POSITION
# ============================================================

# Distance above HT bar.
DELTA_OFFSET = 0.13

# Prevent delta text from reaching the very top.
DELTA_TOP_PADDING = 0.035


# ============================================================
# MATPLOTLIB SETTINGS
# ============================================================

plt.rcParams.update(
    {
        "font.family": "DejaVu Sans",
        "pdf.fonttype": 42,
        "ps.fonttype": 42,
    }
)


# ============================================================
# HELPERS
# ============================================================

def add_axis_break(ax):
    """Draw a small // symbol when the y-axis is truncated."""

    kwargs = dict(
        transform=ax.transAxes,
        color=AXIS_COLOR,
        clip_on=False,
        linewidth=SPINE_LW,
    )

    ax.plot(
        (-0.017, 0.017),
        (-0.006, 0.028),
        **kwargs,
    )

    ax.plot(
        (-0.017, 0.017),
        (0.014, 0.048),
        **kwargs,
    )


def draw_panel(ax, cfg, panel_index):
    """Draw one benchmark panel."""

    lo, hi = cfg["ylim"]
    span = hi - lo

    ax.set_xlim(-0.58, 2.58)

    # --------------------------------------------------------
    # Y axis
    #
    # Important:
    # set ticks BEFORE resetting ylim.
    #
    # Matplotlib's set_yticks() is allowed to expand the view
    # limits to include all ticks. Calling set_ylim() afterward
    # guarantees that the configured limits stay exact.
    # --------------------------------------------------------

    ax.set_yticks(cfg["yticks"])
    ax.set_ylim(lo, hi)

    ax.tick_params(
        axis="y",
        labelsize=YTICK_FS,
        width=0.30,
        length=1.6,
        pad=1.1,
        colors=TEXT_COLOR,
    )

    # --------------------------------------------------------
    # X axis
    # --------------------------------------------------------

    ax.set_xticks([0, 1, 2])

    ax.set_xticklabels(
        cfg["labels"],
        fontsize=XTICK_FS,
    )

    ax.tick_params(
        axis="x",
        length=0,
        pad=1.7,
        colors=TEXT_COLOR,
    )

    # --------------------------------------------------------
    # Spines
    # --------------------------------------------------------

    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    for side in ["left", "bottom"]:
        ax.spines[side].set_color(AXIS_COLOR)
        ax.spines[side].set_linewidth(SPINE_LW)

    # ========================================================
    # BARS
    # ========================================================

    for j, (value, color) in enumerate(
        zip(cfg["values"], COLORS)
    ):

        if value is None:

            # Show n/a only for simulation panels.
            if panel_index < 5:
                ax.text(
                    j,
                    lo + 0.045 * span,
                    "n/a",
                    ha="center",
                    va="bottom",
                    fontsize=NA_FS,
                    color="#9AA1A8",
                )

            continue

        ax.bar(
            j,
            value - lo,
            bottom=lo,
            width=0.66,
            color=color,
            edgecolor="none",
            zorder=3,
        )

        # ----------------------------------------------------
        # Numerical value above bar
        # ----------------------------------------------------

        # Simulation Flow/DP numbers are taken from the paper /
        # official repo; mark them with a star (explained in
        # the caption).
        label = f"{value:.1f}"
        if j == 0 and panel_index < 5:
            label += "*"

        ax.text(
            j,
            value + 0.022 * span,
            label,
            ha="center",
            va="bottom",
            fontsize=VALUE_FS,
            color=(
                HT_TEXT_COLOR
                if j == 2
                else VALUE_COLOR
            ),
            fontweight=(
                HT_WEIGHT
                if j == 2
                else "normal"
            ),
        )

    # ========================================================
    # HT - FLOW / DP DIFFERENCE
    # ========================================================

    flow = cfg["values"][0]
    ht = cfg["values"][2]

    if flow is not None and ht is not None:

        delta = ht - flow

        # ----------------------------------------------------
        # IMPORTANT:
        #
        # Anchor the delta annotation to the HT bar.
        #
        # Old behavior:
        #     max(flow, mse, ht) + offset
        #
        # caused negative deltas such as SIMPLER's -1.9 to
        # jump above the taller Flow bar.
        #
        # New behavior:
        #     ht + fixed offset
        #
        # so +7.0, -1.9, +0.7, etc. always sit in the same
        # semantic location: above HT.
        # ----------------------------------------------------

        delta_y = ht + DELTA_OFFSET * span

        # Never let annotation run into / beyond panel top.
        delta_y = min(
            delta_y,
            hi - DELTA_TOP_PADDING * span,
        )

        ax.text(
            2,
            delta_y,
            f"{delta:+.1f}",
            ha="center",
            va="bottom",
            fontsize=DELTA_FS,
            color=HT_TEXT_COLOR,
            fontweight=HT_WEIGHT,
        )

    # ========================================================
    # AXIS BREAK
    # ========================================================

    if lo > 0:
        add_axis_break(ax)

    # ========================================================
    # EMPTY REAL-WORLD PANELS
    # ========================================================

    if (
        panel_index >= 6
        and all(v is None for v in cfg["values"])
    ):

        ax.tick_params(
            axis="y",
            colors="#ABB5C0",
        )

        for side in ["left", "bottom"]:
            ax.spines[side].set_color(EMPTY_COLOR)


# ============================================================
# BUILD FIGURE
# ============================================================

def make_figure():
    """Build and return the full figure."""

    # --------------------------------------------------------
    # Canvas
    #
    # 7.15 x 1.72 at 500 dpi
    # = 3575 x 860 px
    #
    # Aspect ratio ~4.16:1.
    # --------------------------------------------------------

    fig = plt.figure(
        figsize=(7.15, 1.72),
        facecolor="white",
    )

    # ========================================================
    # OUTER LAYOUT
    #
    # row 0:
    #     Simulation / Real world
    #
    # row 1:
    #     simulation group labels / real-world legend
    #
    # row 2:
    #     task title + subplot
    #
    # column 5:
    #     dashed sim/real separator
    # ========================================================

    outer = fig.add_gridspec(
        3,
        9,

        height_ratios=[
            0.105,
            0.125,
            0.770,
        ],

        width_ratios=[
            1, 1, 1, 1, 1,
            0.085,
            1, 1, 1,
        ],

        left=0.052,
        right=0.995,
        bottom=0.095,
        top=0.975,

        hspace=0.0,
        wspace=0.34,
    )

    # ========================================================
    # TOP LEVEL TITLES
    # ========================================================

    sim_header = fig.add_subplot(
        outer[0, 0:5]
    )

    sim_header.axis("off")

    sim_header.text(
        0.5,
        0.55,
        "Simulation",
        ha="center",
        va="center",
        fontsize=SECTION_FS,
        fontweight=SECTION_WEIGHT,
        color=TEXT_COLOR,
    )

    real_header = fig.add_subplot(
        outer[0, 6:9]
    )

    real_header.axis("off")

    real_header.text(
        0.5,
        0.55,
        "Real world",
        ha="center",
        va="center",
        fontsize=SECTION_FS,
        fontweight=SECTION_WEIGHT,
        color=TEXT_COLOR,
    )

    # ========================================================
    # SIMULATION GROUP HEADERS
    # ========================================================

    sim_groups = [
        (
            0,
            1,
            "Trained from scratch",
        ),
        (
            1,
            3,
            "Mid-size VLA · pretrained action head",
        ),
        (
            3,
            5,
            "Large VLA · pretrained action head",
        ),
    ]

    for col_start, col_end, text in sim_groups:

        ax_group = fig.add_subplot(
            outer[1, col_start:col_end]
        )

        ax_group.set_xlim(0, 1)
        ax_group.set_ylim(0, 1)
        ax_group.axis("off")

        # Header
        ax_group.text(
            0.5,
            0.66,
            text,
            ha="center",
            va="center",
            fontsize=GROUP_FS,
            color=MUTED_COLOR,
        )

        # Thin horizontal rule
        ax_group.plot(
            [0.02, 0.98],
            [0.13, 0.13],
            color=AXIS_COLOR,
            linewidth=RULE_LW,
        )

    # ========================================================
    # REAL-WORLD LEGEND
    # ========================================================

    legend_ax = fig.add_subplot(
        outer[1, 6:9]
    )

    legend_ax.set_xlim(0, 1)
    legend_ax.set_ylim(0, 1)
    legend_ax.axis("off")

    legend_handles = [
        Patch(
            facecolor=FLOW_COLOR,
            edgecolor="none",
            label="Flow / DP",
        ),
        Patch(
            facecolor=MSE_COLOR,
            edgecolor="none",
            label="MSE",
        ),
        Patch(
            facecolor=HT_COLOR,
            edgecolor="none",
            label="HT (ours)",
        ),
    ]

    legend_ax.legend(
        handles=legend_handles,
        loc="center",
        bbox_to_anchor=(0.5, 0.64),

        ncol=3,
        frameon=False,

        fontsize=LEGEND_FS,

        handlelength=1.05,
        handleheight=0.65,
        handletextpad=0.35,

        columnspacing=1.7,
    )

    legend_ax.plot(
        [0.02, 0.98],
        [0.13, 0.13],
        color=AXIS_COLOR,
        linewidth=RULE_LW,
    )

    # ========================================================
    # PANELS
    # ========================================================

    axes = []

    for i, cfg in enumerate(PANELS):

        # Column 5 is reserved for separator.
        col = i if i < 5 else i + 1

        # ----------------------------------------------------
        # Each task gets:
        #
        # 1. title/subtitle region
        # 2. actual chart
        # ----------------------------------------------------

        panel_grid = outer[2, col].subgridspec(
            2,
            1,

            height_ratios=[
                0.24,
                0.76,
            ],

            hspace=0.0,
        )

        # ====================================================
        # TASK HEADER
        # ====================================================

        title_ax = fig.add_subplot(
            panel_grid[0, 0]
        )

        title_ax.set_xlim(0, 1)
        title_ax.set_ylim(0, 1)
        title_ax.axis("off")

        title_ax.text(
            0.5,
            0.74,
            cfg["title"],
            ha="center",
            va="center",
            fontsize=TASK_FS,
            fontweight=TASK_WEIGHT,
            color=TEXT_COLOR,
        )

        if cfg["subtitle"]:

            title_ax.text(
                0.5,
                0.30,
                cfg["subtitle"],
                ha="center",
                va="center",
                fontsize=SUBTITLE_FS,
                color=MUTED_COLOR,
                linespacing=0.95,
            )

        # ====================================================
        # ACTUAL CHART
        # ====================================================

        ax = fig.add_subplot(
            panel_grid[1, 0]
        )

        axes.append(ax)

        draw_panel(
            ax,
            cfg,
            panel_index=i,
        )

    # ========================================================
    # SHARED Y LABEL
    # ========================================================

    axes[0].set_ylabel(
        "Success rate (%)",
        fontsize=YLABEL_FS,
        color=TEXT_COLOR,
        labelpad=4,
    )

    # ========================================================
    # SIMULATION / REAL-WORLD DIVIDER
    # ========================================================

    separator_ax = fig.add_subplot(
        outer[:, 5]
    )

    separator_ax.set_xlim(0, 1)
    separator_ax.set_ylim(0, 1)
    separator_ax.axis("off")

    separator_ax.plot(
        [0.5, 0.5],
        [0.02, 0.98],

        transform=separator_ax.transAxes,

        color="#8795A3",
        linewidth=DIVIDER_LW,

        linestyle=(
            0,
            (4, 5),
        ),

        clip_on=False,
    )

    return fig


# ============================================================
# EXPORT
# ============================================================

def main():

    # Works both as a .py script and when copied into an
    # interactive environment.
    try:
        output_dir = Path(__file__).resolve().parent
    except NameError:
        output_dir = Path.cwd()

    fig = make_figure()

    pdf_path = (
        output_dir
        / "simulation_realworld_benchmarks.pdf"
    )

    png_path = (
        output_dir
        / "simulation_realworld_benchmarks.png"
    )

    # --------------------------------------------------------
    # IMPORTANT:
    #
    # Do NOT use bbox_inches="tight".
    #
    # The figure has deliberately been designed with an exact
    # 7.15 x 1.72 canvas. Tight cropping changes that geometry.
    # --------------------------------------------------------

    fig.savefig(
        pdf_path,
        bbox_inches=None,
        facecolor="white",
    )

    fig.savefig(
        png_path,
        dpi=500,
        bbox_inches=None,
        facecolor="white",
    )

    print(f"Saved {pdf_path}")
    print(f"Saved {png_path}")

    plt.close(fig)


if __name__ == "__main__":
    main()
