"""Illustrate the proposed 2D no-free-lunch construction for MIP.

This is a design / analytic-prediction figure, not an empirical result.  The
counterexample uses asymmetric, zero-mean nuisance action jitter.  Regression
returns its conditional mean, whereas MIP composes that mean with a denoiser
whose posterior estimate is pulled toward the nearby, dominant action mode.
"""

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch, Rectangle
import numpy as np


ROOT = Path(__file__).resolve().parents[1]
OUT_PNG = ROOT / "analysis/paper/toy_mip_nfl_design.png"
OUT_PDF = ROOT / "analysis/paper/toy_mip_nfl_design.pdf"

# Normalized lateral-action construction.  This matches MIP's default
# second-view input scale: act_t = act + (1 - t_two_step) eps, t=0.9.
P = 0.8
B = 0.10
R_NEG, R_POS = -B, 4 * B
ANCHOR_SIGMA = 0.10

# Physical controller used only to illustrate the closed-loop consequence.
# One normalized lateral-action unit is 10 mm.
ACTION_MM = 10.0
K = 0.5
DOCK_TOL_MM = 0.5

BLUE = "#2878b5"
RED = "#d84a3a"
PURPLE = "#7b5aa6"
ORANGE = "#e49b35"
GREEN = "#3a9856"
GRAY = "#666666"


def posterior_denoiser(x):
    """E[R | R + sigma*eps = x] for the two-point residual mixture."""
    w_neg = P * np.exp(-0.5 * ((x - R_NEG) / ANCHOR_SIGMA) ** 2)
    w_pos = (1 - P) * np.exp(-0.5 * ((x - R_POS) / ANCHOR_SIGMA) ** 2)
    return (R_NEG * w_neg + R_POS * w_pos) / (w_neg + w_pos + 1e-30)


def rounded_box(ax, xy, w, h, text, fc, ec, fontsize=9):
    box = FancyBboxPatch(
        xy,
        w,
        h,
        boxstyle="round,pad=0.018,rounding_size=0.025",
        facecolor=fc,
        edgecolor=ec,
        linewidth=1.4,
        transform=ax.transAxes,
    )
    ax.add_patch(box)
    ax.text(
        xy[0] + w / 2,
        xy[1] + h / 2,
        text,
        ha="center",
        va="center",
        fontsize=fontsize,
        transform=ax.transAxes,
    )
    return box


def arrow_axes(ax, start, end, color=GRAY, connectionstyle="arc3"):
    ax.add_patch(
        FancyArrowPatch(
            start,
            end,
            arrowstyle="-|>",
            mutation_scale=12,
            linewidth=1.4,
            color=color,
            connectionstyle=connectionstyle,
            transform=ax.transAxes,
        )
    )


fig = plt.figure(figsize=(15.5, 9.2), facecolor="white")
gs = fig.add_gridspec(
    2,
    3,
    height_ratios=[1.08, 0.92],
    width_ratios=[1.25, 1.0, 1.0],
    left=0.055,
    right=0.975,
    bottom=0.08,
    top=0.84,
    hspace=0.42,
    wspace=0.32,
)
fig.suptitle(
    "A constructive 2D no-free-lunch task: when action modes are nuisance, "
    "MIP's denoising projection can hurt",
    fontsize=16,
    weight="bold",
    y=0.968,
)
fig.text(
    0.5,
    0.905,
    "DESIGN + POPULATION PREDICTION (not a trained-model result): asymmetric "
    "zero-mean joystick jitter; convex averaging is the correct control law",
    ha="center",
    fontsize=11,
    color="#333333",
)

# -------------------------------------------------------------------------
# A. Robot-like geometry and representative noisy demonstrations.
ax = fig.add_subplot(gs[0, :2])
x = np.linspace(0, 160, 300)
half = 5.0 + 9.0 * (1 - np.clip(x / 130.0, 0, 1))
ax.fill_between(x, -half, half, color="#e9eef2", zorder=0)
ax.plot(x, half, color="#7c858c", lw=1.7)
ax.plot(x, -half, color="#7c858c", lw=1.7)
ax.axhline(0, color="#9aa1a6", lw=1.0, ls="--", zorder=1)

# Analytic-looking successful noisy demonstrations: a stable servo plus
# zero-mean skew jitter, with a short settle section near the dock.
rng = np.random.default_rng(5)
for j in range(12):
    yy = rng.uniform(-10, 10)
    xs, ys = [0.0], [yy]
    for step in range(55):
        xx = min(4.0 * (step + 1), 160.0)
        r = R_NEG if rng.random() < P else R_POS
        # taper only the plotted realization during the final settle steps;
        # the conditional-label construction remains active in the dataset.
        settle = 0.35 if xx >= 156 else 1.0
        ay = -K * yy + settle * ACTION_MM * r
        yy += ay
        xs.append(xx)
        ys.append(yy)
        if xx >= 160 and abs(yy) < DOCK_TOL_MM:
            break
    ax.plot(xs, ys, color=ORANGE, alpha=0.32, lw=1.05)

# Dock face and target.
ax.add_patch(Rectangle((158.8, -DOCK_TOL_MM), 2.4, 2 * DOCK_TOL_MM,
                       facecolor=GREEN, edgecolor="#1c6a35", lw=1.5, zorder=6))
ax.scatter([160], [0], marker="*", s=160, color="#1c6a35", zorder=7)
ax.scatter([0], [0], s=52, color="black", zorder=7)
ax.annotate("start distribution", (4, 8.8), xytext=(22, 15.2),
            arrowprops=dict(arrowstyle="->", color=GRAY), fontsize=9)
ax.annotate("tight dock\n$|y|<0.5$ mm", (159.5, 0), xytext=(126, 15.0),
            arrowprops=dict(arrowstyle="->", color=GREEN), fontsize=9,
            color="#1c6a35", ha="center")
ax.text(15, -17.2,
        "training demos: stable corrective controller + temporally varying,\n"
        "asymmetric zero-mean action jitter (orange)", fontsize=9.5,
        color="#8a5c19")
ax.set_xlim(-3, 166)
ax.set_ylim(-20, 20)
ax.set_xlabel("forward position $x$ (mm)")
ax.set_ylabel("lateral position $y$ (mm)")
ax.set_title("A   2D funnel-to-dock task; training and evaluation states overlap",
             loc="left", fontsize=11.5, weight="bold")
ax.grid(alpha=0.12)

# -------------------------------------------------------------------------
# B. Conditional action-label distribution.
ax = fig.add_subplot(gs[0, 2])
ax.axhline(0, color="#888888", lw=1)
ax.vlines([R_NEG, R_POS], 0, [P, 1 - P], colors=[RED, ORANGE], lw=8, alpha=0.85)
ax.scatter([R_NEG, R_POS], [P, 1 - P], s=85, color=[RED, ORANGE], zorder=4)
ax.axvline(0, color=BLUE, lw=2.2, ls="--")
ax.scatter([0], [0], marker="D", s=70, color=BLUE, zorder=5)
ax.text(R_NEG, P + 0.055, "$a^*-b$\n80%", ha="center", color=RED,
        fontsize=10, weight="bold")
ax.text(R_POS, 0.255, "$a^*+4b$\n20%", ha="center", color="#a96709",
        fontsize=10, weight="bold")
ax.text(0.015, 0.055, r"$E[a\mid s]=a^*$" + "\n(correct convex action)", color=BLUE,
        fontsize=9.5)
ax.text(-0.23, 0.93, "$0.8(-b)+0.2(4b)=0$", fontsize=11,
        bbox=dict(boxstyle="round,pad=0.25", fc="white", ec="#aaaaaa"))
ax.set_xlim(-0.27, 0.55)
ax.set_ylim(-0.03, 1.04)
ax.set_xlabel("lateral action label, relative to $a^*(s)$")
ax.set_ylabel("probability mass")
ax.set_title("B   Same state, skewed labels\n      Modes are nuisance—not strategies",
             loc="left", fontsize=11.5, weight="bold")
ax.spines[["top", "right"]].set_visible(False)

# -------------------------------------------------------------------------
# C. Objective / inference pathways.
ax = fig.add_subplot(gs[1, 0])
ax.axis("off")
ax.set_title("C   The inductive-bias collision", loc="left", fontsize=11.5,
             weight="bold", pad=8)
rounded_box(ax, (0.02, 0.65), 0.23, 0.18, "state $s$", "#f4f4f4", "#777777")
rounded_box(ax, (0.37, 0.65), 0.27, 0.18, "conditional mean\n$a^*(s)$", "#e7f1f9", BLUE)
arrow_axes(ax, (0.25, 0.74), (0.37, 0.74), BLUE)
rounded_box(ax, (0.73, 0.65), 0.24, 0.18, "execute $a^*$\n(center)", "#e5f4e9", GREEN)
arrow_axes(ax, (0.64, 0.74), (0.73, 0.74), GREEN)
ax.text(0.02, 0.88, "L2 / HG", color=BLUE, fontsize=11, weight="bold",
        transform=ax.transAxes)

rounded_box(ax, (0.02, 0.18), 0.23, 0.18, "state $s$", "#f4f4f4", "#777777")
rounded_box(ax, (0.34, 0.18), 0.25, 0.18, "step 1\n$a^*(s)$", "#eee9f5", PURPLE)
rounded_box(ax, (0.67, 0.18), 0.30, 0.18,
            "step-2 denoiser\n" + r"$D(a^*)\approx a^*-b$",
            "#f9e8e5", RED)
arrow_axes(ax, (0.25, 0.27), (0.34, 0.27), PURPLE)
arrow_axes(ax, (0.59, 0.27), (0.67, 0.27), RED)
ax.text(0.02, 0.41, "MIP (full)", color=PURPLE, fontsize=11, weight="bold",
        transform=ax.transAxes)
ax.text(0.50, 0.03,
        "Causal control: deploy the same MIP checkpoint step-1-only.\n"
        "Prediction: it centers; adding step 2 creates the bias.",
        ha="center", fontsize=8.8, color="#333333", transform=ax.transAxes)

# -------------------------------------------------------------------------
# D. Analytic posterior denoiser.
ax = fig.add_subplot(gs[1, 1])
xx = np.linspace(-0.32, 0.58, 600)
dd = posterior_denoiser(xx)
d0 = float(posterior_denoiser(np.array([0.0]))[0])
ax.plot(xx, dd, color=PURPLE, lw=2.5,
        label=r"$D(x)=E[R\mid R+\sigma\epsilon=x]$")
ax.plot(xx, xx, color="#999999", lw=1.1, ls=":", label="identity")
ax.axvline(0, color=BLUE, lw=1.5, ls="--")
ax.axhline(0, color="#aaaaaa", lw=0.9)
ax.scatter([0], [d0], s=90, color=RED, zorder=5)
ax.annotate(
    f"MIP inference input: mean = 0\nposterior output = {d0:+.3f} "
    + r"$\approx -b$",
    xy=(0, d0),
    xytext=(0.11, -0.26),
    arrowprops=dict(arrowstyle="->", color=RED),
    fontsize=9,
    color=RED,
)
ax.set_xlim(-0.32, 0.58)
ax.set_ylim(-0.34, 0.5)
ax.set_xlabel("input residual $x-a^*(s)$")
ax.set_ylabel("denoised residual $D(x)-a^*(s)$")
ax.set_title("D   Population prediction for MIP's second view\n"
             + r"      ($b=\sigma=0.1$, no finite-data argument)",
             loc="left", fontsize=11.5, weight="bold")
ax.legend(frameon=False, fontsize=8, loc="upper left")
ax.grid(alpha=0.16)

# -------------------------------------------------------------------------
# E. Closed-loop consequence.
ax = fig.add_subplot(gs[1, 2])
steps = np.arange(24)
y0 = 7.0
y_l2 = [y0]
y_mip = [y0]
world_bias = d0 * ACTION_MM
for _ in steps[1:]:
    y_l2.append((1 - K) * y_l2[-1])
    y_mip.append((1 - K) * y_mip[-1] + world_bias)
y_l2 = np.asarray(y_l2)
y_mip = np.asarray(y_mip)
eq = world_bias / K
ax.axhspan(-DOCK_TOL_MM, DOCK_TOL_MM, color=GREEN, alpha=0.16,
           label="dock tolerance")
ax.plot(steps, y_l2, color=BLUE, lw=2.5, label="L2 / HG / MIP step 1")
ax.plot(steps, y_mip, color=RED, lw=2.5, label="full MIP (predicted)")
ax.axhline(eq, color=RED, lw=1.2, ls="--")
ax.text(9.0, -2.55,
        "biased equilibrium " + rf"$\approx{eq:.1f}$ mm",
        color=RED, fontsize=8.8)
ax.scatter([steps[-1]], [y_l2[-1]], marker="*", s=110, color=GREEN, zorder=5)
ax.scatter([steps[-1]], [y_mip[-1]], marker="x", s=95, color=RED,
           linewidths=2.5, zorder=5)
ax.set_ylim(-2.85, 7.4)
ax.set_xlabel("replanning cycle")
ax.set_ylabel("lateral position $y$ (mm)")
ax.set_title("E   Predicted closed loop\n      Averaging docks; mode projection misses",
             loc="left", fontsize=11.5, weight="bold")
ax.legend(frameon=False, fontsize=8, loc="upper right")
ax.grid(alpha=0.16)

fig.text(
    0.5,
    0.018,
    "Falsifiers / controls: symmetric mixture -> tie  |  matched Gaussian jitter -> tie  |  "
    "reveal nuisance/style -> MIP recovers  |  step-1-only -> centers  |  "
    "sweep mixture separation / anchor noise -> analytic phase curve",
    ha="center",
    fontsize=9.5,
    color="#333333",
)

OUT_PNG.parent.mkdir(parents=True, exist_ok=True)
fig.savefig(OUT_PNG, dpi=190, facecolor="white")
fig.savefig(OUT_PDF, facecolor="white")
print(f"wrote {OUT_PNG}")
print(f"wrote {OUT_PDF}")
