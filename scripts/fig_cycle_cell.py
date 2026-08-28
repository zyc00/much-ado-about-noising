"""Cancelling-cycle cell: why the design failed, made visible.

Left  — full corridor: demonstrations (executed period-5 offset cycle) plus the
        three trained policies.
Right — zoom on the dock. The 0.5 mm tolerance band is the green strip; this is
        where the design breaks: the DEMONSTRATOR itself lands ~1 mm low, because
        position low-pass-filters the offset sequence instead of cancelling it.
        MSE recovers the mean action and docks; HT parks at the persistent-bias
        equilibrium b/K = 1.0 mm; flow matching samples the same dominant mode.

Usage: python scripts/fig_cycle_cell.py
"""

import json

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

C_MSE, C_DP, C_HT = "#2b6cb0", "#6b46c1", "#c23b22"
XG, K, FWD, HORIZON, ACTION_MM = 160.0, 0.5, 4.0, 8, 10.0
B, DOCK = 0.05, 0.5

arms = json.load(open("analysis/toy2d_cycle.json"))["arms"]


def half_width(x):
    return 5.0 + 9.0 * (1.0 - np.clip(x / 130.0, 0.0, 1.0))


def demo_traj(rng):
    """Executed period-5 cycle: 4x -b then +4b (sums to zero)."""
    y = float(rng.uniform(-10, 10)); x = 0.0
    phase = int(rng.integers(0, 5))
    tx, ty = [x], [y]
    while x < XG:
        d = (4.0 * B) if (phase % 5 == 4) else (-B)
        phase += 1
        for _ in range(HORIZON):
            y = y - K * y + d * ACTION_MM
            x += FWD
        tx.append(x); ty.append(y)
    return tx, ty


fig, (axL, axR) = plt.subplots(1, 2, figsize=(15.5, 5.2),
                               gridspec_kw={"width_ratios": [2.1, 1]})
fig.suptitle("Cancelling-cycle cell: the policies separate exactly as designed — "
             "but the DEMONSTRATOR misses the dock, so the cell is not a valid witness",
             fontsize=13.5, fontweight="bold")

for ax, zoom in ((axL, False), (axR, True)):
    xs = np.linspace(0, XG, 200)
    ax.fill_between(xs, -half_width(xs), half_width(xs), color="#e8eef6", zorder=0)
    ax.plot(xs, half_width(xs), color="0.5", lw=1.0)
    ax.plot(xs, -half_width(xs), color="0.5", lw=1.0)
    ax.axhspan(-DOCK, DOCK, color="tab:green", alpha=0.22, zorder=1)
    ax.plot(XG, 0, "*", color="tab:green", ms=15, zorder=9)

    rng = np.random.default_rng(4)
    for _ in range(16):
        tx, ty = demo_traj(rng)
        ax.plot(tx, ty, color="0.45", alpha=0.55, lw=1.0, zorder=2)
    for key, lab, c in (("l2", "MSE", C_MSE), ("flow", "Flow matching", C_DP),
                        ("ht2", "HT nu=2", C_HT)):
        a = arms[key]
        for t in a["seed0"]["traces"]:
            ax.plot(t["x"], t["y"], color=c, alpha=0.7, lw=1.2, zorder=5)
        if not zoom:
            ax.plot([], [], color=c, lw=2.2, label=f"{lab}  SR {a['sr_mean']:.2f}")

    if zoom:
        ax.set_xlim(120, XG + 3)
        ax.set_ylim(-2.2, 1.6)
        ax.axhline(-1.0, color=C_HT, ls=":", lw=1.4)
        ax.text(122, -1.18, "persistent-bias equilibrium  b/K = -1.0 mm",
                fontsize=8.8, color=C_HT)
        ax.text(122, 0.62, "dock tolerance +-0.5 mm", fontsize=8.8, color="tab:green")
        ax.set_title("B   Zoom on the dock", fontsize=11.5, fontweight="bold", loc="left")
    else:
        ax.plot([], [], color="0.45", lw=1.6, label="demonstrations (oracle SR 0.00)")
        ax.legend(fontsize=9, loc="lower left")
        ax.set_xlim(0, XG + 4); ax.set_ylim(-15, 15)
        ax.set_ylabel("lateral y (mm)", fontsize=10.5)
        ax.set_title("A   Full corridor", fontsize=11.5, fontweight="bold", loc="left")
    ax.set_xlabel("forward x (mm)", fontsize=10.5)

fig.tight_layout(rect=[0, 0.01, 1, 0.91])
fig.savefig("analysis/paper/cycle_cell.png", dpi=170)
print("saved analysis/paper/cycle_cell.png")
