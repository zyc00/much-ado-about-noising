"""Retry/regrasp toy visualization: executed-demo design (A) and trained
rollouts (B mean family, C diffusion, D HT). Traces from the generous budget.

Usage: python scripts/fig_toy2d_retry.py [retry.json]
"""

import json
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

XG = 160.0
K, FWD, STAGE_Y, TRIG_X, F_FLAW = 0.5, 4.0, 8.0, 120.0, 0.4
HORIZON = 8

d = json.load(open(sys.argv[1] if len(sys.argv) > 1 else "analysis/toy2d_retry.json"))
arms = d["arms"]


def half_width(x):
    return 14.0 - 1.5 * np.clip(x / XG, 0.0, 1.0)


def draw_scene(ax):
    xs = np.linspace(0, XG, 200)
    ax.fill_between(xs, -half_width(xs), half_width(xs), color="#dfe7f0", zorder=0)
    ax.plot(xs, half_width(xs), color="0.45", lw=1.2)
    ax.plot(xs, -half_width(xs), color="0.45", lw=1.2)
    ax.axhline(STAGE_Y, ls="--", color="0.6", lw=0.9, zorder=1)
    ax.text(4, STAGE_Y + 0.5, "staging lane", fontsize=8.5, color="0.45")
    ax.plot(XG, 0, "*", color="tab:green", ms=16, zorder=5)
    ax.set_xlim(0, XG + 4)
    ax.set_ylim(-15, 15)


def demo_traj(rng):
    """One executed demo; returns segments [(xs, ys, is_retreat)]."""
    flawed = rng.random() < F_FLAW
    y = float(rng.uniform(-10, 10))
    x = float(rng.uniform(0, FWD * HORIZON))
    segs = []
    retreat_left = 0
    while x < XG:
        retreat = retreat_left > 0 or (flawed and x >= TRIG_X)
        if retreat and retreat_left == 0:
            retreat_left = 2
            flawed = False
        tx, ty = [x], [y]
        for _ in range(HORIZON):
            lane, dx = (STAGE_Y, -2.0) if retreat else (0.0, FWD)
            y = y - K * (y - lane); x += dx
            tx.append(x); ty.append(y)
        if retreat:
            retreat_left -= 1
        segs.append((tx, ty, retreat))
    return segs


fig, axes = plt.subplots(2, 2, figsize=(16, 8.6), sharex=True, sharey=True)
(axA, axB), (axC, axD) = axes
fig.suptitle("Retry/regrasp toy: 40% of demos back off from a flawed first dock approach and retry "
             "(all executed, bounded, oracle 100%)",
             fontsize=14.5, fontweight="bold")

# A: executed demos with retreat segments highlighted
draw_scene(axA)
rng = np.random.default_rng(3)
for _ in range(18):
    for tx, ty, retreat in demo_traj(rng):
        axA.plot(tx, ty, color="tab:red" if retreat else "0.55",
                 alpha=0.85 if retreat else 0.5, lw=1.5 if retreat else 1.0, zorder=4 if retreat else 3)
axA.plot([], [], color="0.55", lw=1.6, label="approach/dock (executed)")
axA.plot([], [], color="tab:red", lw=1.6, label="retreat to staging (40% of demos, one retry)")
axA.legend(fontsize=9.5, loc="lower left")
axA.set_title("A   Demonstrations: bounded regrasp — retreat, re-stage, re-approach, dock",
              fontsize=12, fontweight="bold", loc="left")
axA.set_ylabel("lateral position y (mm)", fontsize=11)

# B: mean family (l2 + hg)
draw_scene(axB)
for key, lab, c in [("l2", "MSE", "#2b6cb0"), ("hg", "Hetero-Gaussian", "#63b3ed")]:
    a = arms[key]
    for t in a["seed0"]["traces"]:
        axB.plot(t["x"], t["y"], color=c, alpha=0.6, lw=1.2, zorder=3)
    axB.plot([], [], color=c, lw=2.2, label=f"{lab}  SR {a['generous']['sr_mean']:.2f}")
axB.legend(fontsize=9.5, loc="lower left")
axB.set_title("B   Mean family: blends dock and staging modes — misses the 0.5mm dock",
              fontsize=12, fontweight="bold", loc="left")

# C: diffusion
draw_scene(axC)
a = arms["flow"]
for t in a["seed0"]["traces"]:
    axC.plot(t["x"], t["y"], color="#6b46c1", alpha=0.65, lw=1.3, zorder=3)
axC.plot([], [], color="#6b46c1", lw=2.2,
         label=(f"Diffusion  SR {a['generous']['sr_mean']:.2f} (2x budget) / "
                f"{arms['flow']['tight']['sr_mean']:.2f} (1.25x)"))
axC.legend(fontsize=9.5, loc="lower left")
axC.set_title("C   Diffusion: re-samples retreat per dock visit — budget-dependent",
              fontsize=12, fontweight="bold", loc="left")
axC.set_ylabel("lateral position y (mm)", fontsize=11)
axC.set_xlabel("forward position x (mm)", fontsize=11)

# D: HT
draw_scene(axD)
a = arms["ht2"]
for t in a["seed0"]["traces"]:
    axD.plot(t["x"], t["y"], color="#dd6b20", alpha=0.65, lw=1.3, zorder=3)
axD.plot([], [], color="#dd6b20", lw=2.2, label=f"HT nu=2  SR {a['generous']['sr_mean']:.2f} (both budgets)")
axD.legend(fontsize=9.5, loc="lower left")
axD.set_title("D   HT: rejects the retreat cluster (mass 0.31 < 1/3) — docks first pass",
              fontsize=12, fontweight="bold", loc="left")
axD.set_xlabel("forward position x (mm)", fontsize=11)

fig.tight_layout(rect=[0, 0.01, 1, 0.94])
fig.savefig("analysis/paper/toy2d_retry_trajs.png", dpi=160)
fig.savefig("analysis/paper/toy2d_retry_trajs.pdf")
print("saved analysis/paper/toy2d_retry_trajs.png")
