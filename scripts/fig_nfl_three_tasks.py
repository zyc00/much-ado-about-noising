"""NFL summary figure: three tasks, data on top, trained policies below.

Row 1 = the collected demonstrations (what the data looks like).
Row 2 = the trained policies of the three families on that data.

Diffusion is ALWAYS the history-conditioned variant (previous action chunk in
the conditioning, episode-rollout training data) — i.e. the strong version that
real diffusion policies use, so the comparison is not against a strawman.
HT is the SAME nu=2 in all three panels (the paper's default), so the three
outcomes come from the tasks, not from tuning the loss per panel.

Every task has a ~100%-success demonstrator and no injected corruption.

Usage: python scripts/fig_nfl_three_tasks.py
"""

import json
import os
import sys
from pathlib import Path

# Panel A's cell parameters must match the run that produced its numbers.
os.environ.setdefault("RW_LEN", "36"); os.environ.setdefault("RW_LEN_JIT", "16")
os.environ.setdefault("RW_GAP", "40"); os.environ.setdefault("RW_GAP_JIT", "15")
os.environ.setdefault("RW_AMP", "6"); os.environ.setdefault("RW_CLEAN", "150")
os.environ.setdefault("RW_GAIN", "1.0"); os.environ.setdefault("RW_START", "8")
os.environ.setdefault("RW_START_JIT", "25"); os.environ.setdefault("RW_DOCK", "1.5")

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from toy2d_rectwave import (  # noqa: E402
    AMP as RW_AMP, CLEAN_X as RW_CLEAN, GAIN as RW_GAIN,
    demo_aim as rw_aim, sample_pulses as rw_pulses,
)

C_MSE, C_DP, C_HT = "#2b6cb0", "#6b46c1", "#c23b22"
XG = 160.0
K, FWD, HORIZON = 0.5, 4.0, 8


def load(*paths):
    """Merge arms from several result files (later files win)."""
    arms = {}
    for p in paths:
        if os.path.exists(p):
            arms.update(json.load(open(p))["arms"])
    return arms


# Panel A: square-wave excursions on a flat baseline (l2/flow in one file, ht2 in the other).
rwv = load("analysis/toy2d_rectwave_ecg.json", "analysis/toy2d_rectwave_ecg_ht.json")
# Panels B/C: history-matched arms only, so the "all policies history-conditioned"
# claim in the caption holds for every family and not just flow.
# 72k/256, 0.05mm execution noise, 70/30 branch split. At 70/30 the HT optimum
# clears the block with margin, so nu=2 commits and its collision rate is 0.000 —
# unlike the 60/40 split, where its aim lands within 0.02mm of the block edge.
blk = load("analysis/toy2d_obstacle_p70n005.json")
slm = load("analysis/toy2d_slalom_p70.json")

SLOTS = np.array([0.0, -6.0, -11.0])
GAP = 1.2
PLATES = [(50.0, 58.0), (100.0, 108.0)]
SWITCH = [62.0, 112.0]
BLK_X0, BLK_X1, BLK_HALF, DET_A = 60.0, 100.0, 3.0, 6.0
AIM0, AIM_JIT = 20.0, 16.0
BLK_EXEC_NOISE = 0.05   # mm/step, panel B only (see PART CDLXVII)
LA_AIM, LA_CLEAN = 3.0, 130.0


def hw_funnel(x):
    return 5.0 + 9.0 * (1.0 - np.clip(x / 130.0, 0.0, 1.0))


def hw_wide(x):
    return 14.0 - 1.5 * np.clip(x / XG, 0.0, 1.0)


def scene(ax, kind):
    hw = hw_funnel if kind == "block" else hw_wide
    xs = np.linspace(0, XG, 200)
    ax.fill_between(xs, -hw(xs), hw(xs), color="#e8eef6", zorder=0)
    ax.plot(xs, hw(xs), color="0.5", lw=1.0)
    ax.plot(xs, -hw(xs), color="0.5", lw=1.0)
    if kind == "block":
        ax.add_patch(plt.Rectangle((BLK_X0, -BLK_HALF), BLK_X1 - BLK_X0, 2 * BLK_HALF,
                                   facecolor="0.35", edgecolor="0.2", zorder=4))
        ax.text((BLK_X0 + BLK_X1) / 2, 0, "BLOCK", ha="center", va="center",
                color="white", fontsize=8.5, fontweight="bold", zorder=5)
    if kind == "rectwave":
        ax.axvspan(RW_CLEAN, XG, color="#dff0d8", alpha=0.6, zorder=1)
        ax.axvline(RW_CLEAN, color="0.35", ls="--", lw=1.2, zorder=7)
    if kind == "slalom":
        for x0, x1 in PLATES:
            h = float(hw_wide((x0 + x1) / 2))
            for lo, hi in [(SLOTS[0] + GAP, h), (SLOTS[1] + GAP, SLOTS[0] - GAP),
                           (SLOTS[2] + GAP, SLOTS[1] - GAP), (-h, SLOTS[2] - GAP)]:
                if hi > lo:
                    ax.add_patch(plt.Rectangle((x0, lo), x1 - x0, hi - lo,
                                               facecolor="0.35", edgecolor="0.2", zorder=4))
    ax.plot(XG, 0, "*", color="tab:green", ms=14, zorder=8)
    ax.set_xlim(0, XG + 4)
    ax.set_ylim(-15, 15)
    ax.tick_params(labelsize=8.5)


def draw_demos(ax, kind, rng, n):
    """Row 1: the collected demonstrations."""
    for _ in range(n):
        if kind == "rectwave":
            pulses = rw_pulses(rng)          # flat baseline, occasional square excursions
            y = float(rng.uniform(-8, 8)); x = 0.0
            tx, ty = [x], [y]
            while x < XG:
                y -= RW_GAIN * (y - rw_aim(x, pulses)); x += FWD
                tx.append(x); ty.append(y)
            ax.plot(tx, ty, color="0.35", alpha=0.45, lw=0.9, zorder=2)
        elif kind == "block":
            branch = 1.0 if rng.random() < 0.70 else -1.0
            a0 = AIM0 + rng.uniform(-AIM_JIT, AIM_JIT)
            y = float(rng.uniform(-10, 10)); x = 0.0
            tx, ty = [x], [y]
            while x < XG:
                yt = branch * DET_A if (a0 <= x <= BLK_X1) else 0.0
                y -= K * (y - yt) - rng.normal(0.0, BLK_EXEC_NOISE); x += FWD
                tx.append(x); ty.append(y)
            ax.plot(tx, ty, color="tab:red" if branch > 0 else "tab:orange",
                    alpha=0.5, lw=0.9, zorder=2)
        else:
            ks = rng.choice(3, p=[0.70, 0.20, 0.10], size=2)
            lanes = [float(SLOTS[k] + np.clip(rng.normal(0, [0.08, 0.3, 0.5][k]), -0.9, 0.9)) for k in ks]
            y = float(rng.uniform(-10, 10)); x = float(rng.uniform(0, FWD * HORIZON))
            tx, ty = [x], [y]
            while x < XG:
                lane = lanes[0] if x < SWITCH[0] else (lanes[1] if x < SWITCH[1] else 0.0)
                y -= K * (y - lane); x += FWD
                tx.append(x); ty.append(y)
            ax.plot(tx, ty, color="0.5", alpha=0.45, lw=0.9, zorder=2)


# (kind, data-title, trained-title, [(source, key, label, colour)], verdict)
PANELS = [
    ("rectwave",
     "A   Transient excursions: each one returns to the baseline",
     [(rwv, "l2", "MSE", C_MSE), (rwv, "flow", "Flow matching", C_DP),
      (rwv, "ht2", "HT nu=2", C_HT)],
     "Both point estimates hold the baseline — flow arrives still deflected"),
    ("block",
     "B   Block detour: 70/30 split, staggered commitment",
     [(blk, "l2_hist", "MSE", C_MSE), (blk, "flow_hist", "Flow matching", C_DP),
      (blk, "ht2_hist", "HT nu=2", C_HT)],
     "Flow wins — HT commits and clears (0% collisions), the mean drives in (100%)"),
    ("slalom",
     "C   Slot slalom: two plates, three gaps each",
     [(slm, "l2_hist", "MSE", C_MSE), (slm, "flow_hist", "Flow matching", C_DP),
      (slm, "ht2_hist", "HT nu=2", C_HT)],
     "HT wins — the mean falls between lanes, the sampler switches lanes"),
]

fig, axes = plt.subplots(2, 3, figsize=(17.5, 8.4), sharey=True)
fig.suptitle("No free lunch: every family loses somewhere\n"
             "~100%-success demonstrators  ·  all executed behaviour  ·  panel B adds 0.05mm/step execution noise  ·  all policies history-conditioned",
             fontsize=13.5, fontweight="bold")

rng = np.random.default_rng(7)
for col, (kind, dtitle, entries, verdict) in enumerate(PANELS):
    # ---- row 1: collected data
    ax = axes[0, col]
    scene(ax, kind)
    if kind == "rectwave":
        ax.text(RW_CLEAN - 44, 11.3, "excursions must end\nbefore this line", fontsize=8.5, color="0.3")
    draw_demos(ax, kind, rng, n={"block": 26, "rectwave": 9}.get(kind, 16))
    ax.set_title(dtitle, fontsize=11.5, fontweight="bold", loc="left")

    # ---- row 2: trained policies
    ax = axes[1, col]
    scene(ax, kind)
    for src, key, lab, c in entries:
        a = src.get(key)
        if a is None or not a.get("seed0"):
            continue
        for t in a["seed0"]["traces"]:
            kind = t.get("kind", "ok")
            ax.plot(t["x"], t["y"], color=c, alpha=0.65 if kind == "ok" else 0.42,
                    lw=1.15, ls="-" if kind == "ok" else (0, (4, 2)), zorder=6)
            if kind == "collision":
                ax.plot(t["x"][-1], t["y"][-1], "x", color=c, ms=6.5, mew=1.7, zorder=9)
            elif kind == "off-dock":
                ax.plot(t["x"][-1], t["y"][-1], "o", mfc="none", mec=c,
                        ms=5.5, mew=1.4, zorder=9)
        sd = a.get("sr_std")
        lbl = f"{lab}  SR {a['sr_mean']:.2f}" + (f" ± {sd:.2f}" if sd else "")
        ax.plot([], [], color=c, lw=2.2, label=lbl)
    ax.legend(fontsize=9, loc="lower left", framealpha=0.92)
    ax.set_xlabel("forward x (mm)", fontsize=10)
    ax.text(0.5, -0.28, verdict, transform=ax.transAxes, ha="center",
            fontsize=10.5, style="italic", color="0.2")

axes[0, 0].set_ylabel("collected demonstrations\n\nlateral y (mm)", fontsize=10.5)
axes[1, 0].set_ylabel("trained policies\n\nlateral y (mm)", fontsize=10.5)

fig.tight_layout(rect=[0, 0.04, 1, 0.92])
fig.savefig("analysis/paper/nfl_three_tasks.png", dpi=170)
fig.savefig("analysis/paper/nfl_three_tasks.pdf")
print("saved analysis/paper/nfl_three_tasks.png")
