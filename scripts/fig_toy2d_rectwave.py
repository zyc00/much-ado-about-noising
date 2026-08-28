"""Figure: transient excursions — demonstrations on top, trained policies below.

Same layout as analysis/paper/nfl_three_tasks.png: one column per task, the
collected demonstrations in the top panel and every trained family overlaid in
the bottom panel with its success rate in the legend.

The demonstrator drives an actual square wave — gain 1.0 reaches the aim in one
4mm step, so edges are vertical and tops are flat — entering the wave at a random
x, flipping up/down at random, and stopping at CLEAN_X so every demo runs
straight into the goal.

Usage: python scripts/fig_toy2d_rectwave.py
Env: RW_TAG (output suffix) plus every RW_* knob from toy2d_rectwave.
"""

import json
import os
import sys
from pathlib import Path

os.environ.setdefault("RW_GAP", "0")        # continuous wave: detour is the norm
os.environ.setdefault("RW_GAP_JIT", "0")
os.environ.setdefault("RW_LEN", "18")
os.environ.setdefault("RW_LEN_JIT", "6")
os.environ.setdefault("RW_AMP", "6")
os.environ.setdefault("RW_CLEAN", "140")
os.environ.setdefault("RW_GAIN", "1.0")
os.environ.setdefault("OB_FLOW_K", "64")

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))
from toy2d_mip_nfl import FORWARD_MM, X_GOAL_MM, train_hist  # noqa: E402
from toy2d_rectwave import (  # noqa: E402
    AMP, CLEAN_X, DOCK, GAIN, HORIZON, demo_aim, gen_episodes, half_width, rollout,
    sample_pulses, train_flow_hist_rw,
)

C_MSE, C_DP, C_HT = "#2b6cb0", "#6b46c1", "#c23b22"
TAG = os.environ.get("RW_TAG", "sq")
CACHE = f"analysis/toy2d_rectwave_fig_{TAG}.json"
ARMS = [("l2", "MSE regression", C_MSE), ("flow", "Flow matching", C_DP)]


def demo_paths(n, seed):
    """One episode per call, each with its own random wave — as collected."""
    rng = np.random.default_rng(seed)
    out = []
    for _ in range(n):
        pulses = sample_pulses(rng)
        y, x = float(rng.uniform(-8.0, 8.0)), 0.0
        px, py = [x], [y]
        while x < X_GOAL_MM:
            for _ in range(HORIZON):
                y += -GAIN * (y - demo_aim(x, pulses))
                x += FORWARD_MM
                px.append(x); py.append(y)
        out.append((px, py))
    return out


def scene(ax, title):
    xs = np.linspace(0, X_GOAL_MM, 400)
    ax.fill_between(xs, -half_width(xs), half_width(xs), color="#e8eef6", zorder=0)
    ax.plot(xs, half_width(xs), color="0.5", lw=1.0)
    ax.plot(xs, -half_width(xs), color="0.5", lw=1.0)
    ax.axvspan(CLEAN_X, X_GOAL_MM, color="#dff0d8", alpha=0.6, zorder=1)
    ax.axvline(CLEAN_X, color="0.35", ls="--", lw=1.2, zorder=7)
    ax.plot(X_GOAL_MM, 0, "*", color="tab:green", ms=14, zorder=8)
    ax.set_xlim(-2, X_GOAL_MM + 6); ax.set_ylim(-13, 13)
    ax.set_title(title, fontsize=11.5, fontweight="bold", loc="left")
    ax.tick_params(labelsize=9)


if os.path.exists(CACHE):
    R = json.load(open(CACHE))
else:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    st, pv, ac, osr = gen_episodes(2500, seed=1000)
    print(f"oracle {osr:.3f}", flush=True)
    R = {"oracle": osr, "demos": demo_paths(26, 7), "roll": {}}
    for key, _, _ in ARMS:
        net = (train_flow_hist_rw(st, pv, ac, 0, 256, 24000, 512, 1e-3, device) if key == "flow"
               else train_hist("ht" if key == "ht2" else "regression",
                               st, pv, ac, 0, 256, 24000, 512, 1e-3, device))
        R["roll"][key] = rollout(net, "flow_hist" if key == "flow" else "reg",
                                 device, n_eval=401)
        print(f"{key}: SR {R['roll'][key]['sr']:.3f}", flush=True)
    json.dump(R, open(CACHE, "w"))

fig, axes = plt.subplots(2, 1, figsize=(9.2, 8.0), sharex=True, sharey=True)

scene(axes[0], "collected demonstrations")
for px, py in R["demos"][:12]:   # 12 is enough to read the shape
    axes[0].plot(px, py, color="0.35", alpha=0.45, lw=1.0, zorder=6)
axes[0].text(3, -10.6, "flat baseline with occasional excursions: out (up or down at random),\n"
                       "hold, back to baseline — the last one ends before the goal", fontsize=9, color="0.25")

scene(axes[1], "trained policies")
for key, lab, c in ARMS:
    r = R["roll"][key]
    for t in r["traces"]:
        axes[1].plot(t["x"], t["y"], color=c, alpha=0.6, lw=1.1, zorder=6)
    axes[1].plot([], [], color=c, lw=2.4,
                 label=f"{lab}   SR {r['sr']:.2f}   |landing| {r['landing_absmed']:.2f}mm")
axes[1].legend(fontsize=9.5, loc="lower left", framealpha=0.92)
axes[1].set_xlabel("forward x (mm)", fontsize=10.5)

axes[0].set_ylabel("collected demonstrations\n\nlateral y (mm)", fontsize=10.5)
axes[1].set_ylabel("trained policies\n\nlateral y (mm)", fontsize=10.5)
fig.suptitle("Transient excursions: every demonstrated excursion returns to the\n"
             "baseline before the goal — a policy that reproduces them need not",
             fontsize=13, fontweight="bold")
fig.text(0.5, 0.005,
         f"amplitude {AMP:g}mm · straight tail from x={CLEAN_X:g} (green) · "
         f"dock tolerance {DOCK:g}mm · oracle {R['oracle']:.2f}",
         ha="center", fontsize=9.5, style="italic", color="0.2")
fig.tight_layout(rect=[0, 0.02, 1, 0.94])
out = f"analysis/paper/rectwave_cell_{TAG}.png"
fig.savefig(out, dpi=180)
print("WROTE", out, "| SR", {k: round(v["sr"], 3) for k, v in R["roll"].items()})
