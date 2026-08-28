"""Figure: unobserved-obstacle rectangles — MSE vs flow matching.

Top row    = the demonstrations under two DIFFERENT layouts. The states visited
             are the same; the demonstrated actions are opposite, because the
             demonstrator conditions on a layout the policy never observes.
Bottom row = the two trained policies rolled out under one fresh layout.

MSE averages the two detours and drives the always-free centre band.
Flow reproduces the detour distribution, picks a side blind, and collides.

Usage: python scripts/fig_toy2d_rects.py
"""

import json
import os
import sys
from pathlib import Path

os.environ.setdefault("RC_INNER", "5.0")     # free band is |y| < 5mm
os.environ.setdefault("RC_OUTER", "13.0")
os.environ.setdefault("RC_BERTH", "7.0")
os.environ.setdefault("RC_PUP", "0.5")
os.environ.setdefault("OB_FLOW_K", "64")

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))
from toy2d_mip_nfl import (  # noqa: E402
    ACTION_MM, DOCK_TOL_MM, FORWARD_MM, K_SERVO, X_GOAL_MM,
    predict_hist, train_hist,
)
from toy2d_obstacle import predict_flow_hist  # noqa: E402
from toy2d_rects import (  # noqa: E402
    BERTH, HORIZON, INNER, MAX_CHUNKS, N_RECT, OUTER, P_UP, RECT_LEN,
    demo_aim, gen_episodes, half_width, hits, train_flow_hist_rc,
)

C_MSE, C_DP = "#2b6cb0", "#6b46c1"
TAG = os.environ.get("RC_TAG", "wide")
CACHE = f"analysis/toy2d_rects_fig_{TAG}.json"


def demo_paths(layout, n, rng):
    """Executed demonstrator paths under a KNOWN layout (what the data shows)."""
    out = []
    for _ in range(n):
        y, x = float(rng.uniform(-8.0, 8.0)), 0.0
        px, py = [x], [y]
        while x < X_GOAL_MM:
            for _ in range(HORIZON):
                y += -K_SERVO * (y - demo_aim(x, layout))
                x += FORWARD_MM
                px.append(x); py.append(y)
        out.append((px, py))
    return out


def rollout_fixed(net, sampler, layout, device, n_eval=60, seed=0):
    """All rollouts share ONE layout so the panel is readable."""
    rng = np.random.default_rng(700 + seed)
    y = rng.uniform(-8.0, 8.0, n_eval)
    x = np.zeros(n_eval)
    active = np.ones(n_eval, dtype=bool)
    hit_pt, paths = [], [([0.0], [float(v)]) for v in y]
    y_goal = np.full(n_eval, np.nan)
    prev = torch.zeros(n_eval, 2 * HORIZON, device=device)
    for _ in range(MAX_CHUNKS):
        ids = np.flatnonzero(active)
        if len(ids) == 0:
            break
        st = np.stack([x[ids], y[ids]], axis=1).astype(np.float32)
        zt = (predict_flow_hist(net, st, prev[ids], device) if sampler == "flow_hist"
              else predict_hist(net, st, prev[ids], device))
        prev[ids] = zt
        a = zt.cpu().numpy()
        for j in range(HORIZON):
            live = active[ids]
            x[ids] = np.where(live, x[ids] + np.clip(a[:, 2 * j] * ACTION_MM, -2.0, 8.0), x[ids])
            y[ids] = np.where(live, y[ids] + np.clip(a[:, 2 * j + 1] * ACTION_MM, -12.0, 12.0), y[ids])
            for i in ids:
                if active[i]:
                    paths[i][0].append(float(x[i])); paths[i][1].append(float(y[i]))
            for i in ids:
                if active[i] and (abs(y[i]) > half_width(x[i]) or hits(x[i], y[i], layout)):
                    hit_pt.append((float(x[i]), float(y[i]))); active[i] = False
            reach = (x[ids] >= X_GOAL_MM) & active[ids]
            y_goal[ids[reach]] = y[ids[reach]]
            active[ids[reach]] = False
    ok = np.isfinite(y_goal) & (np.abs(y_goal) < DOCK_TOL_MM)
    return {"paths": paths, "hits": hit_pt, "sr": float(ok.mean())}


def scene(ax, layout, title):
    xs = np.linspace(0, X_GOAL_MM, 400)
    ax.fill_between(xs, -half_width(xs), half_width(xs), color="#e8eef6", zorder=0)
    ax.plot(xs, half_width(xs), color="0.5", lw=1.0)
    ax.plot(xs, -half_width(xs), color="0.5", lw=1.0)
    for x0, side in layout:
        lo = INNER if side > 0 else -OUTER
        ax.add_patch(plt.Rectangle((x0, lo), RECT_LEN, OUTER - INNER,
                                   facecolor="0.35", edgecolor="0.2", zorder=4))
    ax.axhspan(-INNER, INNER, color="#dff0d8", alpha=0.55, zorder=1)
    ax.plot(X_GOAL_MM, 0, "*", color="tab:green", ms=13, zorder=8)
    ax.set_xlim(-2, X_GOAL_MM + 6); ax.set_ylim(-15, 15)
    ax.set_title(title, fontsize=11, fontweight="bold", loc="left")
    ax.tick_params(labelsize=8.5)


if os.path.exists(CACHE):
    R = json.load(open(CACHE))
else:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    st, pv, ac, osr = gen_episodes(2500, seed=1000)
    print(f"oracle {osr:.3f}", flush=True)
    nets = {
        "l2": train_hist("regression", st, pv, ac, 0, 256, 24000, 512, 1e-3, device),
        "flow": train_flow_hist_rc(st, pv, ac, 0, 256, 24000, 512, 1e-3, device),
    }
    rng = np.random.default_rng(4242)
    xs_r = np.linspace(25.0, 110.0, N_RECT)
    lay = {
        "A": [(float(a), float(s)) for a, s in zip(xs_r, [1, -1, 1, -1])],
        "B": [(float(a), float(s)) for a, s in zip(xs_r, [-1, 1, -1, 1])],
        "C": [(float(a), float(s)) for a, s in zip(xs_r, [-1, -1, 1, -1])],
    }
    R = {"oracle": osr, "layouts": lay,
         "demoA": demo_paths(lay["A"], 14, np.random.default_rng(1)),
         "demoB": demo_paths(lay["B"], 14, np.random.default_rng(2)),
         "roll": {k: rollout_fixed(n, "flow_hist" if k == "flow" else "reg",
                                   lay["C"], device) for k, n in nets.items()}}
    json.dump(R, open(CACHE, "w"))

fig, axes = plt.subplots(2, 2, figsize=(13.0, 7.4), sharex=True, sharey=True)
for ax, key, ttl in [(axes[0, 0], "demoA", "demonstrations, layout A"),
                     (axes[0, 1], "demoB", "demonstrations, layout B")]:
    lay = R["layouts"][key[-1]]
    scene(ax, lay, ttl)
    for px, py in R[key]:
        ax.plot(px, py, color="0.35", alpha=0.6, lw=1.0, zorder=6)
    ax.text(3, 12.8, "demonstrator sees the blocks and berths to the free side",
            fontsize=8.5, color="0.25")

for ax, (k, lab, c) in zip(axes[1], [("l2", "MSE regression", C_MSE),
                                     ("flow", "Flow matching", C_DP)]):
    scene(ax, R["layouts"]["C"], f"{lab}   (fresh layout C)")
    r = R["roll"][k]
    for px, py in r["paths"]:
        ax.plot(px, py, color=c, alpha=0.55, lw=1.0, zorder=6)
    if r["hits"]:
        hx, hy = zip(*r["hits"])
        ax.plot(hx, hy, "x", color="#b91c1c", ms=7, mew=1.8, zorder=9)
    ax.plot([], [], color=c, lw=2.4, label=f"{lab}  SR {r['sr']:.2f}")
    ax.legend(fontsize=9, loc="lower left", framealpha=0.92)
    ax.set_xlabel("forward x (mm)", fontsize=10)

axes[0, 0].set_ylabel("collected demonstrations\n\nlateral y (mm)", fontsize=10)
axes[1, 0].set_ylabel("trained policies\n\nlateral y (mm)", fontsize=10)
fig.suptitle(
    "Unobserved obstacles: the mode selector is hidden,\n"
    "so matching the action distribution is unsafe while its mean is safe"
    f"   (p_up={P_UP:g}, berth {BERTH:g}mm, free band |y|<{INNER:g}mm)",
    fontsize=12.5, fontweight="bold")
fig.text(0.5, 0.005,
         "Blocks sit above OR below at random; the green centre band is free either way. "
         "The policy observes only (x, y) — never the layout.",
         ha="center", fontsize=9.5, style="italic", color="0.2")
fig.tight_layout(rect=[0, 0.02, 1, 0.945])
out = f"analysis/paper/rects_cell_{TAG}.png"
fig.savefig(out, dpi=180)
print("WROTE", out, "| oracle", R["oracle"],
      "| SR", {k: round(v["sr"], 3) for k, v in R["roll"].items()})
