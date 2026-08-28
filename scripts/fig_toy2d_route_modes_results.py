"""Visualize the continuously valid noisy-route decision-gate benchmark."""

from pathlib import Path
import json

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
import numpy as np


ROOT = Path(__file__).resolve().parents[1]
INP = ROOT / "analysis/toy2d_route_modes_swept_gate_checkpointed_results.json"
VERIFY = ROOT / "analysis/checkpoints/toy2d_route_modes_swept_gate/seed_0_reloaded_evaluation.json"
OUT_PNG = ROOT / "analysis/paper/toy2d_route_modes_swept_gate_results.png"
OUT_PDF = ROOT / "analysis/paper/toy2d_route_modes_swept_gate_results.pdf"
R = json.loads(INP.read_text())
V = json.loads(VERIFY.read_text())
CELLS = R["cells"]
C = R["constants"]

METHODS = ("regression", "hg", "mip_step1", "mip_full", "ht")
NAME = {"regression": "L2", "hg": "HG", "mip_step1": "MIP step 1",
        "mip_full": "MIP full", "ht": "HT"}
COLOR = {"regression": "#2878b5", "hg": "#159b88",
         "mip_step1": "#8064a2", "mip_full": "#d84a3a", "ht": "#ef8a25"}


def vals(method, key):
    return np.array([c["methods"][method][key] for c in CELLS], float)


def half_width(x):
    x = np.asarray(x)
    return 5.0 + 9.0 * (1.0 - np.clip(x / 130.0, 0.0, 1.0))


def draw_world(ax):
    xx = np.linspace(0, 160, 400)
    hw = half_width(xx)
    ax.fill_between(xx, -hw, hw, color="#e9eef2", zorder=0)
    ax.plot(xx, hw, color="#7c858c", lw=1.5)
    ax.plot(xx, -hw, color="#7c858c", lw=1.5)
    ax.axhline(0, color="#999999", lw=0.9, ls=":")
    gx = C["gate_x_mm"]
    gh = C["gate_half_mm"]
    ax.plot([gx, gx], [-gh, gh], color="#252525", lw=8,
            solid_capstyle="butt", zorder=7)
    ax.add_patch(Rectangle((158.5, -0.5), 3.0, 1.0,
                           facecolor="#83c995", edgecolor="#27863e", lw=1.3,
                           zorder=7))
    ax.scatter([160], [0], marker="*", s=105, color="#27863e", zorder=8)
    ax.set_xlim(-2, 164)
    ax.set_ylim(-10, 10)
    ax.grid(alpha=0.11)


fig = plt.figure(figsize=(15.2, 9.6), facecolor="white")
gs = fig.add_gridspec(2, 2, left=0.07, right=0.98, bottom=0.09, top=0.875,
                      hspace=0.38, wspace=0.27)
fig.suptitle(
    "Collectable noisy paths: only HT commits far enough to clear the swept decision gate",
    fontsize=15.5, weight="bold", y=0.965,
)
fig.text(
    0.5, 0.917,
    "CHECKPOINT-BACKED TRAINED MODELS — 8 seeds; 3,000 swept-collision-free routes/seed; "
    "route modes −1 (70%), +1 (20%), +5 (10%); shared decision states balanced 40×",
    ha="center", fontsize=10.3, color="#333333",
)

# A. Demonstration routes.
ax = fig.add_subplot(gs[0, 0])
draw_world(ax)
route_colors = {-1.0: "#2878b5", 1.0: "#ef8a25", 5.0: "#b1443f"}
route_names = {-1.0: "lower route (70%)", 1.0: "upper route (20%)",
               5.0: "far-upper route (10%)"}
seen = set()
for p in CELLS[0]["data"]["paths"]:
    mode = p["mode"]
    ax.plot(p["x"], p["y"], color=route_colors[mode], alpha=0.48, lw=1.6,
            label=route_names[mode] if mode not in seen else None)
    seen.add(mode)
ax.annotate("thin gate blocks the mean/partial route",
            (C["gate_x_mm"], 0), (27, 4.8),
            arrowprops=dict(arrowstyle="->", color="#222222", lw=1.3),
            color="#222222", fontsize=9.4, weight="bold")
ax.text(102, -6.8,
        "executed data policy: 3,000/3,000 = 100% SR\n"
        "0 swept collisions; all routes reconverge",
        ha="center", va="center", fontsize=9.2,
        bbox=dict(boxstyle="round,pad=0.4", fc="white", ec="#90979c"))
ax.set_xlabel("forward position $x$ (mm)")
ax.set_ylabel("lateral position $y$ (mm)")
ax.set_title("A   Every executed noisy route is swept-collision-free and docks",
             loc="left", fontsize=11.5, weight="bold")
ax.legend(frameon=False, fontsize=8.5, loc="lower left",
          bbox_to_anchor=(0.01, 0.015))

# The gate is sub-millimetric relative to the full 160 mm route, so show the
# actual continuous crossings at readable scale.
zoom = ax.inset_axes([0.57, 0.55, 0.39, 0.35])
for mode in (-1.0, 1.0, 5.0):
    p = next(p for p in CELLS[0]["data"]["paths"] if p["mode"] == mode)
    zoom.plot(p["x"], p["y"], color=route_colors[mode], lw=1.8)
zoom.plot([C["gate_x_mm"], C["gate_x_mm"]],
          [-C["gate_half_mm"], C["gate_half_mm"]],
          color="#252525", lw=7, solid_capstyle="butt", zorder=7)
zoom.axhline(0, color="#999999", lw=0.7, ls=":")
zoom.set_xlim(0, 10)
zoom.set_ylim(-1.9, 1.9)
zoom.set_title("swept crossing at $x=3$ mm", fontsize=7.7, pad=2)
zoom.tick_params(labelsize=6.5)
zoom.grid(alpha=0.12)

# B. Success rates.
ax = fig.add_subplot(gs[0, 1])
rng = np.random.default_rng(31)
for i, method in enumerate(METHODS):
    v = vals(method, "success_rate")
    ax.bar(i, v.mean(), width=0.66, color=COLOR[method], alpha=0.86, zorder=2)
    ax.scatter(np.full(len(v), i) + rng.uniform(-0.12, 0.12, len(v)), v,
               color="black", alpha=0.58, s=24, zorder=4)
    ax.text(i, v.mean() + 0.045, f"{v.mean():.1%}", ha="center",
            fontsize=9.0, weight="bold")
ax.axhline(1, color="#888888", lw=1, ls=":")
ax.set_xticks(range(len(METHODS)), [NAME[m] for m in METHODS], rotation=8)
ax.set_ylim(-0.03, 1.11)
ax.set_ylabel("task success (clear gate and dock)")
ax.set_title("B   HT wins 100% vs MIP 14.2% vs L2 0%",
             loc="left", fontsize=11.5, weight="bold")
ax.grid(axis="y", alpha=0.16)

# C. First branch action.
ax = fig.add_subplot(gs[1, 0])
for i, method in enumerate(METHODS):
    v = vals(method, "first_lateral_action_mean")
    ax.bar(i, v.mean(), width=0.66, color=COLOR[method], alpha=0.86, zorder=2)
    ax.scatter(np.full(len(v), i) + rng.uniform(-0.12, 0.12, len(v)), v,
               color="black", alpha=0.58, s=24, zorder=4)
    ax.text(i, v.mean() + 0.010, f"{v.mean():+.3f}", ha="center",
            fontsize=8.8, weight="bold")
blocked = C["nominal_blocked_first_action_half"]
ax.axhspan(-blocked, blocked, color="#d84a3a", alpha=0.10,
           label="nominal gate-collision band ($y_0\\approx0$)")
ax.axhline(-0.20, color="#27863e", lw=1.8, ls="--",
           label="full lower-route action")
ax.axhline(0, color="#555555", lw=1)
ax.set_xticks(range(len(METHODS)), [NAME[m] for m in METHODS], rotation=8)
ax.set_ylim(-0.23, 0.17)
ax.set_ylabel("first lateral action (normalized units)")
ax.set_title("C   MIP partially projects; HT fully commits",
             loc="left", fontsize=11.5, weight="bold")
ax.legend(frameon=False, fontsize=8.5, loc="upper right")
ax.grid(axis="y", alpha=0.16)

# D. Actual first-decision trajectories (seed 0).
ax = fig.add_subplot(gs[1, 1])
cell0 = next(c for c in CELLS if c["seed"] == 0)

# Two feasible collected branches.  The rare far-upper route is in panel A;
# omitting it here lets the lower/upper separation be read at physical scale.
for mode, label in ((-1.0, "collected lower route (70%)"),
                    (1.0, "collected upper route (20%)")):
    candidates = [p for p in cell0["data"]["paths"] if p["mode"] == mode]
    p = min(candidates, key=lambda q: abs(q["y"][0]))
    ax.plot(p["x"], p["y"], color=route_colors[mode], lw=2.2,
            ls="--", alpha=0.55, label=label, zorder=2)

gx = C["gate_x_mm"]
gh = C["gate_half_mm"]
ax.plot([gx, gx], [-gh, gh], color="#252525", lw=10,
        solid_capstyle="butt", label="blocking gate", zorder=8)


def clip_at_gate(tr):
    """Clip a displayed polyline at its first swept collision with the gate."""
    xs = np.asarray([0.0] + tr["x"], float)
    ys = np.asarray([tr["y0"]] + tr["y"], float)
    for j in range(len(xs) - 1):
        if xs[j] < gx <= xs[j + 1]:
            alpha = (gx - xs[j]) / (xs[j + 1] - xs[j])
            yc = ys[j] + alpha * (ys[j + 1] - ys[j])
            if abs(yc) < gh:
                return (np.r_[xs[:j + 1], gx],
                        np.r_[ys[:j + 1], yc], True)
    return xs, ys, False


collision_points = {}
for method in ("regression", "mip_full", "ht"):
    tr = min(V["evaluation"][method]["traces"],
             key=lambda q: abs(q["y0"]))
    xs, ys, collided = clip_at_gate(tr)
    ax.plot(xs, ys, color=COLOR[method], lw=3.2,
            label=f"{NAME[method]} loaded checkpoint", zorder=6)
    if collided:
        collision_points[method] = (xs[-1], ys[-1])
        ax.scatter([xs[-1]], [ys[-1]], marker="X", s=100,
                   color=COLOR[method], edgecolor="white", lw=0.8, zorder=10)

ax.annotate("L2: mean route → collision", collision_points["regression"],
            (5.0, 0.28), arrowprops=dict(arrowstyle="->", color=COLOR["regression"]),
            color=COLOR["regression"], fontsize=8.6, weight="bold")
ax.annotate("MIP: partial lower projection → collision", collision_points["mip_full"],
            (5.0, -0.32), arrowprops=dict(arrowstyle="->", color=COLOR["mip_full"]),
            color=COLOR["mip_full"], fontsize=8.6, weight="bold")
ax.annotate("HT follows the complete lower route", (8.0, -0.68), (5.3, -1.00),
            arrowprops=dict(arrowstyle="->", color=COLOR["ht"]),
            color=COLOR["ht"], fontsize=8.6, weight="bold")
ax.axhline(0, color="#999999", lw=0.8, ls=":")
ax.set_xlim(-0.2, 12.0)
ax.set_ylim(-1.13, 1.13)
ax.grid(alpha=0.14)
ax.set_xlabel("forward position $x$ (mm)")
ax.set_ylabel("lateral position $y$ (mm)")
ax.set_title("D   Reloaded seed-0 checkpoints: HT follows lower; L2/MIP hit gate",
             loc="left", fontsize=11.5, weight="bold")
ax.legend(frameon=False, fontsize=7.7, loc="upper right", ncol=2)

fig.text(
    0.5, 0.022,
    "Mechanism: route mean = 0 → L2 hits the gate; MIP moves only partway toward the dominant route; "
    "HT selects the complete 70% route. Noise is temporally coherent at path level and executed during collection—not injected post hoc.",
    ha="center", fontsize=9.1, color="#333333",
)

OUT_PNG.parent.mkdir(parents=True, exist_ok=True)
fig.savefig(OUT_PNG, dpi=190, facecolor="white")
fig.savefig(OUT_PDF, facecolor="white")
print(f"wrote {OUT_PNG}")
print(f"wrote {OUT_PDF}")
