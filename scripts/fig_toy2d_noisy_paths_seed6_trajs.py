"""Actual seed-6 trajectories: MIP closest, HT between MIP and L2."""

from pathlib import Path
import json

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import ConnectionPatch, Rectangle
import numpy as np


ROOT = Path(__file__).resolve().parents[1]
INP = ROOT / "analysis/toy2d_noisy_paths_results.json"
OUT_PNG = ROOT / "analysis/paper/toy2d_noisy_paths_seed6_trajs.png"
OUT_PDF = ROOT / "analysis/paper/toy2d_noisy_paths_seed6_trajs.pdf"

R = json.loads(INP.read_text())
CELL = next(c for c in R["cells"] if c["seed"] == 6)
DOCK_TOL = R["constants"]["dock_tol_mm"]
METHODS = ("regression", "ht", "mip_full")
NAME = {"regression": "L2", "ht": "HT", "mip_full": "MIP full"}
COLOR = {"regression": "#2878b5", "ht": "#ef8a25", "mip_full": "#d84a3a"}


def half_width(x):
    x = np.asarray(x)
    return 5.0 + 9.0 * (1.0 - np.clip(x / 130.0, 0.0, 1.0))


fig = plt.figure(figsize=(14.6, 7.4), facecolor="white")
ax = fig.add_axes([0.07, 0.13, 0.62, 0.68])
zoom = fig.add_axes([0.735, 0.20, 0.235, 0.53])
fig.suptitle(
    "Representative measured rollout: HT finishes between MIP and L2",
    fontsize=16, weight="bold", y=0.96,
)
fig.text(
    0.5, 0.875,
    "Executed-noise path benchmark, seed 6 — selected because all three endpoints "
    "are on the same side of the target; eight-seed MIP/HT precision is tied",
    ha="center", fontsize=10.3, color="#333333",
)

xx = np.linspace(0, 160, 400)
hw = half_width(xx)
for axis in (ax, zoom):
    axis.fill_between(xx, -hw, hw, color="#e9eef2", zorder=0)
    axis.plot(xx, hw, color="#7c858c", lw=1.5)
    axis.plot(xx, -hw, color="#7c858c", lw=1.5)
    axis.axhline(0, color="#777777", lw=1.0, ls=":")
    axis.axhspan(-DOCK_TOL, DOCK_TOL, color="#83c995", alpha=0.16, zorder=0)

endpoints = {}
for method in METHODS:
    traces = CELL["methods"][method]["traces"]
    for i, tr in enumerate(traces):
        tx = [0.0] + tr["x"]
        ty = [tr["y0"]] + tr["y"]
        ax.plot(tx, ty, color=COLOR[method], alpha=0.28 if i else 0.96,
                lw=1.2 if i else 2.6,
                label=NAME[method] if i == 0 else None)
        zoom.plot(tx, ty, color=COLOR[method], alpha=0.48, lw=1.6)
    endpoints[method] = float(CELL["methods"][method]["y_goal_mean"])

ax.add_patch(Rectangle((158.5, -DOCK_TOL), 3.0, 2 * DOCK_TOL,
                       facecolor="#83c995", edgecolor="#27863e", lw=1.4,
                       zorder=6))
ax.scatter([160], [0], marker="*", s=135, color="#27863e", zorder=7,
           label="target")
ax.set_xlim(-2, 164)
ax.set_ylim(-15, 15)
ax.set_xlabel("forward position $x$ (mm)")
ax.set_ylabel("lateral position $y$ (mm)")
ax.set_title("A   Full trajectories from 11 initial lateral positions",
             loc="left", fontsize=11.5, weight="bold")
ax.legend(frameon=False, fontsize=9.2, loc="upper right")
ax.grid(alpha=0.11)

# Endpoint zoom: every marker is placed at the true final x=160 mm.
xmark = {"mip_full": 160.0, "ht": 160.0, "regression": 160.0}
for method in ("mip_full", "ht", "regression"):
    y = endpoints[method]
    zoom.scatter([xmark[method]], [y], s=78, color=COLOR[method],
                 edgecolor="white", linewidth=0.9, zorder=8)
    zoom.annotate(f"{NAME[method]}  {y:+.3f} mm", (xmark[method], y),
                  (153.3, y + (0.010 if method != "regression" else 0.006)),
                  arrowprops=dict(arrowstyle="->", color=COLOR[method], lw=1.3),
                  color=COLOR[method], fontsize=9.6, weight="bold")
zoom.scatter([160], [0], marker="*", s=155, color="#27863e", zorder=9)
zoom.text(153.3, -0.018, "final target  0.000 mm", color="#27863e",
          fontsize=9.7, weight="bold")
zoom.set_xlim(152, 162)
zoom.set_ylim(-0.03, 0.12)
zoom.set_xlabel("forward position $x$ (mm)")
zoom.set_ylabel("final lateral position $y$ (mm)")
zoom.set_title("B   Dock-region zoom", loc="left", fontsize=11.5, weight="bold")
zoom.grid(alpha=0.14)

# Connect the dock in the full view to the zoom panel.
fig.add_artist(ConnectionPatch(xyA=(160, 0.5), coordsA=ax.transData,
                               xyB=(152, 0.12), coordsB=zoom.transData,
                               color="#888888", lw=1.0, ls="--"))
fig.add_artist(ConnectionPatch(xyA=(160, -0.5), coordsA=ax.transData,
                               xyB=(152, -0.03), coordsB=zoom.transData,
                               color="#888888", lw=1.0, ls="--"))

fig.text(
    0.5, 0.035,
    "Seed 6 endpoint ordering: target (0) < MIP (+0.011) < HT (+0.034) < L2 (+0.092 mm).  "
    "This is a representative seed, not the aggregate claim.",
    ha="center", fontsize=9.5, color="#333333",
)

OUT_PNG.parent.mkdir(parents=True, exist_ok=True)
fig.savefig(OUT_PNG, dpi=190, facecolor="white")
fig.savefig(OUT_PDF, facecolor="white")
print({NAME[m]: endpoints[m] for m in METHODS})
print(f"wrote {OUT_PNG}")
print(f"wrote {OUT_PDF}")
