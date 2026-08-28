"""Panel-A-style design figure for the BigCubeLift NFL settings.

Trajectories are read from the actual training datasets (th_cb*_200.hdf5,
dumped to cube_traj_dump.npz by scripts/dump_cube_trajs.py on the cluster);
PMF insets show the injected nuisance construction per setting.

Usage: python scripts/fig_cube_nfl_design.py <dump.npz> <out.png>
"""

import sys

import matplotlib.pyplot as plt
import numpy as np

DUMP = sys.argv[1] if len(sys.argv) > 1 else "analysis/cube_traj_dump.npz"
OUT = sys.argv[2] if len(sys.argv) > 2 else "analysis/paper/cube_nfl_design.png"

d = np.load(DUMP)
N = 12

CUBE_HALF = 0.033
OPEN_HALF = 0.040  # ~80 mm gripper opening
TOL = OPEN_HALF - CUBE_HALF  # 7 mm lateral slack for the grasp

SETTINGS = [
    dict(
        tag="clean",
        title="clean (control)",
        pmf=([0.0], [1.0]),
        pmf_note="no injection",
        cert="cert 96%",
        sr="HT .93  L2 .95  MIP .95  (ceiling)",
        srcolor="0.35",
    ),
    dict(
        tag="sk5",
        title="toyskew B=0.05",
        pmf=([-0.05, 0.20], [0.8, 0.2]),
        pmf_note="0.8(-B)+0.2(4B)=0",
        cert="cert 99%",
        sr="HT .878 > L2 .828 > MIP .738",
        srcolor="tab:green",
    ),
    dict(
        tag="sk10",
        title="toyskew B=0.10",
        pmf=([-0.10, 0.40], [0.8, 0.2]),
        pmf_note="same shape, 2x magnitude",
        cert="cert 100%",
        sr="L2 .655 > MIP .623 > HT .580",
        srcolor="tab:blue",
    ),
    dict(
        tag="symm",
        title="symm ±B (falsifier)",
        pmf=([-0.05, 0.05], [0.5, 0.5]),
        pmf_note="symmetric, same hold",
        cert="cert 100%",
        sr="HT .86  L2 .86  MIP .83  (no reordering)",
        srcolor="0.35",
    ),
]

fig, axes = plt.subplots(1, 4, figsize=(22, 6.2), sharey=True)
fig.suptitle(
    "BigCubeLift NFL settings: one task, one oracle, four label-nuisance distributions",
    fontsize=19, fontweight="bold", y=1.02,
)
fig.text(
    0.5, 0.955,
    "executed corruption, chunk-held (SKEW_HOLD=8); the scripted servo corrects it "
    "(certificate = collection acceptance); 66 mm cube vs 80 mm opening leaves "
    "±7 mm lateral grasp slack",
    ha="center", fontsize=12, color="0.35",
)

for ax, S in zip(axes, SETTINGS):
    tag = S["tag"]
    # grasp slack band (the "dock" of this task)
    ax.axvspan(-TOL * 1000, TOL * 1000, color="tab:green", alpha=0.10, zorder=0)
    ax.axvline(0, color="0.6", lw=0.8, ls="--", zorder=1)

    for i in range(N):
        eef = d[f"{tag}_eef_{i}"]
        obj = d[f"{tag}_obj_{i}"]
        y = (eef[:, 1] - obj[1]) * 1000.0
        z = eef[:, 2]
        ax.plot(y, z, color="tab:orange", alpha=0.45, lw=1.2, zorder=3)
        ax.plot(y[0], z[0], "o", color="k", ms=3.5, zorder=4)

    # cube (side view) at its true height
    obj0 = d[f"{tag}_obj_0"]
    cube = plt.Rectangle(
        (-CUBE_HALF * 1000, obj0[2] - CUBE_HALF), 2 * CUBE_HALF * 1000, 2 * CUBE_HALF,
        facecolor="0.75", edgecolor="0.4", zorder=2,
    )
    ax.add_patch(cube)

    ax.set_title(f"{S['title']}   ({S['cert']})", fontsize=14, fontweight="bold")
    ax.text(
        0.5, -0.175, S["sr"], transform=ax.transAxes, ha="center",
        fontsize=12.5, fontweight="bold", color=S["srcolor"],
    )
    ax.set_xlabel("eef lateral offset to cube y (mm)", fontsize=11)
    ax.set_xlim(-90, 90)
    ax.set_ylim(0.77, 1.06)

    # PMF inset (panel-B style stems)
    ins = ax.inset_axes([0.66, 0.66, 0.32, 0.30])
    locs, ps = S["pmf"]
    colors = ["tab:red" if p >= 0.5 else "tab:orange" for p in ps]
    if tag == "clean":
        colors = ["tab:blue"]
    for x0, p, c in zip(locs, ps, colors):
        ins.plot([x0, x0], [0, p], color=c, lw=2.5)
        ins.plot(x0, p, "o", color=c, ms=5)
    ins.plot(0, 0, "D", color="tab:blue", ms=4, zorder=5)
    ins.axvline(0, color="tab:blue", lw=0.7, ls=":")
    ins.set_xlim(-0.2, 0.5)
    ins.set_ylim(0, 1.05)
    ins.set_title(S["pmf_note"], fontsize=8.5)
    ins.tick_params(labelsize=7)
    ins.set_xlabel("action nuisance", fontsize=7.5)

axes[0].set_ylabel("eef height z (m)", fontsize=12)
fig.text(
    0.5, -0.105,
    "orange: first 12 training demos per setting (recorded = executed, servo-corrected)  |  "
    "green band: ±7 mm lateral slack the grasp allows  |  gray: 66 mm cube  |  "
    "black dots: episode starts  |  SR = avg_mean_success@300k, mean of 2 seeds",
    ha="center", fontsize=11, color="0.35",
)
fig.tight_layout(rect=[0, 0.0, 1, 0.94])
fig.savefig(OUT, dpi=160, bbox_inches="tight")
fig.savefig(OUT.replace(".png", ".pdf"), bbox_inches="tight")
print("saved", OUT)
