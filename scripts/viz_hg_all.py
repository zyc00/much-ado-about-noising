"""All 24 HG rollout eef trajectories: overlay + 4x6 grid of individual
3D panels (seed, outcome, steps)."""
import glob

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

eps = []
for f in sorted(glob.glob("logs/rd_hg/ep_*.npz")):
    z = np.load(f)
    eps.append((int(z["seed"]), int(z["asm"]), z["eef"], int(z["steps"])))

fig = plt.figure(figsize=(10, 8))
ax = fig.add_subplot(111, projection="3d")
for seed, asm, eef, steps in eps:
    if asm:
        ax.plot(eef[:, 0], eef[:, 1], eef[:, 2], color="C0", lw=0.7,
                alpha=0.45)
    else:
        ax.plot(eef[:, 0], eef[:, 1], eef[:, 2], color="C3", lw=1.8,
                label=f"FAIL {seed}")
        ax.scatter(*eef[-1], color="C3", marker="x", s=80)
ax.set_title("HG-200: all 24 rollouts (blue = success, red = failure)")
ax.set_xlabel("x")
ax.set_ylabel("y")
ax.set_zlabel("z")
ax.legend(fontsize=8)
ax.view_init(elev=22, azim=-60)
plt.tight_layout()
plt.savefig("analysis/failvids/hg_all_overlay.png", dpi=140)

fig = plt.figure(figsize=(22, 15))
for i, (seed, asm, eef, steps) in enumerate(eps):
    ax = fig.add_subplot(4, 6, i + 1, projection="3d")
    c = "C0" if asm else "C3"
    ax.plot(eef[:, 0], eef[:, 1], eef[:, 2], color=c, lw=0.9)
    ax.scatter(*eef[0], color="k", marker="o", s=12)
    ax.scatter(*eef[-1], color=c, marker="x", s=40)
    ax.set_title(f"{seed} {'OK' if asm else 'FAIL'} {steps}st", fontsize=9)
    ax.set_xticklabels([])
    ax.set_yticklabels([])
    ax.set_zticklabels([])
    ax.view_init(elev=22, azim=-60)
plt.tight_layout()
plt.savefig("analysis/failvids/hg_all_grid.png", dpi=110)
print("VIZ saved", flush=True)
