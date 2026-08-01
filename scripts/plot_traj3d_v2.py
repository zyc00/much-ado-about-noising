"""3D eef trajectory figure: human GT demos vs hMSE_s5 vs hMIP_s5 rollouts.
Three panels, shared axis limits; success = saturated, failure = light + thin."""
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D  # noqa

OUT = "analysis/traj_vis"
gt = np.load(f"{OUT}/human_gt.npz")
import os
mip = np.load(f"{OUT}/human_" + os.environ.get("MIPD", "hmipL286k") + ".npz")
ht = np.load(f"{OUT}/human_" + os.environ.get("HTD", "hheterot_s1000") + ".npz")

def rollouts(z):
    eps = []
    i = 0
    while f"ep{i}_pos" in z.files:
        eps.append((z[f"ep{i}_pos"], z[f"ep{i}_meta"]))
        i += 1
    return eps

panels = [
    ("GT (human demos)", [(gt[k], None) for k in sorted(gt.files, key=lambda s: int(s[2:]))], "#555555", "#555555"),
    ("MIP (fresh, band 74-89)", rollouts(mip), "#1f6fb2", "#a9cbe8"),
    ("Student-t best (hetero-t s1000, 78-80)", rollouts(ht), "#8e44ad", "#d7bde2"),
]
allp = np.concatenate([p for _, eps, _, _ in panels for p, _ in eps])
lims = [(allp[:, i].min() - 0.02, allp[:, i].max() + 0.02) for i in range(3)]

fig = plt.figure(figsize=(18, 6.5))
for pi, (title, eps, c_succ, c_fail) in enumerate(panels):
    ax = fig.add_subplot(1, 3, pi + 1, projection="3d")
    n_s = 0
    for p, meta in eps:
        if meta is None:
            ax.plot(p[:, 0], p[:, 1], p[:, 2], color=c_succ, lw=0.9, alpha=0.65)
            n_s += 1
        else:
            succ = bool(meta[0])
            ax.plot(p[:, 0], p[:, 1], p[:, 2],
                    color=(c_succ if succ else c_fail), lw=(1.1 if succ else 0.7),
                    alpha=(0.85 if succ else 0.75))
            n_s += int(succ)
        ax.scatter(*p[0], color="#27ae60", s=14, depthshade=False)   # start
        ax.scatter(*p[-1], color="#111111", s=14, marker="x", depthshade=False)  # end
    extra = "" if eps[0][1] is None else f"  ({n_s}/{len(eps)} success, saturated=success)"
    ax.set_title(title + extra, fontsize=11)
    ax.set_xlim(*lims[0]); ax.set_ylim(*lims[1]); ax.set_zlim(*lims[2])
    ax.set_xlabel("x [m]", fontsize=8); ax.set_ylabel("y [m]", fontsize=8); ax.set_zlabel("z [m]", fontsize=8)
    ax.tick_params(labelsize=7)
    ax.view_init(elev=22, azim=-60)
fig.suptitle("Human ToolHang: end-effector trajectories — GT demos vs MIP vs best student-t rollouts "
             "(green dot = start, black x = end)", fontsize=13)
plt.tight_layout(rect=[0, 0, 1, 0.94])
plt.savefig("analysis/paper/traj3d_gt_mip_ht.png", dpi=150, bbox_inches="tight")
print("SAVED analysis/paper/traj3d_gt_mip_ht.png")
