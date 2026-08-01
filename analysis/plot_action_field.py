"""Render the 3D action flow field in/around the trajectory tube.
Input: npz files produced by scripts/dump_action_field.py (run on cluster with a
checkpoint; env MODELS="tag:ckpt:loss,..."), pulled to --npz paths.
Usage: python analysis/plot_action_field.py --npz /tmp/actfield_MSE300k.npz /tmp/actfield_MIP300k.npz \
          --out analysis/traj_vis/action_flow_field.png [--elev 22 --azim -58]
Coloring: on-tube arrows royalblue (flow); off-tube arrows coolwarm by inward
component r=-<a_hat, delta_hat> (blue=recovery, red=outward). Arrow length = 0.05m/unit.
"""
import sys, argparse
sysp = [p for p in sys.path if "dist-packages" in p]
sys.path = [p for p in sys.path if "dist-packages" not in p] + sysp
import mpl_toolkits
mpl_toolkits.__path__ = [p + "/mpl_toolkits" for p in sys.path if "site-packages" in p and "local" in p] + list(mpl_toolkits.__path__)
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D

ap = argparse.ArgumentParser()
ap.add_argument("--npz", nargs="+", required=True)
ap.add_argument("--out", default="analysis/traj_vis/action_flow_field.png")
ap.add_argument("--elev", type=float, default=22)
ap.add_argument("--azim", type=float, default=-58)
args = ap.parse_args()

n = len(args.npz)
fig = plt.figure(figsize=(9.5 * n, 9))
for j, path in enumerate(args.npz):
    z = np.load(path, allow_pickle=True)
    pts, vec, rad, inw = z["pts"], z["vec"], z["rad"], z["inw"]
    tag = path.split("actfield_")[-1].replace(".npz", "")
    ax = fig.add_subplot(1, n, j + 1, projection="3d")
    for t in z["tube"]:
        t = np.asarray(t, dtype=np.float32)
        ax.plot(t[:, 0], t[:, 1], t[:, 2], color="gray", lw=0.5, alpha=0.25)
    m0 = rad == 0
    v0 = 0.05 * vec[m0]
    ax.quiver(pts[m0, 0], pts[m0, 1], pts[m0, 2], v0[:, 0], v0[:, 1], v0[:, 2],
              color="royalblue", alpha=0.9, arrow_length_ratio=0.25, linewidth=1.3)
    for rsel, lw in [(1.0, 0.8), (2.0, 1.0), (4.0, 1.2)]:
        m = rad == rsel
        v = 0.05 * vec[m]
        colors = plt.cm.coolwarm_r((inw[m] + 1) / 2)
        ax.quiver(pts[m, 0], pts[m, 1], pts[m, 2], v[:, 0], v[:, 1], v[:, 2],
                  color=colors, alpha=0.85, arrow_length_ratio=0.25, linewidth=lw)
    frac_in = np.nanmean(inw[rad > 0] > 0)
    ax.set_title(f"{tag}\noff-tube inward frac={frac_in:.2f}, mean inward={np.nanmean(inw[rad>0]):+.2f}")
    ax.set_xlim(-0.35, 0.15); ax.set_ylim(-0.45, 0.15); ax.set_zlim(0.8, 1.25)
    ax.view_init(elev=args.elev, azim=args.azim)
leg = [Line2D([0], [0], color="royalblue", lw=2, label="on-tube action (flow)"),
       Line2D([0], [0], color=plt.cm.coolwarm_r(0.95), lw=2, label="off-tube: inward (recovery)"),
       Line2D([0], [0], color=plt.cm.coolwarm_r(0.05), lw=2, label="off-tube: outward")]
fig.legend(handles=leg, loc="lower center", ncol=3)
plt.tight_layout(rect=[0, 0.04, 1, 1])
plt.savefig(args.out, dpi=150)
print("saved", args.out)
