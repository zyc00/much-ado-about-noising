"""3D eef trajectories: rotlim dataset vs original MP-200, colored by
local speed; side-by-side comparison figure."""
import h5py
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

fig = plt.figure(figsize=(18, 8))
for pi, (path, title) in enumerate([
        ("data/tool_hang_full2ins_mp_200.hdf5", "ORIGINAL MP-200"),
        ("data/tool_hang_mp200_rotlim.hdf5", "ROTLIM (rotation-budgeted)")]):
    h = h5py.File(path, "r")
    names = sorted(h["data"].keys(), key=lambda d: int(d.split("_")[1]))[:15]
    ax = fig.add_subplot(1, 2, pi + 1, projection="3d")
    Ls = []
    for dn in names:
        E = np.asarray(h[f"data/{dn}/obs/robot0_eef_pos"])
        Ls.append(len(E))
        sp = np.linalg.norm(np.diff(E, axis=0), axis=1) * 1000
        sc = ax.scatter(E[1:, 0], E[1:, 1], E[1:, 2], c=np.clip(sp, 0, 15),
                        cmap="viridis", s=3, vmin=0, vmax=15)
    h.close()
    plt.colorbar(sc, ax=ax, shrink=0.6, label="speed (mm/step)")
    ax.set_title(f"{title}\n15 demos, ep length p50 {int(np.median(Ls))} "
                 f"steps, points colored by speed")
    ax.view_init(elev=22, azim=-60)
plt.tight_layout()
plt.savefig("analysis/paper/rotlim_traj.png", dpi=130)
print("VR saved", flush=True)
