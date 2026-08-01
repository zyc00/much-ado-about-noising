"""3D eef inference trajectories for HT and HG from roll_dump2 npz dumps.
Successes gray, failures colored by seed with escape marker. Saves PNG."""
import glob
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

fig = plt.figure(figsize=(16, 7))
for pi, (tag, dr) in enumerate([("HT (16/24)", "logs/rd_ht"),
                                ("HG (22/24)", "logs/rd_hg")]):
    ax = fig.add_subplot(1, 2, pi + 1, projection="3d")
    fails = 0
    for f in sorted(glob.glob(f"{dr}/ep_*.npz")):
        z = np.load(f)
        eef, asm, seed = z["eef"], int(z["asm"]), int(z["seed"])
        if asm:
            ax.plot(eef[:, 0], eef[:, 1], eef[:, 2], color="0.75", lw=0.6,
                    alpha=0.5)
        else:
            c = plt.cm.tab10(fails % 10)
            ax.plot(eef[:, 0], eef[:, 1], eef[:, 2], color=c, lw=1.4,
                    label=f"fail seed {seed}")
            ax.scatter(*eef[-1], color=c, marker="x", s=60)
            fails += 1
    ax.set_title(f"{tag} — {fails} failures colored, X = final eef")
    ax.set_xlabel("x")
    ax.set_ylabel("y")
    ax.set_zlabel("z")
    ax.legend(fontsize=6, loc="upper left")
    ax.view_init(elev=22, azim=-60)
plt.tight_layout()
plt.savefig("analysis/failvids/hthg_traj3d.png", dpi=140)
print("VIZ saved", flush=True)
import os
for tag, dr in [("ht", "logs/rd_ht"), ("hg", "logs/rd_hg")]:
    seeds = []
    for f in sorted(glob.glob(f"{dr}/ep_*.npz")):
        z = np.load(f)
        if not int(z["asm"]):
            seeds.append(int(z["seed"]))
    print(f"VIZ {tag} fail seeds: {seeds}", flush=True)
