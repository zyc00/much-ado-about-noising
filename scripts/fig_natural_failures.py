"""Natural full episodes (twofactor harness, no kick, no takeover):
3D trajectories + d(t) per arm. Failures develop from the policy's own
prediction error only."""
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

z = np.load("analysis/natpack.npz", allow_pickle=True)
arms, asm = z["arms"], z["asm"]
n = len(arms)
eefs = [z[f"eef_{i}"] for i in range(n)]
dsers = [z[f"dser_{i}"] for i in range(n)]
demo_tracks = [np.asarray(d, np.float32) for d in z["demo_tracks"]]

allp = np.concatenate(demo_tracks)
lo3, hi3 = allp.min(0), allp.max(0)
pad = 0.3 * (hi3 - lo3)

order = ["l2", "hgcbest", "mip"]
titles = {"l2": "L2", "hgcbest": "HT+condreg", "mip": "MIP"}

fig = plt.figure(figsize=(18.5, 9.4))
for j, tg in enumerate(order):
    ax = fig.add_subplot(2, 3, 1 + j, projection="3d")
    for d in demo_tracks:
        ax.plot(d[:, 0], d[:, 1], d[:, 2], color="gray", lw=0.6, alpha=0.35)
    ns = nf = 0
    for i in range(n):
        if arms[i] != tg:
            continue
        tr = eefs[i]
        if asm[i]:
            ax.plot(tr[:, 0], tr[:, 1], tr[:, 2], color="tab:green", lw=0.9,
                    alpha=0.55, zorder=1)
            ns += 1
        else:
            ax.plot(tr[:, 0], tr[:, 1], tr[:, 2], color="tab:red", lw=1.7,
                    alpha=0.95, zorder=3)
            ax.scatter(*np.clip(tr[-1], lo3 - pad, hi3 + pad), color="red",
                       s=60, marker="x", zorder=4)
            nf += 1
    ax.set_xlim(lo3[0] - pad[0], hi3[0] + pad[0])
    ax.set_ylim(lo3[1] - pad[1], hi3[1] + pad[1])
    ax.set_zlim(lo3[2] - pad[2], hi3[2] + pad[2])
    ax.set_title(f"{titles[tg]} — success {ns}, failure {nf} (of 30)",
                 fontsize=11)
    ax.view_init(elev=22, azim=-60)

for j, tg in enumerate(order):
    ax = fig.add_subplot(2, 3, 4 + j)
    c4 = 0
    for i in range(n):
        if arms[i] != tg:
            continue
        d = np.maximum(dsers[i], 0.2)
        if asm[i]:
            ax.plot(np.arange(len(d)), d, color="tab:green", lw=0.8,
                    alpha=0.5, zorder=1)
        else:
            ax.plot(np.arange(len(d)), d, color="tab:red", lw=1.5,
                    alpha=0.95, zorder=3)
        c4 += bool((dsers[i] >= 4.0).any())
    ax.set_yscale("log")
    ax.set_ylim(0.2, 2000)
    ax.axhline(2, color="k", ls=":", lw=0.9)
    ax.axhline(4, color="k", ls="--", lw=0.9)
    ax.text(5, 2.15, "d=2 (band entry)", fontsize=8)
    ax.text(5, 4.3, "d=4 (deep)", fontsize=8)
    ax.set_xlabel("episode step")
    if j == 0:
        ax.set_ylabel("distance to nearest training state\n"
                      "(normalized units, log)")
    ax.set_title(f"{titles[tg]} — d(t); episodes crossing d=4: {c4}/30",
                 fontsize=10.5)
    ax.grid(alpha=0.25)

fig.suptitle(
    "Natural full episodes, canonical twofactor harness (fresh placements, "
    "seeds 21000–21029, no kick, no takeover, deterministic simulator): "
    "green = success, red = failure. All deviation originates from the "
    "policy's own prediction error. SR: L2 19/30, HT+condreg 27/30, "
    "MIP 28/30.", fontsize=12, y=0.995)
fig.tight_layout()
fig.savefig("analysis/paper/real_natural_3d.png", dpi=150,
            bbox_inches="tight")
print("saved analysis/paper/real_natural_3d.png")
