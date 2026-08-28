"""Takeover-only (no kick) 3D figure: off-support behavior from rollout
error alone. Same instrument and reference as real_kick_3d.png."""
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

z = np.load("analysis/kickpack6.npz", allow_pickle=True)
tags, doses, probes = z["tags"], z["doses"], z["probes"]
n = len(tags)
trajs = [z[f"traj_{i}"] for i in range(n)]
dsup = [z[f"d_{i}"] for i in range(n)]
dins = [z[f"dins_{i}"] for i in range(n)]
demo_tracks = [np.asarray(d, np.float32) for d in z["demo_tracks"]]

INS_MM = 12.0
allp = np.concatenate(demo_tracks)
lo3, hi3 = allp.min(0), allp.max(0)
pad = 0.35 * (hi3 - lo3)


def cut_of(i, hard=None):
    hit = np.where(dins[i] < INS_MM)[0]
    cut, done = (int(hit[0]) + 1, True) if len(hit) else (len(trajs[i]),
                                                         False)
    if hard is not None and cut > hard:
        return hard, False
    return cut, done


def reached_full(i):
    return bool((dins[i] < INS_MM).any())


order = ["l2", "hgcbest", "mip"]
titles = {"l2": "L2", "hgcbest": "HT+condreg", "mip": "MIP"}

fig = plt.figure(figsize=(18.5, 9.2))
for j, tg in enumerate(order):
    ax = fig.add_subplot(2, 3, 1 + j, projection="3d")
    for d in demo_tracks:
        ax.plot(d[:, 0], d[:, 1], d[:, 2], color="gray", lw=0.6, alpha=0.35)
    nr = tot = 0
    nrp = {50: [0, 0], 120: [0, 0]}
    for i in range(n):
        if tags[i] != tg or doses[i] > 0:
            continue
        cut, done = cut_of(i, hard=160)
        full = reached_full(i)
        tr = trajs[i]
        ax.plot(tr[:cut, 0], tr[:cut, 1], tr[:cut, 2], color="tab:green",
                lw=1.6, alpha=0.95)
        ax.scatter(*tr[0], color="k", s=25, marker="o")
        if done:
            ax.scatter(*tr[cut - 1], color="tab:green", s=70, marker="*")
        elif not full:
            ax.scatter(*np.clip(tr[min(cut, len(tr) - 1)], lo3 - pad,
                                hi3 + pad), color="red", s=55, marker="x")
        nr += full
        tot += 1
        nrp[int(probes[i])][0] += full
        nrp[int(probes[i])][1] += 1
    ax.set_xlim(lo3[0] - pad[0], hi3[0] + pad[0])
    ax.set_ylim(lo3[1] - pad[1], hi3[1] + pad[1])
    ax.set_zlim(lo3[2] - pad[2], hi3[2] + pad[2])
    ax.set_title(f"{titles[tg]} — insertion reached {nr}/{tot}  "
                 f"(t=50: {nrp[50][0]}/{nrp[50][1]}, "
                 f"t=120: {nrp[120][0]}/{nrp[120][1]})", fontsize=11)
    ax.view_init(elev=22, azim=-60)

for j, tg in enumerate(order):
    ax = fig.add_subplot(2, 3, 4 + j)
    first = True
    for i in range(n):
        if tags[i] == "oracle" and doses[i] == 0:
            hard = int(185 - probes[i])
            cut, _ = cut_of(i, hard=hard)
            ax.plot(np.arange(cut), np.maximum(dsup[i][:cut], 0.05),
                    color="k", lw=1.0, alpha=0.6,
                    label="recorded-action replay" if first else None)
            first = False
    for i in range(n):
        if tags[i] != tg or doses[i] > 0:
            continue
        cut, done = cut_of(i, hard=160)
        full = reached_full(i)
        d = np.maximum(dsup[i][:cut], 0.05)
        ax.plot(np.arange(cut), d, color="tab:green", lw=1.3, alpha=0.9)
        if done:
            ax.plot(cut - 1, d[-1], marker="*", color="tab:green", ms=10,
                    mew=2, ls="none")
        elif not full:
            ax.plot(cut - 1, d[-1], marker="x", color="red", ms=9, mew=2,
                    ls="none")
    ax.set_yscale("log")
    ax.set_ylim(0.05, 250)
    ax.axhline(2, color="k", ls=":", lw=0.8)
    ax.axhline(4, color="k", ls="--", lw=0.8)
    ax.text(100, 2.1, "d=2 (band entry)", fontsize=8)
    ax.text(100, 4.3, "d=4 (deep)", fontsize=8)
    ax.set_xlabel("steps after takeover")
    if j == 0:
        ax.set_ylabel("distance to training support (mm, log)")
        ax.legend(fontsize=8, loc="upper left")
    ax.set_title(f"{titles[tg]} — d(t), no kick", fontsize=10.5)
    ax.grid(alpha=0.25)

fig.suptitle(
    "Script data (mp_200), takeover WITHOUT kick (n=16/arm, fresh "
    "latent per chunk): the policy replaces the recorded actions at "
    "t=50/120 on its own training route. t=120 takeover carries a "
    "$\\approx$1.5 mm controller transient (measured on the replay); "
    "t=50 carries none. * insertion zone reached, x never reached.",
    fontsize=12, y=0.995)
fig.tight_layout()
fig.savefig("analysis/paper/real_takeover_3d.png", dpi=150,
            bbox_inches="tight")
print("saved analysis/paper/real_takeover_3d.png")
