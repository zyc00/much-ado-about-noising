"""Real-data kick/takeover figures, corrected instrument (all references =
training set mp_200). Fig 1: 3D kicked trajectories + d(t). Fig 2: deviation
source, kick response, drift rates, exemplar traces."""
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

z = np.load("analysis/kickpack5.npz", allow_pickle=True)
tags, doses, probes = z["tags"], z["doses"], z["probes"]
n = len(tags)
trajs = [z[f"traj_{i}"] for i in range(n)]
amags = [z[f"amag_{i}"] for i in range(n)]
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

# ---------------- Fig 1: kicked, 3D + d(t) ----------------
fig = plt.figure(figsize=(18.5, 9.2))
for j, tg in enumerate(order):
    ax = fig.add_subplot(2, 3, 1 + j, projection="3d")
    for d in demo_tracks:
        ax.plot(d[:, 0], d[:, 1], d[:, 2], color="gray", lw=0.6, alpha=0.35)
    nr = tot = 0
    for i in range(n):
        if tags[i] != tg or doses[i] == 0:
            continue
        cut, done = cut_of(i, hard=120)
        full = reached_full(i)
        tr = trajs[i]
        ax.plot(tr[:cut, 0], tr[:cut, 1], tr[:cut, 2], color="tab:orange",
                lw=1.5, alpha=0.9)
        ax.scatter(*tr[0], color="k", s=25, marker="o")
        if done:
            ax.scatter(*tr[cut - 1], color="tab:orange", s=70, marker="*")
        elif not full:
            ax.scatter(*np.clip(tr[min(cut, len(tr) - 1)], lo3 - pad,
                                hi3 + pad), color="red", s=55, marker="x")
        nr += full
        tot += 1
    ax.set_xlim(lo3[0] - pad[0], hi3[0] + pad[0])
    ax.set_ylim(lo3[1] - pad[1], hi3[1] + pad[1])
    ax.set_zlim(lo3[2] - pad[2], hi3[2] + pad[2])
    ax.set_title(f"{titles[tg]} — insertion zone reached {nr}/{tot}",
                 fontsize=11)
    ax.view_init(elev=22, azim=-60)

cur = []
for i in range(n):
    if tags[i] == "oracle" and doses[i] > 0:
        hard = int(185 - probes[i])
        c, _ = cut_of(i, hard=hard)
        cur.append(dsup[i][:c])
L = min(60, max(len(c) for c in cur))
med_or = np.array([np.median([c[t] for c in cur if len(c) > t])
                   for t in range(L)])
for j, tg in enumerate(order):
    ax = fig.add_subplot(2, 3, 4 + j)
    ax.plot(np.arange(L), np.maximum(med_or, 0.5), color="k", lw=1.8,
            label="recorded-action replay (median)")
    for i in range(n):
        if tags[i] != tg or doses[i] == 0:
            continue
        cut, done = cut_of(i, hard=100)
        full = reached_full(i)
        d = np.maximum(dsup[i][:cut], 0.5)
        ax.plot(np.arange(cut), d, color="tab:orange", lw=1.2, alpha=0.85)
        if done:
            ax.plot(cut - 1, d[-1], marker="*", color="tab:orange", ms=9,
                    mew=2, ls="none")
        elif not full:
            ax.plot(cut - 1, d[-1], marker="x", color="red", ms=9, mew=2,
                    ls="none")
    ax.set_yscale("log")
    ax.set_ylim(0.5, 250)
    ax.axhline(4, color="k", ls="--", lw=0.8)
    ax.text(60, 4.3, "d=4 (deep threshold)", fontsize=8)
    ax.set_xlabel("steps after kick")
    if j == 0:
        ax.set_ylabel("distance to training support (mm, log)")
        ax.legend(fontsize=8, loc="upper left")
    ax.set_title(f"{titles[tg]} — d(t) after 3.4 mm kick", fontsize=10.5)
    ax.grid(alpha=0.25)
fig.suptitle(
    "Script data (mp_200), kick response on the training route: position "
    "kick 1.0 action units $\\times$ 2 steps ($\\approx$3.4 mm), takeover at "
    "t=50/120, n=12 branches per arm. Top: first 120 steps, gray = demo "
    "bundle. Reach counted over the full continuation; * reached, x never "
    "reached.", fontsize=12, y=0.995)
fig.tight_layout()
fig.savefig("analysis/paper/real_kick_3d.png", dpi=150, bbox_inches="tight")

# ---------------- Fig 2: mechanism panels ----------------
fig2, axes = plt.subplots(1, 4, figsize=(20.5, 4.4))
cols = {"oracle": "gray", "l2": "tab:red", "hgcbest": "tab:blue",
        "mip": "tab:green"}
lbls = {"oracle": "recorded-action replay", "l2": "L2",
        "hgcbest": "HT+condreg", "mip": "MIP"}

ax = axes[0]
for tg in ["oracle", "l2", "hgcbest", "mip"]:
    for i in range(n):
        if tags[i] != tg or doses[i] != 0:
            continue
        hard = int(185 - probes[i])
        cut, _ = cut_of(i, hard=hard)
        d = np.maximum(dsup[i][:cut], 0.05)
        ax.plot(np.arange(cut), d, color=cols[tg], lw=1.1, alpha=0.75,
                label=lbls[tg] if i == min(
                    k for k in range(n)
                    if tags[k] == tg and doses[k] == 0) else None)
ax.set_yscale("log")
ax.set_ylim(0.05, 250)
ax.set_xlabel("steps after takeover (no kick)")
ax.set_ylabel("distance to training support (mm, log)")
ax.legend(fontsize=8, loc="upper left")
ax.set_title("A  takeover without kick, deterministic simulator.\n"
             "Replay stays at 0.1 mm; policy prediction error compounds",
             fontsize=10)
ax.grid(alpha=0.25)

ax = axes[1]
labels = ["replay", "L2", "HT+condreg", "MIP"]
medmax = [7.3, 103.4, 14.6, 27.4]
p90max = [15.5, 146.7, 45.9, 110.8]
x = np.arange(4)
colors = ["gray", "tab:red", "tab:blue", "tab:green"]
ax.bar(x - 0.19, medmax, 0.38, color=colors, alpha=0.9, label="median")
ax.bar(x + 0.19, p90max, 0.38, color=colors, alpha=0.45, hatch="//",
       label="p90")
ax.axhline(3.4, color="k", ls=":", lw=1.2)
ax.text(-0.4, 3.8, "kick displacement 3.4 mm", fontsize=8.5)
ax.set_yscale("log")
ax.set_ylim(1, 300)
ax.set_xticks(x)
ax.set_xticklabels(labels, fontsize=9)
ax.set_ylabel("max d within 40 steps of kick (mm, log)")
ax.legend(fontsize=8.5)
ax.set_title("B  peak excursion after a 3.4 mm kick (n=12/arm).\n"
             "L2 median 103 mm = 30$\\times$ the displacement",
             fontsize=10)
ax.grid(axis="y", alpha=0.25)

ax = axes[2]
dd_k = [-0.02, 2.04, 0.35, 1.41]
esc = [2, 7, 4, 6]
ret = [9, 1, 7, 5]
ax.bar(x, dd_k, 0.5, color=colors, alpha=0.9)
for xi, (e, r) in zip(x, zip(esc, ret)):
    ax.text(xi, max(dd_k[xi], 0) + 0.12,
            f"esc {e}/12\nret {r}/12", ha="center", fontsize=8)
ax.axhline(0, color="k", lw=0.8)
ax.set_xticks(x)
ax.set_xticklabels(labels, fontsize=9)
ax.set_ylabel("mean $\\Delta$d per step in 4<d<60 mm band")
ax.set_ylim(-0.4, 2.9)
ax.set_title("C  drift rate while off-support, kicked.\n"
             "esc: d32>1.5$\\times$d8; ret: returns toward band",
             fontsize=10)
ax.grid(axis="y", alpha=0.25)

ax = axes[3]
shown = {"l2": 0, "hgcbest": 0, "mip": 0}
pick = {}
for tg in order:
    cand = [(dsup[i][:40].max(), i) for i in range(n)
            if tags[i] == tg and doses[i] > 0]
    cand.sort()
    pick[tg] = cand[len(cand) // 2][1]
for tg in order:
    i = pick[tg]
    cut, done = cut_of(i, hard=100)
    d = np.maximum(dsup[i][:cut], 0.5)
    ax.plot(np.arange(cut), d, color=cols[tg], lw=1.9, label=lbls[tg])
    if done or not reached_full(i):
        ax.plot(cut - 1, d[-1], marker="*" if done else "x",
                color=cols[tg], ms=10, mew=2, ls="none")
ax.axhline(3.4, color="k", ls=":", lw=1)
ax.text(2, 3.6, "kick size", fontsize=8)
ax.set_yscale("log")
ax.set_ylim(0.5, 250)
ax.set_xlabel("steps after kick")
ax.set_ylabel("d (mm, log)")
ax.legend(fontsize=8.5)
ax.set_title("D  median-excursion branch per arm, same kick.\n"
             "* insertion zone reached, x not reached", fontsize=10)
ax.grid(alpha=0.25)

fig2.tight_layout()
fig2.savefig("analysis/paper/real_onset_chain.png", dpi=150,
             bbox_inches="tight")
print("saved both")
