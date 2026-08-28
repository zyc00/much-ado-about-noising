"""One-segment mechanism evidence on script data: L2 natural failure vs
same-seed MIP success (seed 21003, natural twofactor episodes, no kick).
Left: local spatial segment. Right: canonical harness distance d(t)."""
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

z = np.load("analysis/natpack.npz", allow_pickle=True)
arms, asm, seeds = z["arms"], z["asm"], z["seeds"]
n = len(arms)


def get(arm, seed):
    for i in range(n):
        if arms[i] == arm and seeds[i] == seed:
            return z[f"eef_{i}"], z[f"dser_{i}"], int(asm[i])


SEED, TN = 21003, 198
eL, dL, aL = get("l2", SEED)
eM, dM, aM = get("mip", SEED)
demo_tracks = [np.asarray(d, np.float32) for d in z["demo_tracks"]]

# raw (y, z) front view — the miss is a lateral y offset (mm)
BX = (-30, 130)
BZ = (880, 1120)
AX = 1  # y


def crop(p):
    q = p * 1000.0
    m = ((q[:, AX] > BX[0]) & (q[:, AX] < BX[1]) &
         (q[:, 2] > BZ[0]) & (q[:, 2] < BZ[1]))
    return np.where(m[:, None], q, np.nan)


fig, axes = plt.subplots(1, 2, figsize=(13.2, 4.8))
ax = axes[0]
for pth in demo_tracks:
    q = crop(pth)
    ax.plot(q[:, AX], q[:, 2], color="gray", lw=1.0, alpha=0.4, zorder=1)
qM = crop(eM)
ax.plot(qM[:, AX], qM[:, 2], color="tab:green", lw=2.0, alpha=0.9, zorder=2,
        label="MIP, same placement (success)")
eMend = eM[-1] * 1000
ax.plot(eMend[AX], eMend[2], marker="*", color="tab:green", ms=15, zorder=4,
        ls="none")
qL = crop(eL[:TN + 22])
ax.plot(qL[:, AX], qL[:, 2], color="tab:red", lw=2.0, zorder=3,
        label="L2 (failure)")
seg = eL[TN - 50:TN + 22] * 1000
ax.scatter(seg[::5, AX], seg[::5, 2], color="tab:red", s=14, zorder=3)
dep = eL[TN] * 1000
ax.plot(dep[AX], dep[2], marker="o", mfc="none", mec="k", ms=13, mew=2,
        zorder=5, ls="none")
last = eL[TN + 21] * 1000
ax.plot(last[AX], last[2], marker="x", color="red", ms=12, mew=2.5,
        zorder=5, ls="none")
ax.annotate("insertion line y=44\n(demo ends $\\pm$0.4 mm)",
            xy=(44.8, 1029), xytext=(-22, 1080), fontsize=8.5,
            arrowprops=dict(arrowstyle="->", lw=0.9))
ax.annotate("L2 descends 16 mm beside the hole,\n50 mm past target "
            "depth", xy=(61, 990), xytext=(66, 930), fontsize=8.5,
            arrowprops=dict(arrowstyle="->", lw=0.9))
ax.annotate("departure (no-return point)", xy=(dep[AX], dep[2]),
            xytext=(dep[AX] - 85, dep[2] - 45), fontsize=9,
            arrowprops=dict(arrowstyle="->", lw=1))
ax.set_xlim(*BX)
ax.set_ylim(*BZ)
ax.set_xlabel("y (mm)")
ax.set_ylabel("z (mm)")
ax.legend(fontsize=8.5, loc="upper left")
ax.set_title(f"insertion-approach segment, seed {SEED} (fresh placement), "
             "front view (y, z)\ngray: demo bundle; dots every 5 steps",
             fontsize=10.5)
ax.grid(alpha=0.25)

ax = axes[1]
t0, t1 = TN - 90, TN + 25
ax.plot(np.arange(t0, len(dM)), np.maximum(dM[t0:], 0.3), color="tab:green",
        lw=1.9, label="MIP (ends in success)")
ax.plot(len(dM) - 1, max(dM[-1], 0.3), marker="*", color="tab:green",
        ms=14, ls="none")
ax.plot(np.arange(t0, t1), np.maximum(dL[t0:t1], 0.3), color="tab:red",
        lw=1.9, label="L2")
ax.plot(t1 - 1, dL[t1 - 1], marker="x", color="red", ms=11, mew=2.5,
        ls="none")
ax.axvline(TN, color="k", ls=":", lw=1)
ax.text(TN + 1, 0.42, "no-return point", fontsize=8.5, rotation=90)
ax.axhline(2, color="k", ls=":", lw=0.8)
ax.axhline(4, color="k", ls="--", lw=0.8)
ax.text(t0 + 2, 2.1, "d=2", fontsize=8)
ax.text(t0 + 2, 4.2, "d=4 (deep)", fontsize=8)
ax.set_yscale("log")
ax.set_ylim(0.3, 40)
ax.set_xlim(t0, t1 + 4)
ax.set_xlabel("episode step")
ax.set_ylabel("distance to nearest training state\n(harness units, log)")
ax.legend(fontsize=8.5, loc="upper left")
ax.set_title("same two episodes: d(t) around the departure\n"
             "L2 pre-mean 2.1 (30 steps) $\\to$ 16 within 5 steps; "
             "MIP max 1.1 in this window", fontsize=10.5)
ax.grid(alpha=0.25)

fig.suptitle(
    "Script data, natural episode (no kick, no takeover): the departure "
    "segment. L2 tracks in-band, exits at the insertion approach and moves "
    "away from support (motion-vs-support cosine $-$0.37, 83% of steps "
    "away); MIP completes the same placement.", fontsize=11.5, y=1.0)
fig.tight_layout()
fig.savefig("analysis/paper/departure_segment.png", dpi=150,
            bbox_inches="tight")
print("saved analysis/paper/departure_segment.png")
