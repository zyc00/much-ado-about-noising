"""The three-mechanism composite: columns = idle / block detour / slot slalom;
top row = demonstration data, bottom row = trained policies (MSE blue,
diffusion purple, HT red-orange in every panel).

Usage: python scripts/fig_toy2d_mechanisms.py
"""

import json

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

C_MSE, C_DP, C_HT = "#2b6cb0", "#6b46c1", "#c23b22"
XG = 160.0
K, FWD, HORIZON = 0.5, 4.0, 8


def load(path):
    return json.load(open(path))["arms"]


idle = load("analysis/toy2d_idle_fig.json")
blk_l2 = load("analysis/toy2d_obstacle_l2.json")
blk_ht = load("analysis/toy2d_obstacle_ht.json")
blk_dp = load("analysis/toy2d_obstacle_dpmem.json")
slm = load("analysis/toy2d_slalom.json")

# ---- scenes ------------------------------------------------------------
def hw_funnel(x):
    return 5.0 + 9.0 * (1.0 - np.clip(x / 130.0, 0.0, 1.0))


def hw_wide(x):
    return 14.0 - 1.5 * np.clip(x / XG, 0.0, 1.0)


SLOTS = np.array([0.0, -6.0, -11.0])
GAP = 1.2
PLATES = [(50.0, 58.0), (100.0, 108.0)]
SWITCH = [62.0, 112.0]
BLK_X0, BLK_X1, BLK_HALF, DET_A, AIM0, AIM1 = 60.0, 100.0, 3.0, 6.0, 20.0, 100.0


def scene(ax, kind):
    hw = hw_funnel if kind == "idle" else hw_wide if kind == "slalom" else hw_funnel
    xs = np.linspace(0, XG, 200)
    ax.fill_between(xs, -hw(xs), hw(xs), color="#dfe7f0", zorder=0)
    ax.plot(xs, hw(xs), color="0.45", lw=1.0)
    ax.plot(xs, -hw(xs), color="0.45", lw=1.0)
    if kind == "block":
        ax.add_patch(plt.Rectangle((BLK_X0, -BLK_HALF), BLK_X1 - BLK_X0, 2 * BLK_HALF,
                                   facecolor="0.35", edgecolor="0.2", zorder=2))
    if kind == "slalom":
        for x0, x1 in PLATES:
            h = float(hw_wide((x0 + x1) / 2))
            for lo, hi in [(SLOTS[0] + GAP, h), (SLOTS[1] + GAP, SLOTS[0] - GAP),
                           (SLOTS[2] + GAP, SLOTS[1] - GAP), (-h, SLOTS[2] - GAP)]:
                if hi > lo:
                    ax.add_patch(plt.Rectangle((x0, lo), x1 - x0, hi - lo,
                                               facecolor="0.35", edgecolor="0.2", zorder=2))
    ax.plot(XG, 0, "*", color="tab:green", ms=13, zorder=6)
    ax.set_xlim(0, XG + 4)
    ax.set_ylim(-15, 15)
    ax.tick_params(labelsize=8.5)


# ---- figure ------------------------------------------------------------
fig, axes = plt.subplots(2, 3, figsize=(17, 7.6), sharex=True, sharey=True)
fig.suptitle("Three failure mechanisms from ~100%-success demonstrators "
             "(all behavior executed, no injected corruption)",
             fontsize=15, fontweight="bold")

COLTITLES = [
    "Idle: 70% of recorded chunks are pauses",
    "Block detour: minority mode is a legitimate branch",
    "Slot slalom: sequential three-way commitments",
]

rng = np.random.default_rng(7)

# top-left: idle data — executed clean servo + recorded pause chunks (dots)
ax = axes[0, 0]; scene(ax, "idle")
for i in range(14):
    y = float(rng.uniform(-10, 10)); x = 0.0
    tx, ty = [x], [y]
    while x < XG:
        y = y - K * y; x += FWD
        tx.append(x); ty.append(y)
    ax.plot(tx, ty, color="0.55", alpha=0.55, lw=0.9, zorder=3)
    for cx in range(0, int(XG), int(FWD * HORIZON)):
        if rng.random() < 0.7:
            j = cx // int(FWD)
            ax.plot(tx[min(j, len(tx) - 1)], ty[min(j, len(ty) - 1)], "o",
                    color="tab:red", ms=4.5, zorder=4)
ax.plot([], [], color="0.55", lw=1.4, label="executed demo (clean servo)")
ax.plot([], [], "o", color="tab:red", ms=5, label="recorded pause chunk (70%)")
ax.legend(fontsize=8, loc="lower left")
ax.set_ylabel("demonstrations\n\nlateral y (mm)", fontsize=10.5)

# top-middle: block data — forked detours
ax = axes[0, 1]; scene(ax, "block")
for i in range(26):
    branch = 1.0 if rng.random() < 0.6 else -1.0
    y = float(rng.uniform(-10, 10)); x = 0.0
    tx, ty = [x], [y]
    while x < XG:
        yt = branch * DET_A if (AIM0 <= x <= BLK_X1) else 0.0
        y = y - K * (y - yt); x += FWD
        tx.append(x); ty.append(y)
    ax.plot(tx, ty, color="tab:red" if branch > 0 else "tab:orange",
            alpha=0.45, lw=0.9, zorder=3)
ax.plot([], [], color="tab:red", lw=1.4, label="detour up (60%)")
ax.plot([], [], color="tab:orange", lw=1.4, label="detour down (40%)")
ax.legend(fontsize=8, loc="lower left")

# top-right: slalom data
ax = axes[0, 2]; scene(ax, "slalom")
for i in range(22):
    ks = rng.choice(3, p=[0.6, 0.28, 0.12], size=2)
    lanes = [float(SLOTS[k] + np.clip(rng.normal(0, [0.08, 0.3, 0.5][k]), -0.9, 0.9)) for k in ks]
    y = float(rng.uniform(-10, 10)); x = float(rng.uniform(0, FWD * HORIZON))
    tx, ty = [x], [y]
    while x < XG:
        lane = lanes[0] if x < SWITCH[0] else (lanes[1] if x < SWITCH[1] else 0.0)
        y = y - K * (y - lane); x += FWD
        tx.append(x); ty.append(y)
    ax.plot(tx, ty, color="0.5", alpha=0.5, lw=0.9, zorder=3)
ax.plot([], [], color="0.5", lw=1.4, label="lane choice per plate (60/28/12)")
ax.legend(fontsize=8, loc="lower left")

# bottom row: trained policies -------------------------------------------
PANELS = [
    ("idle", [(idle, "l2", "MSE", C_MSE), (idle, "flow", "Flow matching", C_DP),
              (idle, "ht2", "HT", C_HT)]),
    ("block", [(blk_l2, "l2", "MSE", C_MSE), (blk_dp, "flow_hist", "Flow matching +history", C_DP),
               (blk_ht, "ht05", "HT", C_HT)]),
    ("slalom", [(slm, "l2", "MSE", C_MSE), (slm, "flow", "Flow matching", C_DP),
                (slm, "ht05", "HT", C_HT)]),
]
for col, (kind, entries) in enumerate(PANELS):
    ax = axes[1, col]; scene(ax, kind)
    for src, key, lab, c in entries:
        a = src[key]
        for t in a["seed0"]["traces"]:
            ax.plot(t["x"], t["y"], color=c, alpha=0.55, lw=1.0, zorder=3)
        ax.plot([], [], color=c, lw=2.0, label=f"{lab}  SR {a['sr_mean']:.2f}")
    ax.legend(fontsize=8, loc="lower left")
    ax.set_xlabel("forward x (mm)", fontsize=10)
axes[1, 0].set_ylabel("trained policies\n\nlateral y (mm)", fontsize=10.5)

# idle: HT freezes at the start line — make the zero-motion failure visible
ax = axes[1, 0]
for t in idle["ht2"]["seed0"]["traces"]:
    ax.plot(t["x"][0], t["y"][0], "o", color=C_HT, ms=6, zorder=6,
            markeredgecolor="0.2")
ax.annotate("HT: commits to the pause mode,\nnever moves", xy=(1.5, 6.0),
            xytext=(28, 10.5), fontsize=9, color=C_HT,
            arrowprops=dict(arrowstyle="->", color=C_HT, lw=1.2))

VERDICTS = ["mean family wins; committers freeze",
            "committers win; mean family collides",
            "only HT wins; churn and averaging both collide"]
for col in range(3):
    axes[0, col].set_title(f"{chr(65 + col)}   {COLTITLES[col]}", fontsize=11.5,
                           fontweight="bold", loc="left")
    axes[1, col].set_title(VERDICTS[col], fontsize=10.5, loc="left", style="italic")

fig.tight_layout(rect=[0, 0.01, 1, 0.95])
fig.savefig("analysis/paper/toy2d_mechanisms.png", dpi=160)
fig.savefig("analysis/paper/toy2d_mechanisms.pdf")
print("saved analysis/paper/toy2d_mechanisms.png")
