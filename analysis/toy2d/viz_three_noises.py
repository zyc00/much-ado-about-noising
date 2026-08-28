"""Three aleatoric uncertainty types, one panel each: speed (pace), direction
(tremor), z-turn (route choice). Each panel highlights only its own noise band
and renders demos of only that type; the speed panel adds per-step dots and a
y-vs-time inset (pace noise is spatially invisible)."""
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

import toytube as T

YS = np.linspace(-0.02, 1.4, 1000)
YLIM = (-0.05, 1.72)
CS = np.array([T.center(y) for y in YS])
HW = np.array([T.half_width(y) for y in YS])
XLIM = (-0.22, 0.30)

BANDS = {
    "speed": (T.Y_TUBE_LO, T.Y_TUBE_TOP),
    "zturn": (T.Y_STR_LO, T.Y_TUBE_LO),
    "dir": (T.Y_FUN_LO + 0.06, T.Y_STR_LO),
}


def draw_world(ax, active):
    for name, (lo, hi) in BANDS.items():
        if name == active:
            ax.axhspan(lo, hi, color="tab:red", alpha=0.16, zorder=0)
        else:
            ax.axhspan(lo, hi, color="grey", alpha=0.05, zorder=0)
    ax.axhspan(T.Y_DOCK, T.Y_FUN_LO, color="tab:green", alpha=0.08, zorder=0)
    zb = (YS > T.Y_STR_LO) & (YS <= T.Y_TUBE_LO)
    lo = np.where(zb, -T.Z_BOX, CS - HW)
    hi = np.where(zb, T.X_OFF + T.Z_BOX, CS + HW)
    ax.fill_betweenx(YS, lo, hi, color="grey", alpha=0.22, zorder=1)
    ax.plot(lo, YS, color="dimgrey", lw=1.4, zorder=2)
    ax.plot(hi, YS, color="dimgrey", lw=1.4, zorder=2)
    cz = np.array([T.x_target(y, T.T1_FIX, T.T2_FIX) for y in YS])
    ax.plot(cz, YS, "k--", lw=0.8, alpha=0.55, zorder=2)
    ax.plot([0], [0], "k*", ms=14, zorder=6)
    ax.set_xlim(*XLIM)
    ax.set_ylim(*YLIM)
    ax.set_xlabel("x")


def demos(noise_type, n, seed=0):
    T.NOISE_TYPE = noise_type
    rng = np.random.RandomState(seed)
    return [np.array(T.gen_episode(rng)[0]) for _ in range(n)]


fig, axes = plt.subplots(1, 3, figsize=(17.5, 7.2))

# ---- A: SPEED (pace) noise ------------------------------------------------
ax = axes[0]
draw_world(ax, "speed")
sp = demos("speed", 5, seed=3)
cl = demos("none", 1, seed=0)
for k, s in enumerate(sp):
    ax.plot(s[:, 0], s[:, 1], "-", lw=0.8, alpha=0.55, zorder=4,
            color=plt.cm.tab10(k))
    ax.plot(s[:, 0], s[:, 1], ".", ms=3.2, alpha=0.85, zorder=5,
            color=plt.cm.tab10(k))
ax.set_title("A  SPEED noise (tube band):\npace mixture "
             r"$\eta\!\cdot\!v_y$, $\eta\sim$clip$(1{+}1.2\,t(2))$, pauses",
             fontsize=11, loc="left")
ax.set_ylabel("height $y$")
ax.text(0.075, 1.26, "same spatial path;\nuncertainty lives in TIMING\n"
        "(dot spacing: bunched = pause,\nsparse = rush)", fontsize=9,
        color="tab:red", va="center")
ax.text(0.075, 0.15, "8-seed SR:  L2 0.68 / hetero-t 0.96", fontsize=9,
        color="0.25")
ins = ax.inset_axes([0.60, 0.33, 0.38, 0.25])
for k, s in enumerate(sp[:3]):
    ins.plot(np.arange(len(s)), s[:, 1], lw=1.0, color=plt.cm.tab10(k))
ins.plot(np.arange(len(cl[0])), cl[0][:, 1], "k--", lw=1.0)
ins.set_xlabel("step", fontsize=7)
ins.set_ylabel("$y$", fontsize=7)
ins.tick_params(labelsize=6)
ins.set_title("y vs time: plateaus = pauses\n(dashed = clean)", fontsize=7)

# ---- B: Z-TURN choice -----------------------------------------------------
ax = axes[1]
draw_world(ax, "zturn")
for k, s in enumerate(demos("zturn", 12, seed=2)):
    ax.plot(s[:, 0], s[:, 1], "-", lw=0.9, alpha=0.8, zorder=4)
ax.set_title("B  Z-TURN choice (Z band): each demo draws its own out/back\n"
             "turn heights $t_1{\\sim}U(0.68,0.86)$, $t_2{\\sim}U(0.52,0.58)$",
             fontsize=11, loc="left")
ax.text(0.105, 0.72, "uncertainty lives in the\nROUTE: which heights the\n"
        "detour turns at; the clean\nlaw (dashed) goes straight",
        fontsize=9, color="tab:red", va="center")
ax.text(0.105, 0.15, "8-seed SR:  L2 0.56 / hetero-t 1.00", fontsize=9,
        color="0.25")

# ---- C: DIRECTION noise ---------------------------------------------------
ax = axes[2]
draw_world(ax, "dir")
for k, s in enumerate(demos("dir", 10, seed=1)):
    ax.plot(s[:, 0], s[:, 1], "-", lw=0.9, alpha=0.8, zorder=4)
ax.set_title("C  DIRECTION noise (funnel band):\n$0.07\\,t(2)$ lateral tremor, "
             "wall-bounded", fontsize=11, loc="left")
ax.text(0.075, 0.42, "uncertainty lives in the\nlateral action: heavy-tailed\n"
        "wall-bounded wiggle;\naction labels keep the\nraw $t(2)$ kicks",
        fontsize=9, color="tab:red", va="center")
ax.text(0.075, 0.15, "8-seed SR:  L2 0.10 / hetero-t 0.94", fontsize=9,
        color="0.25")

fig.suptitle("Three aleatoric uncertainty types in disjoint task segments — "
             "timing (speed), route (z-turn), action direction (tremor) — all "
             "upstream of the same clean precision insert", fontsize=13)
fig.tight_layout(rect=[0, 0, 1, 0.93])
fig.savefig("three_noise_types.png", dpi=160)
print("wrote three_noise_types.png")
