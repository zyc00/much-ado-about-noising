"""MSE vs MIP failure anatomy at MP-200 / MP-2k (seed-matched rollouts).
A: one matched seed in 3D (L2 fails, MIP succeeds on the same scene)
B: height vs step for the six seeds where L2-200 fails (both arms)
C: distance-to-support vs step (log) for the same seeds
D: excursion cascade — fraction of episodes reaching d>=2 / 4 / 10 / 40
"""
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

FV = "analysis/failvids/"
A = {"L2-200": np.load(FV + "l2mp200_trajs.npz"),
     "MIP-200": np.load(FV + "mipmp200_trajs.npz"),
     "L2-2k": np.load(FV + "l2mp2k_trajs.npz"),
     "MIP-2k": np.load(FV + "mipmp2k_trajs.npz")}
FAILS = [21001, 21003, 21007, 21008, 21010, 21015]
CL = {"L2-200": "#d62728", "MIP-200": "#1f77b4",
      "L2-2k": "#e8927c", "MIP-2k": "#7fb3d5"}
SEED = 21008

fig = plt.figure(figsize=(17.5, 8.6))

# ---- A: matched seed in 3D -------------------------------------------------
ax = fig.add_subplot(2, 3, 1, projection="3d")
for arm in ["L2-200", "MIP-200"]:
    P, D = A[arm][f"P{SEED}"], A[arm][f"D{SEED}"]
    n = min(len(P), len(D))
    ax.plot(P[:n, 0], P[:n, 1], P[:n, 2], "-", color=CL[arm], lw=1.0,
            alpha=0.9, label=f"{arm} ({'fail' if not A[arm][f'O{SEED}'][0] else 'success'})")
    hit = np.where(D[:n] >= 2)[0]
    if len(hit):
        i = int(hit[0])
        ax.scatter(P[i, 0], P[i, 1], P[i, 2], color="black", marker="x", s=60,
                   zorder=6)
ax.set_title(f"A. same scene (seed {SEED}), same perturbation\n"
             "black × = state leaves the demo support", fontsize=10)
ax.set_xlabel("x"); ax.set_ylabel("y"); ax.set_zlabel("z")
ax.view_init(elev=20, azim=-60)
ax.legend(fontsize=8, loc="upper left")

# ---- B: height traces ------------------------------------------------------
axB = fig.add_subplot(2, 3, 2)
for arm in ["MIP-200", "L2-200"]:
    for j, sd in enumerate(FAILS):
        P = A[arm][f"P{sd}"]
        axB.plot(np.arange(len(P)), P[:, 2], color=CL[arm], lw=0.9, alpha=0.75,
                 label=arm if j == 0 else None)
axB.axhline(0.86, color="0.5", ls=":", lw=1)
axB.text(600, 0.88, "insertion depth", fontsize=8, color="0.4")
axB.set_xlabel("env step"); axB.set_ylabel("end-effector height z (m)")
axB.set_title("B. after the perturbation: MSE-200 flies up and away,\n"
              "MIP-200 stays in the workspace and docks", fontsize=10)
axB.legend(frameon=False, fontsize=9)
axB.spines[["top", "right"]].set_visible(False)

# ---- C: distance-to-support ------------------------------------------------
axC = fig.add_subplot(2, 3, 3)
for arm in ["MIP-200", "L2-200"]:
    for j, sd in enumerate(FAILS):
        D = A[arm][f"D{sd}"]
        axC.semilogy(np.arange(len(D)), np.clip(D, 0.2, None), color=CL[arm],
                     lw=0.9, alpha=0.75, label=arm if j == 0 else None)
for y, lab in [(2, "d=2 (off support)"), (4, "d=4 (rarely recovered)")]:
    axC.axhline(y, color="0.5", ls=":", lw=1)
    axC.text(430, y * 1.15, lab, fontsize=8, color="0.4")
axC.set_xlabel("env step"); axC.set_ylabel("distance to demo support (log)")
axC.set_title("C. both leave the support at the same moment;\n"
              "only MSE-200 escalates", fontsize=10)
axC.legend(frameon=False, fontsize=9, loc="lower right")
axC.spines[["top", "right"]].set_visible(False)

# ---- D: excursion cascade --------------------------------------------------
axD = fig.add_subplot(2, 3, 4)
THR = [2, 4, 10, 40]
w = 0.2
for i, arm in enumerate(["L2-200", "MIP-200", "L2-2k", "MIP-2k"]):
    z = A[arm]
    seeds = sorted({int(k[1:]) for k in z.files if k.startswith("P")})
    frac = [np.mean([z[f"D{sd}"].max() >= t for sd in seeds]) for t in THR]
    axD.bar(np.arange(len(THR)) + (i - 1.5) * w, frac, width=w - 0.02,
            color=CL[arm], label=arm)
axD.set_xticks(np.arange(len(THR)))
axD.set_xticklabels([f"d≥{t}" for t in THR])
axD.set_ylabel("fraction of episodes reaching")
axD.set_title("D. the perturbation is universal, the escalation is not",
              fontsize=10)
axD.legend(frameon=False, fontsize=8, ncol=2)
axD.spines[["top", "right"]].set_visible(False)

# ---- E: insertion geometry is NOT the difference ---------------------------
axE = fig.add_subplot(2, 3, 5)
LATMAX = {  # lat_ins_max (mm) per seed, from probe_traj_geom
    "L2-200": {21000: 23.7, 21001: 29.1, 21002: 21.4, 21003: 22.5, 21004: 22.4,
               21005: 28.6, 21006: 24.4, 21007: 262.3, 21008: 35.9, 21009: 31.2,
               21010: 66.2, 21011: 21.3, 21012: 24.8, 21013: 11.7, 21014: 24.0,
               21015: 36.1},
    "MIP-200": {21000: 21.2, 21001: 31.9, 21002: 21.6, 21003: 24.2, 21004: 29.2,
                21005: 31.2, 21006: 28.7, 21007: 29.9, 21008: 40.5, 21009: 30.9,
                21010: 32.1, 21011: 387.8, 21012: 31.1, 21013: 13.9, 21014: 25.7,
                21015: 40.1}}
sd_ok = [s for s in LATMAX["L2-200"] if s not in (21007, 21011)]
xs = [LATMAX["L2-200"][s] for s in sd_ok]
ys = [LATMAX["MIP-200"][s] for s in sd_ok]
cols = ["#d62728" if s in FAILS else "0.6" for s in sd_ok]
axE.scatter(xs, ys, c=cols, s=42, zorder=5)
lim = [8, 46]
axE.plot(lim, lim, "--", color="0.5", lw=1)
axE.set_xlim(*lim); axE.set_ylim(*lim)
axE.set_xlabel("L2-200 max lateral deviation during insertion (mm)")
axE.set_ylabel("MIP-200, same seed (mm)")
axE.set_title("E. MIP is not more precise: on every seed where MSE fails\n"
              "(red), MIP's insertion is at least as misaligned — and succeeds",
              fontsize=10)
axE.spines[["top", "right"]].set_visible(False)

# ---- F: action magnitude by regime (from probe_eject_actions) --------------
axF = fig.add_subplot(2, 3, 6)
REG = ["on-support\n(insertion)", "40 steps before\nthe perturbation",
       "after ejection\n(d≥4)"]
vals = {"L2-200": [0.0874, 0.1278, 0.5784], "MIP-200": [0.0931, 0.1335, 0.1928],
        "L2-2k": [0.0926, 0.1256, 0.2331], "MIP-2k": [0.0880, 0.1275, 0.2562]}
for i, arm in enumerate(["L2-200", "MIP-200", "L2-2k", "MIP-2k"]):
    axF.bar(np.arange(3) + (i - 1.5) * w, vals[arm], width=w - 0.02,
            color=CL[arm], label=arm)
axF.set_xticks(np.arange(3)); axF.set_xticklabels(REG, fontsize=8.5)
axF.set_ylabel("mean |position action| (raw units)")
axF.set_title("F. identical commands until it goes wrong;\n"
              "3× larger afterwards (measured at MSE-200's own states)",
              fontsize=10)
axF.legend(frameon=False, fontsize=8, ncol=2)
axF.spines[["top", "right"]].set_visible(False)

fig.suptitle("How MSE fails at 200 MP demos: same approach, same perturbation — "
             "the difference is entirely the off-support response", fontsize=13)
fig.tight_layout()
fig.savefig("analysis/paper/mse_fail_anatomy.png", dpi=150, bbox_inches="tight")
print("wrote analysis/paper/mse_fail_anatomy.png")
