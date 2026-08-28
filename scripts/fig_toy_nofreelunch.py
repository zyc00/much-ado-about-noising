"""No-free-lunch toy figure: corridor label-noise sweep (h=48/3k) +
mechanism spread + the long-route boundary case."""
import pickle

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

with open("analysis/toy2d_corr48.pkl", "rb") as fh:
    d = pickle.load(fh)
res, spread = d["res1b"], d["spread"]
SIG = [0.0, 1.0, 2.0, 3.0]
METH = ["L2", "MIP", "Flow"]
COL = {"L2": "tab:red", "MIP": "tab:green", "Flow": "tab:blue"}

fig, axes = plt.subplots(1, 5, figsize=(21.5, 4.1),
                         gridspec_kw={"width_ratios": [1.15, 1, 1, 1, 1]})

ax = axes[0]
KP, CAP = 0.3, 4.0
rng = np.random.RandomState(0)
for y0 in np.linspace(-15, 15, 12):
    p = np.array([0.0, y0])
    tr = [p.copy()]
    for _ in range(40):
        a = np.clip(KP * (np.array([50.0, 0.0]) - p), -CAP, CAP)
        p = p + a
        tr.append(p.copy())
        if p[0] >= 45:
            break
    tr = np.array(tr)
    ax.plot(tr[:, 0], tr[:, 1], color="gray", lw=1.0, alpha=0.5)
p = np.array([0.0, -12.0])
for _ in range(16):
    a = np.clip(KP * (np.array([50.0, 0.0]) - p), -CAP, CAP)
    an = a + rng.randn(2) * 2.0
    ax.annotate("", xy=(p[0] + an[0] * 1.6, p[1] + an[1] * 1.6),
                xytext=(p[0], p[1]),
                arrowprops=dict(arrowstyle="->", color="tab:orange",
                                lw=1.2, alpha=0.85))
    p = p + a
ax.axvline(40, color="k", lw=1)
ax.add_patch(plt.Rectangle((39.6, -1), 0.8, 2, color="tab:green",
                           alpha=0.6))
ax.text(40.7, -0.4, "gate: |y| < 1 mm", fontsize=9)
ax.set_xlim(-3, 55)
ax.set_ylim(-17, 17)
ax.set_xlabel("x (mm)")
ax.set_ylabel("y (mm)")
ax.set_title("A  task: 15-step corridor, tight gate.\norange: noisy "
             "action labels ($\\sigma$=2) on one demo", fontsize=10)
ax.grid(alpha=0.25)

ax = axes[1]
for m in METH:
    ax.plot(SIG, [res[(m, s)][0] for s in SIG], "-o", color=COL[m],
            lw=2, label=m)
ax.set_xlabel("label noise $\\sigma$ (mm)")
ax.set_ylabel("SR @ 1 mm gate")
ax.set_ylim(0, 1.05)
ax.legend(fontsize=9)
ax.set_title("B  success vs label noise\n(h=48, 3k steps; all 1.00 at "
             "$\\sigma$=0)", fontsize=10)
ax.grid(alpha=0.25)

ax = axes[2]
for m in METH:
    ax.plot(SIG, [res[(m, s)][2] for s in SIG], "-o", color=COL[m], lw=2)
ax.axhline(1.0, color="k", ls=":", lw=0.8)
ax.set_xlabel("label noise $\\sigma$ (mm)")
ax.set_ylabel("|y| at gate, p90 (mm)")
ax.set_title("C  gate error growth\nFlow $\\propto\\sigma$; L2/MIP "
             "denoise", fontsize=10)
ax.grid(alpha=0.25)

ax = axes[3]
for m in METH:
    aa = spread[m]
    ax.scatter(aa[:, 0], aa[:, 1], s=8, color=COL[m], alpha=0.5,
               label=f"{m} (std {aa[:,1].std():.2f})")
ax.set_xlabel("sampled $a_x$")
ax.set_ylabel("sampled $a_y$")
ax.legend(fontsize=8)
ax.set_title("D  200 sampled actions, one state,\ntrained at "
             "$\\sigma$=2: Flow re-injects $\\sigma$", fontsize=10)
ax.grid(alpha=0.25)

ax = axes[4]
vals = [0.22, 0.67, 1.00]
ax.bar(range(3), vals, color=[COL[m] for m in METH], alpha=0.85)
for i, v in enumerate(vals):
    ax.text(i, v + 0.02, f"{v:.2f}", ha="center", fontsize=10)
ax.set_xticks(range(3))
ax.set_xticklabels(METH)
ax.set_ylim(0, 1.1)
ax.set_ylabel("SR (clean labels)")
ax.set_title("E  boundary case: long fan+merge route,\novertrained "
             "(h=256/12k): instability owns L2", fontsize=10)
ax.grid(axis="y", alpha=0.25)

fig.suptitle(
    "No-free-lunch toy: Gaussian action-label noise + tight tolerance. "
    "Mean-seeking methods (L2, MIP) denoise; Flow reproduces the "
    "conditional distribution and re-injects the noise. On long "
    "route-structured data (E) the ordering inverts against L2.",
    fontsize=11.5, y=1.02)
fig.tight_layout()
fig.savefig("analysis/paper/toy_nofreelunch_fig.png", dpi=150,
            bbox_inches="tight")
print("saved analysis/paper/toy_nofreelunch_fig.png")
