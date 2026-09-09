"""GR1 (GR00T N1.7, 24 tasks x 20 episodes @60k) success rate vs the Student-t degrees of freedom nu.
nu = c^2 * d with d = 232 (29 dims x 8 steps); c is annotated on every point. HG = heteroscedastic Gaussian (nu -> inf),
whose run was stopped at 22k of 60k steps (hollow marker). Error bars: binomial SE over 480 episodes."""
import numpy as np, matplotlib
matplotlib.use("Agg"); import matplotlib.pyplot as plt
d = 232
pts = [(2, 24.5), (116, 44.8), (464, 47.5), (928, 51.5), (1450, 47.6)]
hg = 20.6
flow, mse = 44.1, 37.8
se = lambda p: 100 * np.sqrt(p / 100 * (1 - p / 100) / 480)
fig, ax = plt.subplots(figsize=(5.6, 3.3))
x = list(range(len(pts))); y = [s_ for _, s_ in pts]; xhg = len(pts) + 0.6
ax.axhline(flow, color="#6c757d", lw=1, ls="--"); ax.text(-0.45, flow + 0.7, "flow (4 NFE) 44.1", fontsize=8, color="#6c757d")
ax.axhline(mse, color="#adb5bd", lw=1, ls=":"); ax.text(-0.45, mse + 0.7, "MSE 37.8", fontsize=8, color="#868e96")
ax.errorbar(x, y, yerr=[se(v) for v in y], fmt="o-", color="#1b6ca8", lw=1.6, ms=6, capsize=2, zorder=3)
ax.errorbar([xhg], [hg], yerr=[se(hg)], fmt="o", mfc="white", mec="#1b6ca8", color="#1b6ca8", ms=6, capsize=2, zorder=3)
for (n, s_), xx in zip(pts, x):
    ax.annotate(f"{s_:.1f}", (xx, s_), xytext=(0, 8), textcoords="offset points", ha="center", fontsize=8.5)
ax.annotate(f"{hg:.1f}\n(run stopped\nat 22k / 60k)", (xhg, hg), xytext=(0, 8), textcoords="offset points", ha="center", fontsize=8)
labels = [f"{n}\nc = {np.sqrt(n / d):.2f}" for n, _ in pts] + ["$\\infty$ (hetero.\nGaussian)\nc = $\\infty$"]
ax.set_xticks(x + [xhg]); ax.set_xticklabels(labels, fontsize=8)
ax.set_xlabel("Student-t degrees of freedom $\\nu = c^2 d$  (d = 232)", fontsize=9)
ax.set_ylabel("success rate (%)", fontsize=9); ax.set_ylim(10, 60); ax.set_xlim(-0.55, xhg + 0.5)
ax.tick_params(axis="y", labelsize=8); ax.spines[["top", "right"]].set_visible(False)
ax.set_title("GR00T N1.7 on RoboCasa GR1: HT head, 24 tasks x 20 episodes, 60k steps", fontsize=9)
plt.tight_layout(); plt.savefig("fig_gr1_nu_sweep.png", dpi=200); plt.savefig("fig_gr1_nu_sweep.pdf"); print("saved")
