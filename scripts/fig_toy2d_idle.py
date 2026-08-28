"""Idle-cell visualization: pause-heavy demos (top) vs trained policies (bottom)."""
import json
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

P_IDLE, K, FWD, XG, HOLD = 0.6, 0.5, 4.0, 160.0, 8
d = json.load(open("analysis/toy2d_idle.json"))

fig, (axT, axB) = plt.subplots(2, 1, figsize=(16, 9.5), sharex=True, sharey=True)
fig.suptitle("Idle-mode cell: 60% of recorded chunks are pauses — the mode is inaction",
             fontsize=16, fontweight="bold")


def scene(ax):
    xs = np.linspace(0, XG, 200)
    hw = 5.0 + 9.0 * (1.0 - np.clip(xs / 130.0, 0.0, 1.0))
    ax.fill_between(xs, -hw, hw, color="#dfe7f0", zorder=0)
    ax.plot(xs, hw, color="0.45", lw=1.2); ax.plot(xs, -hw, color="0.45", lw=1.2)
    ax.plot(XG, 0, "*", color="tab:green", ms=18, zorder=4)
    ax.set_xlim(-2, XG + 4); ax.set_ylim(-15, 15)
    ax.set_ylabel("lateral position y (mm)", fontsize=11.5)


# top: demos as x(t) shown via time-embedded markers — plot y vs x with pause dots
scene(axT)
rng = np.random.default_rng(3)
for i in range(26):
    y = float(rng.uniform(-10, 10)); x = 0.0
    tx, ty, px, py = [x], [y], [], []
    for c in range(70):
        if rng.random() < P_IDLE:
            px.append(x); py.append(y)  # paused chunk: no motion
            continue
        for _ in range(HOLD):
            y = y + (-K * y); x += FWD
            tx.append(x); ty.append(y)
        if x >= XG: break
    axT.plot(tx, ty, color="tab:orange", alpha=0.5, lw=1.1, zorder=3)
    axT.plot(px, py, "o", color="tab:red", ms=3.5, alpha=0.5, zorder=4)
axT.set_title("A   Demonstrations: red dots = recorded PAUSE chunks (60%, action=0, no state cue); "
              "all demos still reach the goal — oracle 1.000",
              fontsize=12, fontweight="bold", loc="left")

# bottom: trained policies
scene(axB)
for k, lab, c in [("l2", "MSE (mean)", "#2b6cb0"), ("ht2", "HT nu=2", "#dd6b20"),
                  ("ht05", "HT nu=0.5", "#c23b22"), ("flow", "Flow matching", "#2f855a")]:
    if not d["arms"][k].get("seed0"):
        continue
    for t in d["arms"][k]["seed0"]["traces"]:
        axB.plot(t["x"], t["y"], color=c, alpha=0.6, lw=1.4, zorder=3)
        axB.plot(t["x"][-1], t["y"][-1], "o", color=c, ms=4, zorder=4)
    axB.plot([], [], color=c, lw=2.4, label=f"{lab}  (SR {d['arms'][k]['sr_mean']:.2f})")
axB.set_title("B   Trained policies: mean family and diffusion travel to the goal; "
              "HT nu=0.5 commits to the pause mode and PARKS (dots at start)",
              fontsize=12, fontweight="bold", loc="left")
axB.set_xlabel("forward position x (mm)", fontsize=11.5)
axB.legend(fontsize=10.5, loc="upper right")

fig.tight_layout(rect=[0, 0.01, 1, 0.965])
fig.savefig("analysis/paper/toy2d_idle_trajs.png", dpi=160)
fig.savefig("analysis/paper/toy2d_idle_trajs.pdf")
print("saved")
