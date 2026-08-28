"""Panel-D-style trajectory figure for the branch+noise witness.

Left: demonstrator (data) trajectories — per-EPISODE branch aim (0.8/0.2) +
per-chunk symmetric smear, executed through the servo. Right: trained-policy
rollouts (L2 / MIP full / HT nu=2, seed 0) from spread starts.

Run on a GPU (retrains the three arms at the standard recipe, ~4 min).
"""

import os
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
import torch  # noqa: E402
from toy2d_branchnoise import (  # noqa: E402
    A_BRANCH, HORIZON, P_MAJ, SIG, half_width_mm, make_ds,
)
from toy2d_mip_nfl import (  # noqa: E402
    ACTION_MM, FORWARD_MM, K_SERVO, X_GOAL_MM, normalize_state, predict,
    train, train_nll,
)

OUT = "analysis/paper/toy_branchnoise_trajs.png"
rng = np.random.default_rng(7)

fig, (axL, axR) = plt.subplots(1, 2, figsize=(17, 5.6), sharey=True)


def draw_funnel(ax):
    xs = np.linspace(0, X_GOAL_MM, 200)
    hw = half_width_mm(xs)
    ax.fill_between(xs, -hw, hw, color="#dbe4ee", zorder=0)
    ax.plot(xs, hw, color="0.45", lw=1.2)
    ax.plot(xs, -hw, color="0.45", lw=1.2)
    for b, lab in [(-A_BRANCH * 20, "majority branch (-6)"),
                   (A_BRANCH * 20, "minority branch (+6)")]:
        ax.axhline(b, color="0.55", lw=0.8, ls="--", zorder=1)
        ax.text(2, b + 0.35, lab, fontsize=9.5, color="0.35")
    ax.set_xlim(0, X_GOAL_MM + 3)
    ax.set_ylim(-15, 15)
    ax.set_xlabel("forward position x (mm)", fontsize=11)


# ---- left: demonstrator trajectories (per-episode aim, per-chunk smear) ----
draw_funnel(axL)
n_demo = 42
for i in range(n_demo):
    aim = -A_BRANCH if rng.random() < P_MAJ else A_BRANCH
    y = float(rng.uniform(-11, 11))
    x = 0.0
    xs, ys = [x], [y]
    step = 0
    r = 0.0
    while x < X_GOAL_MM:
        if step % HORIZON == 0:
            r = (aim + rng.normal(0, SIG)) * ACTION_MM
        y = y + (-K_SERVO * y + r)
        x += FORWARD_MM
        xs.append(x)
        ys.append(y)
        step += 1
    c = "tab:red" if aim < 0 else "tab:orange"
    axL.plot(xs, ys, color=c, alpha=0.5, lw=1.1, zorder=3)
axL.set_title("A   Training data: operators fork 80/20 to the two branches\n"
              "(per-episode aim + chunk-held symmetric smear, executed)",
              fontsize=12.5, loc="left")
axL.set_ylabel("lateral position y (mm)", fontsize=11)

# ---- right: trained policy rollouts ----
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
st, ac = make_ds(3000, 10, seed=1000)
arms = {}
arms["Regression (L2)"], _ = train("regression", st, ac, 0, 128, 12000, 512, 1e-3, device), None
arms["Regression (L2)"] = arms["Regression (L2)"][0]
mipnet, _ = train("mip", st, ac, 0, 128, 12000, 512, 1e-3, device)
htnet, _ = train_nll("ht", st, ac, 0, 128, 12000, 512, 1e-3, device)

draw_funnel(axR)
specs = [("Regression (L2)", arms["Regression (L2)"], "regression", "tab:blue"),
         ("MIP full", mipnet, "mip_full", "#7c3aed"),
         ("HT nu=2", htnet, "ht", "#c23b22")]
for lab, net, sampler, color in specs:
    for y0 in np.linspace(-10, 10, 9):
        x = np.array([0.0])
        y = np.array([float(y0)])
        xs, ys = [0.0], [float(y0)]
        for _ in range(70):
            stt = np.stack([x, y], axis=1).astype(np.float32)
            a = predict(net, stt, sampler, device)
            for j in range(HORIZON):
                x = x + np.clip(a[:, 2 * j] * ACTION_MM, -2.0, 8.0)
                y = y + np.clip(a[:, 2 * j + 1] * ACTION_MM, -12.0, 12.0)
                xs.append(float(x[0]))
                ys.append(float(y[0]))
            if x[0] >= X_GOAL_MM:
                break
        axR.plot(xs, ys, color=color, alpha=0.65, lw=1.4, zorder=3)
    axR.plot([], [], color=color, lw=2.5, label=lab)
axR.axhline(-3.6, color="tab:blue", lw=0.8, ls=":", zorder=1)
axR.text(122, -3.1, "mixture mean (-3.6)", fontsize=9, color="tab:blue")
axR.legend(fontsize=10.5, loc="upper right")
axR.set_title("B   Trained policies (seed 0): each converges to its estimator's point\n"
              "L2 to the mean, MIP between (on support), HT to the majority branch",
              fontsize=12.5, loc="left")

fig.tight_layout()
fig.savefig(OUT, dpi=160)
fig.savefig(OUT.replace(".png", ".pdf"))
print("saved", OUT, flush=True)
