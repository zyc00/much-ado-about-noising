"""HT+condreg (hgcbest) mechanism figure — measured values only.

Source: eval_twofactor, N=200 episodes/arm, mp200 checkpoints (model_latest),
dataset tool_hang_full2ins_mp_200. The SR decomposition per arm:
  SR is set by deep excursions (SR|stay = 100% for every arm), and the
  deep-crossing rate factors into (band-entry rate) x (escalation given
  entry). HT+condreg improves BOTH factors through two distinct channels:
  the estimation channel (HT: variance absorption -> fewer entries) and
  the constraint channel (better excursion outcomes: highest return
  fraction, lowest escalation).
"""
import matplotlib
matplotlib.use("Agg")
import numpy as np
import matplotlib.pyplot as plt

arms = ["l2", "htrank\n(untuned)", "ht", "ht+condreg\n(hgcbest)", "mip"]
sr = [51.0, 61.0, 74.0, 80.5, 84.0]
cross2 = [0.95, 0.94, 0.85, 0.85, 0.90]          # band-entry rate/episode
cross4 = [0.61, 0.51, 0.30, 0.24, 0.22]          # deep rate/episode
ret20 = [0.53, 0.51, 0.55, 0.64, 0.51]           # excursion return<2 @20
esc20 = [0.30, 0.29, 0.21, 0.17, 0.18]           # excursion cross4 @20
colors = ["tab:red", "tab:purple", "tab:orange", "tab:blue", "tab:green"]
x = np.arange(len(arms))

fig, axes = plt.subplots(1, 3, figsize=(17.5, 4.9))

ax = axes[0]
w = 0.38
ax.bar(x - w / 2, cross2, w, color="lightgray", edgecolor="k",
       label="band-entry rate (d$\\geq$2) per episode")
ax.bar(x + w / 2, cross4, w, color=colors, alpha=0.9,
       label="deep-crossing rate (d$\\geq$4) per episode")
ax.set_xticks(x)
ax.set_xticklabels(arms, fontsize=9)
ax.set_ylim(0, 1.05)
ax.legend(fontsize=8.5)
ax.set_title("A  channel 1 — estimation (HT): variance absorption\n"
             "cuts excursion initiation (entries 0.95 $\\to$ 0.85)",
             fontsize=10.5)
ax.grid(axis="y", alpha=0.3)

ax = axes[1]
ax.bar(x - w / 2, ret20, w, color=colors, alpha=0.9,
       label="excursion returns to d<2 within 20 steps")
ax.bar(x + w / 2, esc20, w, color="dimgray",
       label="excursion escalates to d$\\geq$4 within 20 steps")
ax.set_xticks(x)
ax.set_xticklabels(arms, fontsize=9)
ax.set_ylim(0, 0.75)
ax.legend(fontsize=8.5)
ax.set_title("B  channel 2 — constraint: best excursion outcomes\n"
             "(return 0.64, escalation 0.17; toy analog: shell "
             "regularization)", fontsize=10.5)
ax.grid(axis="y", alpha=0.3)

ax = axes[2]
ax.bar(x, sr, 0.55, color=colors, alpha=0.9)
for xi, s in zip(x, sr):
    ax.text(xi, s + 1.2, f"{s:.1f}", ha="center", fontsize=9.5)
ax.set_xticks(x)
ax.set_xticklabels(arms, fontsize=9)
ax.set_ylabel("SR (%)")
ax.set_ylim(0, 95)
ax.set_title("C  result: the channels compose\n"
             "(SR|stay(d<4) = 100% for every arm)", fontsize=10.5)
ax.grid(axis="y", alpha=0.3)

fig.suptitle("HT+condreg mechanism (measured: twofactor, N=200/arm, script "
             "data): fewer excursions started $\\times$ better excursion "
             "outcomes", fontsize=12.5, y=1.02)
fig.tight_layout()
out = "analysis/paper/htcondreg_mechanism.png"
fig.savefig(out, dpi=150, bbox_inches="tight")
print("saved", out)
