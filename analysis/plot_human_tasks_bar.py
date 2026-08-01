"""Bar chart: best SR per arm across the four human-data tasks."""
import matplotlib.pyplot as plt
import numpy as np

TASKS = ["TH", "TP-ph", "TP-mh", "SQ-mh"]
ARMS = ["L2", "HT", "HG", "MIP"]
C = {"L2": "#d62728", "HT": "#2ca02c", "HG": "#1f77b4", "MIP": "#7f7f7f"}
BEST = {
    "L2":  [43, 47, 12, 70],
    "HT":  [86, 75, 53, 80],
    "HG":  [73, 70, 52.5, 74],
    "MIP": [80, 75, 60, 88],
}

fig, ax = plt.subplots(figsize=(8.6, 4.4))
x = np.arange(len(TASKS))
w = 0.19
for i, arm in enumerate(ARMS):
    vals = np.array(BEST[arm], dtype=float)
    pos = x + (i - 1.5) * w
    ax.bar(pos, vals, width=w - 0.02, color=C[arm], label=arm)
    for p, v in zip(pos, vals):
        ax.text(p, v + 1.2, f"{v:g}", ha="center", fontsize=8.5, color="0.2")


ax.set_xticks(x)
ax.set_xticklabels(TASKS, fontsize=11)
ax.set_ylabel("best success rate (%)", fontsize=10)
ax.set_ylim(0, 100)
ax.spines[["top", "right"]].set_visible(False)
ax.legend(frameon=False, ncol=4, loc="upper center", bbox_to_anchor=(0.5, 1.14))
ax.set_title("Human-data tasks: best SR by loss family", fontsize=12, pad=28)
fig.tight_layout()
fig.savefig("analysis/paper/human_tasks_bar.png", dpi=200, bbox_inches="tight")
fig.savefig("analysis/paper/human_tasks_bar.pdf", bbox_inches="tight")
print("saved analysis/paper/human_tasks_bar.png")
