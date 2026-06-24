"""Task-segment ablation summary: insertion-only / align+insertion / init->insertion,
each MIP vs MSE, clean ToolHang, ChiUNet, success = frame assembled (insertion).
- insertion-only & align+insertion: reset+rerun data, set_state near-hole eval (valid).
- init->insertion: full demos, official faithful eval (mode=eval), model_best, n=100.
"""
import numpy as np
import matplotlib.pyplot as plt

N = np.array([200, 2000, 20000], dtype=float)
SEG = [
    ("Insertion only",     {"MIP": [93, 98, 100], "MSE": [88, 96, 99]}),
    ("Align + Insertion",  {"MIP": [50, 97, 99],  "MSE": [48, 86, 97]}),
    ("Init → Insertion", {"MIP": [26, 41, 81], "MSE": [9, 17, 59]}),
]
COL = {"MIP": "#d62728", "MSE": "#1f77b4"}

plt.rcParams.update({"font.size": 13})
fig, axes = plt.subplots(1, 3, figsize=(16, 5.2), sharey=True)
for ax, (title, d) in zip(axes, SEG):
    for k in ["MIP", "MSE"]:
        y = np.array(d[k])
        ax.plot(N, y, "o-", color=COL[k], ms=10, lw=2.5, label=k)
        for x, yy in zip(N, y):
            ax.annotate(f"{yy:.0f}", (x, yy), textcoords="offset points",
                        xytext=(0, 9), ha="center", fontsize=11, color=COL[k])
    ax.set_xscale("log"); ax.set_xticks(N); ax.set_xticklabels(["200", "2k", "20k"])
    ax.set_title(title, fontsize=16)
    ax.set_xlabel("# demonstrations", fontsize=14)
    ax.set_ylim(0, 104); ax.grid(True, which="both", alpha=0.3)
    ax.legend(fontsize=13, loc="lower right")
axes[0].set_ylabel("insertion success rate (%)", fontsize=14)
fig.suptitle("Task-segment ablation: MIP vs MSE (clean ToolHang)", fontsize=17, y=1.02)
plt.tight_layout()
out = "analysis/segment_ablation.png"
plt.savefig(out, dpi=140, bbox_inches="tight")
print("SAVED", out)
