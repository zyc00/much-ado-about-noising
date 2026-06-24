"""init->insertion (full demos, official faithful eval, model_best, n=100).
success = frame assembled (insertion). MIP vs MSE.
hang-criterion sanity (matches published scaling): MIP 13/21/54, MSE 1/4/13.
"""
import numpy as np
import matplotlib.pyplot as plt

N = np.array([200, 2000, 20000], dtype=float)
SR = {
    "MIP": np.array([51, 95, 99]),
    "MSE": np.array([20, 80, 81]),
}
COL = {"MIP": "#d62728", "MSE": "#1f77b4"}

plt.rcParams.update({"font.size": 14})
fig, ax = plt.subplots(figsize=(7.5, 6))
for k in ["MIP", "MSE"]:
    ax.plot(N, SR[k], "o-", color=COL[k], ms=11, lw=2.5, label=k)
    for x, y in zip(N, SR[k]):
        ax.annotate(f"{y:.0f}", (x, y), textcoords="offset points",
                    xytext=(0, 11), ha="center", fontsize=13, color=COL[k])
ax.set_xscale("log")
ax.set_xlabel("# demonstrations", fontsize=15)
ax.set_ylabel("insertion success rate (%)", fontsize=15)
ax.set_title("Init → Insertion (full task)", fontsize=17)
ax.set_ylim(0, 90)
ax.set_xticks(N); ax.set_xticklabels(["200", "2000", "20000"])
ax.legend(fontsize=14, loc="upper left")
ax.grid(True, which="both", alpha=0.3)
plt.tight_layout()
out = "analysis/init2ins_scaling.png"
plt.savefig(out, dpi=140)
print("SAVED", out)
