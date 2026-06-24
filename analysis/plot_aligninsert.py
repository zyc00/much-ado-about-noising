"""align+insertion (reset to align_done - backoff(0..40)) scaling, presentation
figure. success = _check_frame_assembled, n=100 eval.
"""
import numpy as np
import matplotlib.pyplot as plt

N = np.array([200, 2000, 20000], dtype=float)
SR = {
    "MIP": np.array([50.0, 97.0, 99.0]),
    "MSE": np.array([48.0, 86.0, 97.0]),
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
ax.set_title("Align + Insertion", fontsize=17)
ax.set_ylim(40, 105)
ax.set_xticks(N); ax.set_xticklabels(["200", "2000", "20000"])
ax.legend(fontsize=14, loc="lower right")
ax.grid(True, which="both", alpha=0.3)
plt.tight_layout()
out = "analysis/aligninsert_scaling.png"
plt.savefig(out, dpi=140)
print("SAVED", out)
