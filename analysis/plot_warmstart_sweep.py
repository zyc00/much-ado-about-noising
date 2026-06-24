"""Warm-start depth sweep (faithful: replay scripted to depth K, then policy).
chi MIP vs MSE @2000, curated held-out seeds. Localizes WHERE the MIP>MSE
advantage is created along init->insertion. Gap closes by grasp (c1):
MIP's edge is the init->grasp (free-space reach + grasp) phase.
"""
import numpy as np
import matplotlib.pyplot as plt

labels = ["init\n(0)", "mid\nreach", "grasp\n(c1)", "mid\nalign", "aligned"]
x = np.arange(len(labels))
MIP = np.array([84, 98, 100, 100, 100])
MSE = np.array([22, 59, 95, 97, 100])

fig, (ax, ax2) = plt.subplots(1, 2, figsize=(13, 5.2))

ax.plot(x, MIP, "o-", color="#d62728", ms=11, lw=2.5, label="MIP")
ax.plot(x, MSE, "o-", color="#1f77b4", ms=11, lw=2.5, label="MSE")
ax.fill_between(x, MSE, MIP, color="gray", alpha=0.15)
for xi, a, b in zip(x, MIP, MSE):
    ax.annotate(f"{a}", (xi, a), textcoords="offset points", xytext=(0, 8), ha="center", fontsize=10, color="#d62728")
    ax.annotate(f"{b}", (xi, b), textcoords="offset points", xytext=(0, -14), ha="center", fontsize=10, color="#1f77b4")
ax.set_xticks(x); ax.set_xticklabels(labels)
ax.set_ylabel("insertion success rate (%)", fontsize=13)
ax.set_xlabel("warm-start depth (expert runs up to here, then policy)", fontsize=12)
ax.set_title("Policy takeover from depth K  (chi 2000)", fontsize=14)
ax.set_ylim(0, 105); ax.legend(fontsize=13, loc="lower right"); ax.grid(alpha=0.3)

gap = MIP - MSE
ax2.bar(x, gap, color="#9467bd", alpha=0.8)
for xi, gg in zip(x, gap):
    ax2.annotate(f"{gg}", (xi, gg), textcoords="offset points", xytext=(0, 4), ha="center", fontsize=11)
ax2.set_xticks(x); ax2.set_xticklabels(labels)
ax2.set_ylabel("MIP − MSE  (pts)", fontsize=13)
ax2.set_xlabel("warm-start depth", fontsize=12)
ax2.set_title("MIP advantage is created in init→grasp", fontsize=14)
ax2.grid(alpha=0.3, axis="y")

fig.suptitle("Where does MIP beat MSE? (warm-start sweep, faithful eval)", fontsize=15, y=1.02)
plt.tight_layout()
out = "analysis/warmstart_sweep.png"
plt.savefig(out, dpi=140, bbox_inches="tight")
print("SAVED", out)
