"""Three-peak endpoint-sweep figure for the branch+noise NFL design."""
import json
import sys

import matplotlib.pyplot as plt
import numpy as np

d = json.load(open(sys.argv[1] if len(sys.argv) > 1 else "analysis/toy2d_branchnoise_p08_wide.json"))
docks = np.array(d["config"]["docks"])
ref = dict(d["reference_mm"])
# recompute the mean from the actual mixture (the JSON writer hardcoded 60/40)
pm = d["config"]["p_maj"]; A = d["config"]["A"]
ref["mean"] = (pm * (-A) + (1 - pm) * A) * 20.0

fig, ax = plt.subplots(figsize=(11, 5.5))
styles = {
    "l2": ("L2 (mean)", "#2b6cb0", "-"),
    "hg": ("HG", "#63b3ed", "--"),
    "mip_step1": ("MIP step-1 (control)", "#b794f4", ":"),
    "mip_full": ("MIP full", "#7c3aed", "-"),
    "ht2": ("HT nu=2", "#c23b22", "-"),
    "ht05": ("HT nu=0.5", "#f6ad55", "--"),
}
for k, (lab, c, ls) in styles.items():
    a = d["arms"][k]
    m = np.array(a["sr_curve_mean"]); s = np.array(a["sr_curve_std"])
    ax.plot(docks, m, ls, color=c, lw=2.2, label=lab)
    ax.fill_between(docks, m - s, m + s, color=c, alpha=0.10)

for x, lab in [(ref["mean"], "mixture mean"), (ref["majority_mode"], "majority branch")]:
    ax.axvline(x, color="0.55", lw=0.9, ls="--")
    ax.text(x, 1.045, lab, ha="center", fontsize=10, color="0.35")

ax.set_xlabel("graded endpoint position d (mm)", fontsize=12)
ax.set_ylabel("success rate  (|landing - d| < 0.5 mm)", fontsize=12)
ax.set_title(
    "One dataset, one training per method — the endpoint position alone decides the winner\n"
    "(branch aims 0.8/0.2 at ∓6/+6 mm + symmetric smear; 8 seeds, shading = seed std)",
    fontsize=12.5,
)
ax.set_xlim(-8, 2)
ax.set_ylim(0, 1.1)
ax.legend(fontsize=10.5, loc="upper right")
fig.tight_layout()
fig.savefig("analysis/paper/toy_branchnoise_results.png", dpi=160)
fig.savefig("analysis/paper/toy_branchnoise_results.pdf")
print("saved analysis/paper/toy_branchnoise_results.png")
