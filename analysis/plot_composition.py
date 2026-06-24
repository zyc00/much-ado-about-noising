"""Composition (clean ToolHang, 2000 demos, CORRECTED settle-fixed eval, init->insertion).
success = frame assembled. Numbers via env.reset+10-zero settle (matching training init).
"""
import matplotlib.pyplot as plt
labels = ["MSE\n1 net", "MSE 2-stage\nstitch\nMISMATCH", "MSE 2-stage\nstitch\nMATCHED", "MIP\n1 net"]
vals = [80, 42, 97, 95]
cols = ["#1f77b4", "#5b9bd5", "#2ca02c", "#d62728"]
plt.rcParams.update({"font.size": 13})
fig, ax = plt.subplots(figsize=(9, 5.6))
ax.bar(range(len(vals)), vals, color=cols, width=0.62)
for i, v in enumerate(vals):
    ax.annotate(f"{v}", (i, v), textcoords="offset points", xytext=(0, 6), ha="center", fontsize=14, weight="bold")
ax.set_xticks(range(len(labels))); ax.set_xticklabels(labels)
ax.set_ylabel("insertion success rate (%)", fontsize=14); ax.set_ylim(0, 100)
ax.set_title("init->insertion @2000 (settle-fixed eval): MSE single-net works (80%)", fontsize=13)
ax.grid(True, axis="y", alpha=0.3)
ax.text(0.5, 0.94, "sub-tasks: grasp 100%  |  insert-from-clean-grasp 100%",
        transform=ax.transAxes, ha="center", fontsize=11, style="italic", color="gray")
ax.text(0.5, 0.86, "mismatched stitching (42) is WORSE than one net (80)",
        transform=ax.transAxes, ha="center", fontsize=10, color="#c0504d")
plt.tight_layout(); plt.savefig("analysis/composition_story.png", dpi=140); print("SAVED")
