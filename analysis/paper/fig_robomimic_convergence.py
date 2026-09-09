"""RoboMimic training-convergence curves: HT vs flow (same harness, Chi-UNet, absolute actions, obs history 8,
300k steps, in-train 50-episode evals every 20k). Seeds: 3 per head (transport-mh HT: 3 t12 + 4 k_align seeds).
Data: scripts_gain_rule/../rm_convergence_curves.json (per-seed curves from the cluster run logs)."""
import json, sys, numpy as np, matplotlib
matplotlib.use("Agg"); import matplotlib.pyplot as plt
D = json.load(open(sys.argv[1] if len(sys.argv) > 1 else "analysis/paper/rm_convergence_curves.json"))
steps = np.arange(20, 301, 20)
COL = {"flow": "#0072B2", "flow1": "#0072B2", "ht": "#009E73", "mip": "#8c8c8c", "mse": "#D55E00"}
LAB = {"flow": "Flow, 9 Euler steps", "flow1": "Flow, 1 step", "ht": "HT (ours), 1 pass", "mip": "MIP", "mse": "MSE, 1 pass"}
KEY = {"flow": "mean_success_9", "flow1": "mean_success_1", "ht": "mean_success_1", "mip": "mean_success_1", "mse": "mean_success_1"}
fig, axes = plt.subplots(1, 2, figsize=(7.2, 2.9), sharey=True)
summary = {}
for ax, task in zip(axes, ["transport-ph", "transport-mh"]):
    for head in ["mse", "flow", "flow1", "ht"]:
        src = "flow" if head == "flow1" else head
        if src not in D[task]: continue
        runs = dict(D[task][src])
        if head == "ht" and "ht_kalign" in D[task]: runs.update(D[task]["ht_kalign"])
        M = np.array([[v for _, v in c[KEY[head]]] for c in runs.values() if KEY[head] in c and len(c[KEY[head]]) == 15]) * 100
        m, lo, hi = M.mean(0), M.min(0), M.max(0)
        if head == "flow1":
            ax.plot(steps, m, "--", color=COL[head], lw=1.1, alpha=0.7, label=f"{LAB[head]}")
        else:
            ax.fill_between(steps, lo, hi, color=COL[head], alpha=0.15, lw=0)
            ax.plot(steps, m, "-o", color=COL[head], ms=3.5, lw=1.7, label=f"{LAB[head]} ({len(M)} seeds)")
        summary[(task, head)] = m
    ax.set_title(task.replace("transport", "Transport").replace("-ph", "-PH").replace("-mh", "-MH"), fontsize=10, loc="left")
    ax.set_xlabel("training steps (k)", fontsize=9); ax.set_xticks([20, 100, 200, 300]); ax.grid(axis="y", color="0.9", lw=0.6)
    ax.spines[["top", "right"]].set_visible(False); ax.tick_params(labelsize=8)
axes[0].set_ylabel("success rate (%)", fontsize=9); axes[0].set_ylim(0, 100); axes[0].legend(fontsize=7.5, loc="lower right", frameon=False); axes[1].legend(fontsize=7.5, loc="upper right", frameon=False)
plt.tight_layout(); plt.savefig("analysis/paper/fig_robomimic_convergence.png", dpi=200); plt.savefig("analysis/paper/fig_robomimic_convergence.pdf")
for (task, head), m in summary.items():
    first50 = next((s for s, v in zip(steps, m) if v >= 50), None); first60 = next((s for s, v in zip(steps, m) if v >= 60), None)
    print(f"{task:13s} {head:5s} mean curve: " + " ".join(f"{v:4.0f}" for v in m) + f" | first >=50%: {first50}k >=60%: {first60}k | last-5 {m[-5:].mean():.1f} | best {m.max():.1f}")
