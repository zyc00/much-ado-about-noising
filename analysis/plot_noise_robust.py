"""Held-out validation error under per-feature observation noise."""
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

SIG = [0.0, 0.05, 0.1, 0.2, 0.4]
D = {  # arm: (PR_feat far, SR, [err vs sigma], sparse5, worst1)
    "L2-200":  (19.5, 62, [.0100, .0193, .0307, .0632, .1519], .0685, .0522),
    "HT-200":  (36.4, 75, [.0083, .0173, .0293, .0570, .1331], .0624, .0393),
    "HG-200":  (32.9, 87, [.0093, .0177, .0297, .0556, .1212], .0597, .0394),
    "MIP-200": (39.5, 94, [.0074, .0179, .0305, .0569, .1242], .0597, .0518),
    "L2-2k":   (25.6, 94, [.0050, .0191, .0340, .0650, .1429], .0735, .0583),
    "MIP-2k":  (46.1, 100, [.0021, .0192, .0347, .0637, .1276], .0653, .0530),
}
C = {"L2-200": "#d62728", "HT-200": "#2ca02c", "HG-200": "#1f77b4",
     "MIP-200": "#7f7f7f", "L2-2k": "#e8927c", "MIP-2k": "#bbbbbb"}
fig, (a1, a2, a3) = plt.subplots(1, 3, figsize=(16, 4.7))

for k, (pr, sr, e, sp, w1) in D.items():
    ls = "--" if k.endswith("2k") else "-"
    a1.plot(SIG, e, ls, marker="o", ms=4, lw=2, color=C[k], label=k)
a1.set_xlabel("per-feature observation noise σ (normalized units)")
a1.set_ylabel("held-out action error vs GT (env units)")
a1.set_title("A. validation loss under per-feature noise\n"
             "(192 states from 400 unseen demos)", fontsize=10.5)
a1.legend(frameon=False, fontsize=8.5, ncol=2)
a1.spines[["top", "right"]].set_visible(False)

names = list(D)
x = np.arange(len(names))
a2.bar(x - 0.2, [D[k][3] for k in names], 0.38, color=[C[k] for k in names],
       label="sparse: 5 features @ σ=1")
a2.bar(x + 0.2, [D[k][4] for k in names], 0.38,
       color=[C[k] for k in names], alpha=0.55,
       label="worst single feature @ σ=1")
a2.set_xticks(x); a2.set_xticklabels(names, rotation=20, fontsize=8.5)
a2.set_ylabel("held-out action error vs GT")
a2.set_title("B. sparse and worst-case single-feature corruption",
             fontsize=10.5)
a2.legend(frameon=False, fontsize=8.5)
a2.spines[["top", "right"]].set_visible(False)

for k, (pr, sr, e, sp, w1) in D.items():
    m = "s" if k.endswith("2k") else "o"
    a3.scatter(pr, e[4], s=90, color=C[k], marker=m, zorder=5)
    a3.annotate(k, (pr, e[4]), textcoords="offset points", xytext=(6, 5),
                fontsize=8.5, color=C[k])
p200 = [k for k in D if k.endswith("200")]
xs = np.array([D[k][0] for k in p200]); ys = np.array([D[k][2][4] for k in p200])
o = np.argsort(xs)
a3.plot(xs[o], ys[o], ":", color="0.5", lw=1.2)
a3.set_xlabel("feature-space participation PR_feat (off-support)")
a3.set_ylabel("held-out error at σ = 0.4")
a3.set_title("C. broader feature reliance → less noise damage\n"
             "(dotted: 200-demo arms, matched data)", fontsize=10.5)
a3.spines[["top", "right"]].set_visible(False)
fig.suptitle("Per-feature observation noise: policies that read more features stay closer to ground truth",
             fontsize=12.5)
fig.tight_layout()
fig.savefig("analysis/paper/noise_robustness.png", dpi=150, bbox_inches="tight")
print("wrote analysis/paper/noise_robustness.png")
