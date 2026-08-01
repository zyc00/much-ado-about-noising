"""Plot Step-A recovery probe: action-prior (act_dist vs obs_dist) for MSE vs MIP.
H1 signature: MIP keeps act_dist LOW as obs_dist grows (output pulled to action
manifold); MSE's act_dist rises with obs_dist (output drifts with the state).
"""
import sys
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

pairs = [a.split("=", 1) for a in sys.argv[1:]]
BINS = [0, 0.5, 1, 1.5, 2, 3, 5, 8, 1e9]
xc = [0.25, 0.75, 1.25, 1.75, 2.5, 4, 6.5, 10]

fig, ax = plt.subplots(1, 2, figsize=(12, 4.5))
for label, path in pairs:
    d = np.load(path, allow_pickle=True)
    od, ad = d["pairs"][:, 0], d["pairs"][:, 1]
    ns, ne = int(d["n_succ"]), int(d["n_ep"])
    rr = 100 * int(d["recover"]) / max(int(d["onset"]), 1)
    means, ses, ns_bin = [], [], []
    for lo, hi in zip(BINS[:-1], BINS[1:]):
        m = (od >= lo) & (od < hi)
        means.append(ad[m].mean() if m.sum() else np.nan)
        ses.append(ad[m].std() / max(np.sqrt(m.sum()), 1) if m.sum() else np.nan)
        ns_bin.append(int(m.sum()))
    lab = f"{label} (SR {ns}/{ne}, recover {rr:.0f}%, onset {int(d['onset'])} div {int(d['diverge'])})"
    ax[0].errorbar(xc, means, yerr=ses, marker="o", ms=4, capsize=2, label=lab)
    # histogram of obs_dist (how much each method drifts off-manifold)
    ax[1].hist(np.clip(od, 0, 12), bins=40, histtype="step", lw=2, density=True, label=label)

ax[0].set_title("H1 action-prior: predicted action OOD-ness vs state OOD-ness")
ax[0].set_xlabel("obs_dist (state off-manifold, z)"); ax[0].set_ylabel("act_dist (predicted action off action-manifold, z)")
ax[0].legend(fontsize=8); ax[0].grid(alpha=.3)
ax[1].set_title("distribution of obs_dist over rollout steps")
ax[1].set_xlabel("obs_dist (z)"); ax[1].set_ylabel("density"); ax[1].legend(); ax[1].grid(alpha=.3)
plt.tight_layout()
out = "analysis/recovery/recovery_compare.png"
plt.savefig(out, dpi=120)
print("saved", out)

for label, path in pairs:
    d = np.load(path, allow_pickle=True)
    od, ad = d["pairs"][:, 0], d["pairs"][:, 1]
    hi = od >= 2
    print(f"{label}: steps={len(od)} frac(obs_dist>2)={100*hi.mean():.1f}% | "
          f"act_dist@obs<1={ad[od<1].mean():.2f} @obs>2={ad[hi].mean() if hi.sum() else float('nan'):.2f} | "
          f"recover={100*int(d['recover'])/max(int(d['onset']),1):.0f}% onset={int(d['onset'])} diverge={int(d['diverge'])}")
