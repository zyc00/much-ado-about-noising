"""Plot manifold-deviation curves robustly (median + off-manifold fraction),
since catastrophic divergences (dist in the 100s-1000s) wreck the mean.
Reads the npz files saved by eval_manifold_deviation.py.
"""
import sys
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

# args: pairs of label=path
pairs = [a.split("=", 1) for a in sys.argv[1:]]
OFF = 5.0  # z-scored distance above this = off the expert manifold

fig, ax = plt.subplots(1, 3, figsize=(15, 4))
for label, path in pairs:
    d = np.load(path, allow_pickle=True)
    allc = d["all"]            # (n_ep, bins)
    B = allc.shape[1]
    x = np.linspace(0, 1, B)
    med = np.nanmedian(allc, 0)
    off = np.nanmean(allc > OFF, 0) * 100
    succ = d["succ"]
    smean = np.nanmean(succ, 0) if len(succ) else np.full(B, np.nan)
    n = int(d["n"]); ns = int(d["n_succ"])
    lab = f"{label} ({ns}/{n})"
    ax[0].plot(x, med, marker="o", ms=3, label=lab)
    ax[1].plot(x, off, marker="o", ms=3, label=lab)
    ax[2].plot(x, smean, marker="o", ms=3, label=lab)

ax[0].set_title("MEDIAN nearest-expert dist (robust)")
ax[0].set_xlabel("rollout progress"); ax[0].set_ylabel("z-scored dist"); ax[0].legend(); ax[0].grid(alpha=.3)
ax[1].set_title(f"% episodes OFF-manifold (dist>{OFF})")
ax[1].set_xlabel("rollout progress"); ax[1].set_ylabel("% episodes"); ax[1].legend(); ax[1].grid(alpha=.3)
ax[2].set_title("SUCCESS-only mean dist (drift→recover)")
ax[2].set_xlabel("rollout progress"); ax[2].set_ylabel("z-scored dist"); ax[2].legend(); ax[2].grid(alpha=.3)
plt.tight_layout()
out = "analysis/manifold/manifold_compare.png"
plt.savefig(out, dpi=120)
print("saved", out)

# text summary
for label, path in pairs:
    d = np.load(path, allow_pickle=True)
    allc = d["all"]; B = allc.shape[1]
    med = np.nanmedian(allc, 0)
    off_end = np.nanmean(allc[:, -1] > OFF) * 100
    off_peak = np.nanmax(np.nanmean(allc > OFF, 0)) * 100
    succ = d["succ"]; smean = np.nanmean(succ, 0) if len(succ) else np.array([np.nan])
    print(f"{label}: median end={med[-1]:.2f} peak={np.nanmax(med):.2f} | "
          f"off-manifold end={off_end:.0f}% peak={off_peak:.0f}% | "
          f"succ-mean end={np.nanmin(smean[B//2:]) if len(succ) else float('nan'):.2f}")
