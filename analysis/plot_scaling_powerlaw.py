"""Power-law scaling fit + extrapolation for clean-data full-task SR (top-5 mean).
log(1-SR) = a*log(N) + b.  Adds flow (mean_success_9) alongside MIP/MSE/Cauchy.
"""
import numpy as np
import matplotlib.pyplot as plt

N = np.array([200, 2000, 20000], dtype=float)
# top-5-eval means (%), verified from logs/chi_* metrics.jsonl
SR = {
    "MIP":    np.array([11.6, 23.2, 51.6]),   # mean_success_1 (2-step sampler)
    "Flow":   np.array([22.8, 49.2, 50.8]),   # mean_success_9 (9-step ODE)
    "MSE":    np.array([0.8,  6.4,  12.0]),
    "Cauchy": np.array([7.6,  11.2, 8.0]),
}
COL = {"MIP": "tab:red", "Flow": "tab:purple", "MSE": "tab:blue", "Cauchy": "tab:green"}

logN = np.log(N)


def fit(sr):
    fail = 1 - sr / 100.0
    y = np.log(fail)
    a, b = np.polyfit(logN, y, 1)
    return a, b


fig, (axL, axR) = plt.subplots(1, 2, figsize=(14, 5.5))

# ---- Left: power-law fit on failure rate (log-log) ----
xs = np.logspace(np.log10(150), np.log10(4e5), 100)
for k in ["MIP", "Flow", "MSE", "Cauchy"]:
    a, b = fit(SR[k])
    fail = 1 - SR[k] / 100.0
    axL.plot(N, fail, "o-", color=COL[k], label=f"{k} (data)")
    axL.plot(xs, np.exp(b) * xs ** a, "--", color=COL[k], alpha=0.45,
             label=f"{k} fit a={a:.3f}")
axL.set_xscale("log"); axL.set_yscale("log")
axL.set_xlabel("# demos N"); axL.set_ylabel("failure rate 1-SR (log)")
axL.set_title("Power-law fit: log(1-SR)=a·log(N)+b")
axL.legend(fontsize=8, ncol=1)
axL.grid(True, which="both", alpha=0.3)

# ---- Right: extrapolation to human-level for the two two-call methods ----
xe = np.logspace(np.log10(100), 7, 200)
axR.axhline(84, color="gray", ls="-.", label="human-data BC (84%)")
for k in ["MIP", "Flow"]:
    a, b = fit(SR[k])
    sr_curve = (1 - np.exp(b) * xe ** a) * 100
    axR.plot(N, SR[k], "o", color=COL[k], ms=9)
    axR.plot(xe, sr_curve, "--", color=COL[k], label=f"{k} global fit (a={a:.3f})")
axR.set_xscale("log")
axR.set_xlabel("# demos N"); axR.set_ylabel("SR %")
axR.set_title("Extrapolation to human-level")
axR.set_ylim(-10, 95)
axR.legend(fontsize=9, loc="lower right")
axR.grid(True, which="both", alpha=0.3)

plt.tight_layout()
out = "analysis/scaling_powerlaw_fit.png"
plt.savefig(out, dpi=120)
print("SAVED", out)
for k in SR:
    a, b = fit(SR[k])
    print(f"  {k:7s} a={a:+.3f} b={b:+.3f}  data={SR[k].tolist()}")
