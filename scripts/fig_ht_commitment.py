"""Where each loss aims at the block cell's single decision state.

No network: for each branch split p, take the demonstrations issued from the
decision state (x=32, |y|<1) and minimise each loss exactly over (mu, sigma).
The lateral aim of mu is what a perfectly optimised policy would commit to, so
the block half-height becomes a pass/fail line drawn on the same axis.

Usage: python scripts/fig_ht_commitment.py
"""
import os
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn.functional as F

sys.path.insert(0, str(Path(__file__).resolve().parent))
from toy2d_mip_nfl import ACTION_MM  # noqa: E402

C_MSE, C_HT2, C_HT1, C_HT05 = "#2b6cb0", "#c23b22", "#e08a2e", "#7a9e3a"
PS = [0.50, 0.55, 0.60, 0.65, 0.70, 0.75, 0.80, 0.90]
BLOCK_HALF, DETOUR = 3.0, 6.0


def decision_samples(p):
    os.environ["OB_PMAJ"] = f"{p}"
    for m in ("toy2d_obstacle",):                 # re-import so P_MAJ is re-read
        sys.modules.pop(m, None)
    from toy2d_obstacle import make_ds_episodes
    st, _, ac = make_ds_episodes(4000, seed=1000)
    S, A_ = st.numpy(), ac.numpy()
    x = (S[:, 0] + 1.0) * 80.0
    y = S[:, 1] * 14.0
    m = (np.abs(x - 32) < 1) & (np.abs(y) < 1.0)
    return torch.tensor(A_[m])


def fit(samples, nu, sigma_fixed=None, steps=5000):
    A = samples.shape[1]
    mu = samples.mean(0).clone().requires_grad_(True)
    ls = torch.tensor([-3.0], requires_grad=True)
    opt = torch.optim.Adam([mu] if sigma_fixed else [mu, ls], lr=0.01)
    for _ in range(steps):
        sigma = torch.tensor(float(sigma_fixed)) if sigma_fixed else F.softplus(ls) + 1e-3
        r2 = ((samples - mu) ** 2).sum(1)
        loss = (0.5 * (nu + 1) * torch.log1p(r2 / (nu * sigma ** 2 * A)) * A + A * sigma.log()).mean()
        opt.zero_grad(); loss.backward(); opt.step()
    return float(mu.detach()[1::2].sum()) * ACTION_MM


curves = {k: [] for k in ("mean", 2.0, 1.0, 0.5)}
for p in PS:
    s = decision_samples(p)
    curves["mean"].append(float(s.mean(0)[1::2].sum()) * ACTION_MM)
    for nu in (2.0, 1.0, 0.5):
        curves[nu].append(fit(s, nu))
    print(f"p={p:.2f}  mean {curves['mean'][-1]:+.2f}  nu2 {curves[2.0][-1]:+.2f}  "
          f"nu1 {curves[1.0][-1]:+.2f}  nu05 {curves[0.5][-1]:+.2f}", flush=True)

sig_grid = [0.30, 0.20, 0.10, 0.06, 0.04, 0.02]
s60 = decision_samples(0.60)
sig_curve = [fit(s60, 2.0, sigma_fixed=sg) for sg in sig_grid]

fig, axes = plt.subplots(1, 2, figsize=(13.0, 5.0))
ax = axes[0]
ax.axhspan(0, BLOCK_HALF, color="0.85", zorder=0)
ax.axhline(BLOCK_HALF, color="0.35", lw=1.4, ls="--")
ax.text(0.505, BLOCK_HALF - 0.35, "aims inside this band hit the block", fontsize=9, color="0.25")
ax.axhline(DETOUR, color="0.6", lw=1.0, ls=":")
ax.text(0.505, DETOUR + 0.12, "demonstrated detour", fontsize=9, color="0.45")
for key, lab, c in [("mean", "MSE (conditional mean)", C_MSE), (2.0, "HT nu=2", C_HT2),
                    (1.0, "HT nu=1", C_HT1), (0.5, "HT nu=0.5", C_HT05)]:
    ax.plot(PS, curves[key], "o-", color=c, lw=2.0, ms=5, label=lab)
ax.set_xlabel("majority branch probability", fontsize=10.5)
ax.set_ylabel("lateral aim committed at the decision state (mm)", fontsize=10.5)
ax.set_title("Commitment is graded in nu — and the block sets the bar",
             fontsize=11.5, fontweight="bold", loc="left")
ax.legend(fontsize=9.5, loc="upper left"); ax.grid(alpha=0.25); ax.set_ylim(0, 7)

ax = axes[1]
ax.axhspan(0, BLOCK_HALF, color="0.85", zorder=0)
ax.axhline(BLOCK_HALF, color="0.35", lw=1.4, ls="--")
ax.plot(sig_grid, sig_curve, "o-", color=C_HT2, lw=2.0, ms=6)
ax.axvline(0.081, color="0.4", ls=":", lw=1.4)
ax.text(0.084, 5.4, "sigma the loss selects\nhere (0.081)", fontsize=9, color="0.3")
ax.set_xscale("log"); ax.invert_xaxis()
ax.set_xlabel("sigma (pinned, log scale — smaller to the right)", fontsize=10.5)
ax.set_ylabel("HT nu=2 lateral aim (mm)", fontsize=10.5)
ax.set_title("Why nu=2 falls short at p=0.6: commitment scales with 1/sigma",
             fontsize=11.5, fontweight="bold", loc="left")
ax.grid(alpha=0.25); ax.set_ylim(0, 7)

fig.suptitle("Block cell: what each loss aims for at the single decision state (x=32), fitted on the real data",
             fontsize=12.5, fontweight="bold")
fig.tight_layout(rect=[0, 0, 1, 0.93])
fig.savefig("analysis/paper/ht_commitment.png", dpi=180)
print("WROTE analysis/paper/ht_commitment.png")
