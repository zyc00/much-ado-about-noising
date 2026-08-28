"""Synthetic replica of the MP-demo geometry, with the four objectives.

Data (mirrors mp_demos30): each demo has a frame offset g; the eef runs
parallel at y = g through the transport region (actions identical direction,
no transverse signal), then converges to the common insertion point y = 0 in
the transit region (the only place with transverse action signal).

obs = (x, y, g); action = (dx, dy), delta space. Clean labels (no jitter).

Models: L2 | HG (learned sigma) | L2+condreg (CV^2 of K directional FD
gains, repo formula) | anchor (progress head + weight-decayed correction).

Readouts: (1) transverse gain d a_y / d y over the (x, y) slice per model,
(2) closed-loop SR vs transverse kick at mid-transport, (3) deviation
trajectories. Success: |y| < eps at the insertion x.
"""
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn

torch.manual_seed(0)
np.random.seed(0)
dev = "cuda" if torch.cuda.is_available() else "cpu"

# ---------------- data ----------------
N_DEMO = 30
V = 0.10                 # forward step per env step
X_MERGE, X_INS = 8.3, 9.5
K_MERGE = 0.30           # transit convergence gain (per step fraction)
W = 0.30                 # tube half-width from frame offsets (30 "mm" = 0.30)
EPS_SUCC = float(os.environ.get("EPS", "0.04"))

demos = []
gs = np.linspace(-W, W, N_DEMO)
rng_d = np.random.default_rng(7)
KP = 0.20                 # PD gain (fraction of remaining vector per step)
SIG_DYN = 0.002           # dynamics wobble on y (2 "mm"); labels = exact PD
S0 = 0.85                 # start spread / max spread (measured 12.6/14.8)
X_DIV, X_CONV = 2.0, 8.3  # divergence complete; convergence begins (71%->85%)
for g in gs:
    x, y = 0.0, S0 * g + 0.0
    traj = []
    while x < X_INS:
        # waypoint schedule (counter-free proxy of script B's segments)
        if x < X_DIV:
            w = np.array([X_DIV, g])          # diverge to own line
        elif x < X_CONV:
            w = np.array([X_CONV, g])         # transport along own line
        else:
            w = np.array([X_INS, 0.0])        # converge to common staging
        rem = w - np.array([x, y])
        a = KP * rem
        a[0] = max(a[0], V)                   # keep forward progress
        traj.append([x, y, g, a[0], a[1]])    # label: exact PD response
        x = x + a[0]
        y = y + a[1] + rng_d.normal(0, SIG_DYN)   # dynamics wobble only
    demos.append(np.array(traj))
data = np.concatenate(demos)
S = torch.tensor(data[:, :3], dtype=torch.float32, device=dev)
A = torch.tensor(data[:, 3:], dtype=torch.float32, device=dev)
print(f"data: {len(S)} transitions, {N_DEMO} demos, tube half-width {W}")


def mlp(inp=3, out=2, h=64):
    return nn.Sequential(nn.Linear(inp, h), nn.SiLU(),
                         nn.Linear(h, h), nn.SiLU(), nn.Linear(h, out))


def train(name, steps=6000, lam_cond=0.0, hg=False, anchor=False):
    torch.manual_seed(1)
    if anchor:
        net_a = mlp(1, 2).to(dev)      # progress-only head
        net_c = mlp(3, 2).to(dev)      # full-state correction
        params = list(net_a.parameters()) + list(net_c.parameters())
        opt = torch.optim.AdamW([
            {"params": net_a.parameters(), "weight_decay": 1e-5},
            {"params": net_c.parameters(), "weight_decay": 3e-2}], lr=1e-3)

        def fwd(s):
            return net_a(s[:, :1]) + 0.3 * net_c(s)
    else:
        net = mlp(3, 3 if hg else 2).to(dev)
        opt = torch.optim.AdamW(net.parameters(), lr=1e-3, weight_decay=1e-5)

        def fwd(s):
            return net(s)[:, :2]
    for it in range(steps):
        idx = torch.randint(0, len(S), (256,), device=dev)
        s, a = S[idx], A[idx]
        if hg:
            out = net(s)
            pred = out[:, :2]
            sig = torch.nn.functional.softplus(out[:, 2:3]) + 1e-3
            r2 = (pred - a) ** 2
            nu = 2.0
            loss = (0.5 * (nu + 1.0) * torch.log1p(r2 / (nu * sig ** 2))
                    + torch.log(sig)).mean()
        else:
            pred = fwd(s)
            loss = ((pred - a) ** 2).mean()
        if lam_cond > 0:
            eps, K = 0.05, 6
            d2s = []
            base = fwd(s)
            for _ in range(K):
                v = torch.randn_like(s)
                v = v / (v.norm(dim=1, keepdim=True) + 1e-9)
                d2s.append(((fwd(s + eps * v) - base) ** 2).mean(dim=1))
            D = torch.stack(d2s, 1)
            cv2 = D.var(dim=1) / (D.mean(dim=1) ** 2 + 1e-12)
            loss = loss + lam_cond * torch.relu(cv2 - 1.0).mean()
        opt.zero_grad()
        loss.backward()
        opt.step()
    print(f"{name}: final fit loss {loss.item():.2e}")
    return fwd


def train_mip(steps=6000, zsig=0.12):
    torch.manual_seed(1)
    net = nn.Sequential(nn.Linear(5, 64), nn.SiLU(),
                        nn.Linear(64, 64), nn.SiLU(), nn.Linear(64, 2)).to(dev)
    opt = torch.optim.AdamW(net.parameters(), lr=1e-3, weight_decay=1e-5)
    for it in range(steps):
        idx = torch.randint(0, len(S), (256,), device=dev)
        s, a = S[idx], A[idx]
        z = torch.randn_like(a) * zsig
        y0 = net(torch.cat([z, s], 1))
        y1 = net(torch.cat([y0.detach(), s], 1))
        loss = ((y0 - a) ** 2).mean() + ((y1 - a) ** 2).mean()
        opt.zero_grad()
        loss.backward()
        opt.step()
    print(f"MIP: final fit loss {loss.item():.2e}")

    def fwd(s, zsig=zsig):
        z = torch.randn(s.shape[0], 2, device=dev) * zsig
        y0 = net(torch.cat([z, s], 1))
        return net(torch.cat([y0, s], 1))
    return fwd


GBLIND = os.environ.get("GBLIND", "0") == "1"
if GBLIND:
    S = S.clone()
    S[:, 2] = 0.0          # frame offset removed from the observation
models = {
    "L2": train("L2"),
    "HT": train("HT", hg=True),
    "L2+condreg": train("L2+condreg", lam_cond=3e-3),
    "HT+condreg": train("HT+condreg", hg=True, lam_cond=3e-3),
    "MIP": train_mip(),
    "split-ref": train("split-ref", anchor=True),
}

# ---------------- readout 1: transverse gain map ----------------
xs = np.linspace(0.2, X_INS - 0.1, 46)
ys = np.linspace(-2.5 * W, 2.5 * W, 41)
gain = {}
for name, f in models.items():
    G = np.zeros((len(ys), len(xs)))
    for i, y in enumerate(ys):
        s = torch.tensor([[x, y, 0.0] for x in xs], dtype=torch.float32,
                         device=dev)          # frame g=0 policy view
        s.requires_grad_(True)
        ay = f(s)[:, 1].sum()
        gr = torch.autograd.grad(ay, s)[0][:, 1]
        G[i] = gr.detach().cpu().numpy()
    gain[name] = G

# ---------------- readout 2: closed-loop kick battery ----------------
def rollout(f, g, kick, kick_x=3.5, noise=float(os.environ.get("RNOISE", "0.002")), nmax=220):
    x, y = 0.0, g
    kicked = False
    ys_tr = []
    rng = np.random.default_rng(3)
    sign = 1.0 if (hash((round(g * 1000), round(kick * 1000))) % 2 == 0)         else -1.0
    for _ in range(nmax):
        if not kicked and x >= kick_x:
            y += sign * kick
            kicked = True
        s = torch.tensor([[x, y, 0.0 if GBLIND else g]],
                         dtype=torch.float32, device=dev)
        with torch.no_grad():
            a = f(s)[0].cpu().numpy()
        x, y = x + a[0], y + a[1] + rng.normal(0, noise)
        ys_tr.append([x, y])
        if x >= X_INS:
            return abs(y) < EPS_SUCC, np.array(ys_tr)
    return False, np.array(ys_tr)


kicks = [0.0, 0.05, 0.10, 0.20, 0.30, 0.45]
print("\nSR vs transverse kick (20 starts g in tube):")
print("model        " + "  ".join(f"k={k:4.2f}" for k in kicks))
sr_tab = {}
for name, f in models.items():
    row = []
    for k in kicks:
        succ = [rollout(f, g, k)[0]
                for g in np.linspace(-W, W, 20)]
        row.append(np.mean(succ))
    sr_tab[name] = row
    print(f"{name:12s} " + "  ".join(f"{v:5.2f}" for v in row))

# ---------------- figure ----------------
fig, axes = plt.subplots(2, 6, figsize=(27, 8.2),
                         gridspec_kw={"height_ratios": [1.1, 1.0]})
for j, (name, f) in enumerate(models.items()):
    ax = axes[0, j]
    im = ax.imshow(gain[name], origin="lower", aspect="auto",
                   extent=[xs[0], xs[-1], ys[0] * 1000, ys[-1] * 1000],
                   cmap="RdBu_r", vmin=-0.4, vmax=0.4)
    ax.axvline(X_MERGE, color="k", ls="--", lw=1)
    ax.axhline(W * 1000, color="gray", lw=0.8)
    ax.axhline(-W * 1000, color="gray", lw=0.8)
    ax.set_title(f"{name}: transverse gain $\\partial a_y/\\partial y$",
                 fontsize=10.5)
    ax.set_xlabel("x (transport | transit after dashed line)")
    if j == 0:
        ax.set_ylabel("y offset (mm); gray = tube edge")
    plt.colorbar(im, ax=ax, fraction=0.046)

    ax = axes[1, j]
    for g in [-W, 0.0, W * 0.6]:
        for k, c in [(0.0, "tab:green"), (0.20, "tab:orange"),
                     (0.45, "tab:red")]:
            _, tr = rollout(f, g, k)
            ax.plot(tr[:, 0], tr[:, 1] * 1000, color=c, lw=1.2, alpha=0.8)
    ax.axvline(X_MERGE, color="k", ls="--", lw=1)
    ax.fill_between([0, X_INS], -W * 1000, W * 1000, color="tab:orange",
                    alpha=0.12)
    ax.axhline(0, color="k", lw=0.8)
    ax.plot([X_INS], [0], "k*", ms=14)
    ax.set_ylim(-900, 900)
    ax.set_title(f"rollouts (kick 0 / 0.20 / 0.45)  "
                 f"SR: {sr_tab[name][0]:.2f}/{sr_tab[name][3]:.2f}/"
                 f"{sr_tab[name][5]:.2f}", fontsize=10)
    ax.set_xlabel("x")
    if j == 0:
        ax.set_ylabel("y (mm)")

fig.suptitle("Synthetic MP-geometry (parallel tube from frame offsets; "
             "transverse signal only in transit): learned gain maps and "
             "closed-loop kicks", fontsize=13, y=1.00)
fig.tight_layout()
out = ("analysis/paper/toy_tube_synth_gblind.png" if GBLIND else "analysis/paper/toy_tube_synth.png")
fig.savefig(out, dpi=150, bbox_inches="tight")
print("saved", out)
print("TOY-DONE")
