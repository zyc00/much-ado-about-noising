"""Erosion toy: does fresh-noise supervision (a measure) prevent the
memorize->erode dynamics that kill fine structure under MSE?

Task: x ~ U[0,1]^2 (N=2000 fixed points). y in R^4:
  dims 0-1: big trend + 0.3*t(2) irreducible label noise (everywhere)
  dims 2-3: fine checkerboard grating amp 0.08, only in patch x1>0.7
Arms: l2 | flowF (fresh eps,t per step) | flowZ (eps,t frozen per sample)
      | mip2 (2-step fresh)
Readouts every 2k steps: fine-region clean MSE, Jacobian PR + x2-gain at
patch points, cos(g_patch, g_total). Prints ERODE lines; saves heatmap
snapshots of predicted dim-2 for the figure.
"""
import os
import sys

import numpy as np
import torch
import torch.nn as nn

DEV = "cuda" if torch.cuda.is_available() else "cpu"
STEPS = int(os.environ.get("STEPS", "100000"))
LOG_EVERY = 2000
N = int(os.environ.get("NPTS", "4000"))
W = int(os.environ.get("W", "384"))
FREQ = float(os.environ.get("FREQ", "10"))
AMP = float(os.environ.get("AMP", "0.15"))
NAMP = float(os.environ.get("NAMP", "0.3"))
NS = 16  # flow integration steps at readout

rng = np.random.RandomState(0)
X = rng.rand(N, 2)


def clean_fine(x):
    b = (x[:, 0] > 0.7).astype(np.float64)
    g1 = AMP * np.sin(FREQ * np.pi * x[:, 0]) * np.sin(FREQ * np.pi * x[:, 1]) * b
    g2 = AMP * (np.cos(FREQ * np.pi * x[:, 0]) - np.sin(FREQ * np.pi * x[:, 1])) * 0.5 * b
    return np.stack([g1, g2], 1)


def make_labels(x, rng):
    tr1 = np.sin(2 * np.pi * x[:, 0]) + x[:, 1]
    tr2 = np.cos(2 * np.pi * x[:, 1]) - x[:, 0]
    noise = NAMP * rng.standard_t(2, size=(len(x), 2))
    y01 = np.stack([tr1, tr2], 1) + noise
    return np.concatenate([y01, clean_fine(x)], 1)


Y = make_labels(X, rng)
Xt = torch.tensor(X, device=DEV, dtype=torch.float32)
Yt = torch.tensor(Y, device=DEV, dtype=torch.float32)
FINE = torch.tensor(clean_fine(X), device=DEV, dtype=torch.float32)
_tr1 = np.sin(2 * np.pi * X[:, 0]) + X[:, 1]
_tr2 = np.cos(2 * np.pi * X[:, 1]) - X[:, 0]
Yt_clean01 = torch.tensor(np.stack([_tr1, _tr2], 1), device=DEV,
                          dtype=torch.float32)
B_mask = Xt[:, 0] > 0.7

# eval grid for heatmaps
gg = np.linspace(0, 1, 120)
GX, GY = np.meshgrid(gg, gg)
GRID = torch.tensor(np.stack([GX.ravel(), GY.ravel()], 1), device=DEV,
                    dtype=torch.float32)


class Reg(nn.Module):
    def __init__(self):
        super().__init__()
        self.f = nn.Sequential(nn.Linear(2, W), nn.ReLU(),
                               nn.Linear(W, W), nn.ReLU(),
                               nn.Linear(W, W), nn.ReLU(), nn.Linear(W, 4))

    def forward(self, x):
        return self.f(x)


class Flow(nn.Module):
    def __init__(self):
        super().__init__()
        self.f = nn.Sequential(nn.Linear(2 + 4 + 1, W), nn.ReLU(),
                               nn.Linear(W, W), nn.ReLU(),
                               nn.Linear(W, W), nn.ReLU(), nn.Linear(W, 4))

    def vel(self, x, yt, t):
        return self.f(torch.cat([x, yt, t], 1))

    def readout(self, x, nsteps=NS, ndraw=8, gen=None):
        outs = []
        for _ in range(ndraw):
            y = torch.randn(len(x), 4, device=x.device, generator=gen)
            for j in range(nsteps):
                t = torch.full((len(x), 1), j / nsteps, device=x.device)
                y = y + (1.0 / nsteps) * self.vel(x, y, t)
            outs.append(y)
        return torch.stack(outs).mean(0)


def metrics(readout_fn, arm, step, gtot):
    with torch.no_grad():
        pred = readout_fn(Xt)
        fmse = float(((pred[B_mask, 2:] - FINE[B_mask]) ** 2).mean())
    # jacobian at 24 patch points
    idx = torch.where(B_mask)[0][:24]
    prs, g2s = [], []
    for i in idx:
        x = Xt[i:i + 1].clone().requires_grad_(True)
        J = torch.autograd.functional.jacobian(
            lambda z: readout_fn(z).reshape(-1), x, vectorize=True)
        J = J.reshape(4, 2)
        s2 = torch.linalg.svdvals(J) ** 2
        prs.append(float(s2.sum() ** 2 / (s2 ** 2).sum()))
        g2s.append(float(J[2:, 1].abs().mean()))
    print(f"ERODE {arm} {step} fmse {fmse:.2e} pr {np.mean(prs):.3f} "
          f"g2 {np.mean(g2s):.4f} cos {gtot:.3f}", flush=True)


def grad_cos(loss_fn_all, loss_fn_patch, net):
    g = []
    for lf in (loss_fn_patch, loss_fn_all):
        for p in net.parameters():
            p.grad = None
        lf().backward()
        g.append(torch.cat([p.grad.reshape(-1) for p in net.parameters()]))
    return float(torch.dot(g[0], g[1]) / (g[0].norm() * g[1].norm() + 1e-12))


def train(arm):
    torch.manual_seed(0)
    net = (Reg() if arm in ("l2", "l2clean") else Flow()).to(DEV)
    opt = torch.optim.Adam(net.parameters(), lr=1e-3)
    brng = np.random.RandomState(1)
    tgen = torch.Generator(device=DEV)
    tgen.manual_seed(42)
    if arm == "flowZ":  # frozen supervision set: one (eps, t) per sample
        EPSZ = torch.randn(N, 4, device=DEV, generator=tgen)
        TZ = torch.rand(N, 1, device=DEV, generator=tgen)
    nsteps_arm = 2 if arm == "mip2" else NS
    snaps = {}
    for it in range(STEPS + 1):
        idx = torch.tensor(brng.randint(0, N, 256), device=DEV)
        xb, yb = Xt[idx], Yt[idx]
        if arm in ("l2", "l2clean"):
            yb2 = yb if arm == "l2" else torch.cat(
                [Yt_clean01[idx], yb[:, 2:]], 1)
            loss = ((net(xb) - yb2) ** 2).mean()
        else:
            if arm == "flowZ":
                eps, t = EPSZ[idx], TZ[idx]
            else:
                eps = torch.randn(len(idx), 4, device=DEV, generator=tgen)
                t = torch.rand(len(idx), 1, device=DEV, generator=tgen)
            ytt = (1 - t) * eps + t * yb
            loss = ((net.vel(xb, ytt, t) - (yb - eps)) ** 2).mean()
        opt.zero_grad()
        loss.backward()
        opt.step()
        if it % LOG_EVERY == 0:
            if arm in ("l2", "l2clean"):
                ro = net
                lf_all = lambda: ((net(Xt) - Yt) ** 2).mean()
                lf_patch = lambda: ((net(Xt[B_mask])[:, 2:] - Yt[B_mask][:, 2:]) ** 2).mean()
            else:
                gen2 = torch.Generator(device=DEV)
                gen2.manual_seed(7)
                ro = lambda x: net.readout(x, nsteps=nsteps_arm, gen=gen2)
                e0 = torch.randn(N, 4, device=DEV, generator=tgen)
                t0 = torch.rand(N, 1, device=DEV, generator=tgen)
                yt0 = (1 - t0) * e0 + t0 * Yt
                lf_all = lambda: ((net.vel(Xt, yt0, t0) - (Yt - e0)) ** 2).mean()
                bm = B_mask
                lf_patch = lambda: ((net.vel(Xt[bm], yt0[bm], t0[bm])
                                     - (Yt[bm] - e0[bm]))[:, 2:] ** 2).mean()
            c = grad_cos(lf_all, lf_patch, net)
            metrics(ro, arm, it, c)
            if it in (4000, 20000, 60000, 100000):
                with torch.no_grad():
                    snaps[it] = ro(GRID)[:, 2].reshape(120, 120).cpu().numpy()
    np.savez(f"erode_snaps_{arm}.npz", **{str(k): v for k, v in snaps.items()})
    return net


for arm in (sys.argv[1].split(",") if len(sys.argv) > 1
            else ["l2", "flowF", "flowZ", "mip2"]):
    print(f"=== ARM {arm}", flush=True)
    train(arm)
print("ERODE-DONE", flush=True)
