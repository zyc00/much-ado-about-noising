"""A-vs-B toy (pure learning/optimization, no dynamics): does flow/denoising
help via (A) implicit Jacobian/smoothness regularization or (B) a structural
full-rank requirement of the objective?

Task: x = g(tau) on a 1-D curve in R^20; y = h(tau) in R^10.
  x1 = height-like, NON-MONOTONE (descend 1->0.1, hold ~0.1 on the quiet band
  tau in [.4,.6], rise 0.1->0.5 through the SAME height band = aliasing);
  x2..x20 = small-amplitude smooth functions (jointly identify tau).
  y: quiet band ~0; approach pattern A; stroke pattern B (large); tail C.

Arms: l2 | l2eps (noise INPUT, no target role) | l2jit (input jitter on x) |
      flow (fresh eps,t) | flowgrid (frozen (eps,t) pairs per sample) |
      l2aux (aux target Ax: engagement without denoising).

Instruments (function space only):
  PR(band): participation ratio of d f/d x singular values, per band.
  biasOff(r): ||f(g(tau)+r u) - h(tau)|| median over random normal u.
  misread: at quiet-band tau with off-support probes, cosine of output to the
  ALIASED stroke target h(tau_alias(x1)) and output norm ratio.
  distY(r): distance of output to the y-manifold {h(tau)} (action validity).
  field check (flow arms): eigenvalues of d v/d y_t * (1-t) (should be ~ -1).
"""
import json
import os

import numpy as np
import torch
import torch.nn as nn

DEV = "cuda" if torch.cuda.is_available() else "cpu"
NX, NY = 20, 10
STEPS = int(os.environ.get("STEPS", "20000"))
WIDTH = int(os.environ.get("WIDTH", "256"))
N = int(os.environ.get("N", "2000"))
SEEDS = [int(s) for s in os.environ.get("SEEDS", "0,1,2").split(",")]
ARMS = os.environ.get("ARMS", "l2,l2eps,l2jit,flow,flowgrid,l2aux").split(",")
JIT = float(os.environ.get("JIT", "0.05"))
NS_FLOW = 32

rng_world = np.random.RandomState(7)
PH = rng_world.uniform(0, 2 * np.pi, (NX, 3))
AMP_MINOR = 0.1
W_AUX = rng_world.randn(NX, NX) / np.sqrt(NX)
PH_Y = rng_world.uniform(0, 2 * np.pi, (NY, 2))


def height(tau):
    t = np.asarray(tau)
    h = np.where(t < 0.4, 1.0 - 1.75 * t,            # descend 1 -> 0.3
        np.where(t < 0.6, 0.3,                       # quiet hold at MID-STROKE height
        np.where(t < 0.65, 0.3 - 4.0 * (t - 0.6),    # dip 0.3 -> 0.1
        np.where(t < 0.85, 0.1 + 2.0 * (t - 0.65),   # rise 0.1 -> 0.5 (0.3 @ t=0.75)
                 0.5))))
    return h


def g(tau):
    t = np.atleast_1d(tau)
    x = np.zeros((len(t), NX))
    x[:, 0] = height(t)
    for k in range(1, NX):
        x[:, k] = AMP_MINOR * (np.sin(2 * np.pi * (k % 4 + 1) * t + PH[k, 0])
                               + 0.5 * np.sin(2 * np.pi * (k % 7 + 2) * t + PH[k, 1]))
    return x


def h(tau):
    t = np.atleast_1d(tau)
    y = np.zeros((len(t), NY))
    quiet = (t >= 0.4) & (t <= 0.6)
    appr = t < 0.4
    stroke = (t > 0.6) & (t <= 0.85)
    tail = t > 0.85
    for j in range(NY):
        pa = 0.4 * np.sin(2 * np.pi * (j % 3 + 1) * t + PH_Y[j, 0])
        pb = 1.5 * np.sin(2 * np.pi * (j % 4 + 1) * (t - 0.6) / 0.25
                          + PH_Y[j, 1] + 0.7)
        pc = 0.3 * np.cos(2 * np.pi * (j % 2 + 1) * t)
        y[:, j] = np.where(appr, pa, 0.0) + np.where(stroke, pb, 0.0) \
            + np.where(tail, pc, 0.0)
    y[quiet] *= 0.0
    return y


def tau_alias(x1):
    """the RISING stroke-branch tau with the same height (x1 in [0.1, 0.5]);
    quiet hold (0.3) aliases mid-stroke tau=0.75 where the pattern is large."""
    return 0.65 + (np.clip(x1, 0.1, 0.5) - 0.1) / 2.0


class Net(nn.Module):
    def __init__(self, nin, nout):
        super().__init__()
        self.trunk = nn.Sequential(
            nn.Linear(nin, WIDTH), nn.ReLU(),
            nn.Linear(WIDTH, WIDTH), nn.ReLU(),
            nn.Linear(WIDTH, WIDTH), nn.ReLU())
        self.head = nn.Linear(WIDTH, nout)

    def forward(self, z):
        return self.head(self.trunk(z))


def train_arm(arm, seed, X, Y):
    torch.manual_seed(seed)
    rng = np.random.RandomState(seed)
    Xt = torch.tensor(X, dtype=torch.float32, device=DEV)
    Yt = torch.tensor(Y, dtype=torch.float32, device=DEV)
    if arm in ("flow", "flowgrid"):
        net = Net(NX + NY + 1, NY).to(DEV)
    elif arm == "l2eps":
        net = Net(NX + NY, NY).to(DEV)
    elif arm == "l2aux":
        net = Net(NX, NY + NX).to(DEV)
        At = torch.tensor((X @ W_AUX.T), dtype=torch.float32, device=DEV)
    else:
        net = Net(NX, NY).to(DEV)
    opt = torch.optim.Adam(net.parameters(), lr=1e-3)
    if arm == "flowgrid":  # frozen (eps, t) pairs per sample, no fresh draws
        K = 8
        EPS = torch.tensor(rng.randn(len(X), K, NY), dtype=torch.float32, device=DEV)
        TS = torch.tensor(rng.rand(len(X), K, 1), dtype=torch.float32, device=DEV)
    for it in range(STEPS):
        idx = rng.randint(0, len(X), 256)
        xb, yb = Xt[idx], Yt[idx]
        if arm == "l2":
            loss = ((net(xb) - yb) ** 2).mean()
        elif arm == "l2jit":
            loss = ((net(xb + JIT * torch.randn_like(xb)) - yb) ** 2).mean()
        elif arm == "l2eps":
            eps = torch.randn(len(xb), NY, device=DEV)
            loss = ((net(torch.cat([xb, eps], 1)) - yb) ** 2).mean()
        elif arm == "l2aux":
            out = net(xb)
            loss = ((out[:, :NY] - yb) ** 2).mean() \
                + ((out[:, NY:] - At[idx]) ** 2).mean()
        else:
            if arm == "flow":
                eps = torch.randn(len(xb), NY, device=DEV)
                t = torch.rand(len(xb), 1, device=DEV)
            else:  # flowgrid
                kk = torch.randint(0, 8, (len(xb),), device=DEV)
                eps = EPS[idx, kk]
                t = TS[idx, kk]
            yt = (1 - t) * eps + t * yb
            loss = ((net(torch.cat([xb, yt, t], 1)) - (yb - eps)) ** 2).mean()
        opt.zero_grad()
        loss.backward()
        opt.step()
    return net


def policy_out(arm, net, Xq):
    """deterministic representative output (flow: eps=0 endpoint, 32 steps)."""
    Z = torch.tensor(Xq, dtype=torch.float32, device=DEV)
    with torch.no_grad():
        if arm in ("flow", "flowgrid"):
            y = torch.zeros(len(Z), NY, device=DEV)
            for j in range(NS_FLOW):
                t = torch.full((len(Z), 1), j / NS_FLOW, device=DEV)
                y = y + (1.0 / NS_FLOW) * net(torch.cat([Z, y, t], 1))
            return y.cpu().numpy()
        if arm == "l2eps":
            eps = torch.zeros(len(Z), NY, device=DEV)
            return net(torch.cat([Z, eps], 1)).cpu().numpy()
        out = net(Z).cpu().numpy()
        return out[:, :NY] if arm == "l2aux" else out


def jac_pr(arm, net, Xq):
    """participation ratio of d policy/d x singular values, mean over points."""
    prs = []
    for x0 in Xq:
        z = torch.tensor(x0[None, :], dtype=torch.float32, device=DEV,
                         requires_grad=True)
        if arm in ("flow", "flowgrid"):
            y = torch.zeros(1, NY, device=DEV)
            for j in range(NS_FLOW):
                t = torch.full((1, 1), j / NS_FLOW, device=DEV)
                y = y + (1.0 / NS_FLOW) * net(torch.cat([z, y, t], 1))
            out = y
        elif arm == "l2eps":
            out = net(torch.cat([z, torch.zeros(1, NY, device=DEV)], 1))
        else:
            out = net(z)
            out = out[:, :NY]
        J = torch.zeros(NY, NX)
        for j in range(NY):
            gr = torch.autograd.grad(out[0, j], z, retain_graph=True)[0][0]
            J[j] = gr[:NX].cpu()
        s = np.linalg.svd(J.numpy(), compute_uv=False)
        prs.append((s.sum() ** 2) / (np.square(s).sum() + 1e-12))
    return float(np.mean(prs))


def field_check(net):
    """eigenvalues of (1-t) * d v/d y_t: requirement says ~ -1 (full rank)."""
    vals = []
    rngl = np.random.RandomState(0)
    for _ in range(6):
        x0 = g(rngl.rand())[0]
        t0 = rngl.uniform(0.2, 0.8)
        yt0 = rngl.randn(NY) * 0.7
        z = torch.tensor(np.concatenate([x0, yt0, [t0]])[None, :],
                         dtype=torch.float32, device=DEV, requires_grad=True)
        out = net(z)
        J = torch.zeros(NY, NY)
        for j in range(NY):
            gr = torch.autograd.grad(out[0, j], z, retain_graph=True)[0][0]
            J[j] = gr[NX:NX + NY].cpu()
        ev = np.linalg.eigvals(J.numpy()) * (1 - t0)
        vals.append(np.real(ev))
    v = np.concatenate(vals)
    return float(v.mean()), float(v.std())


def main():
    taus = np.linspace(0.01, 0.99, N)
    X, Y = g(taus), h(taus)
    # y-manifold for validity distance
    tt = np.linspace(0, 1, 4000)
    YMAN = h(tt)
    quiet_t = np.linspace(0.42, 0.58, 24)
    act_t = np.concatenate([np.linspace(0.05, 0.35, 12),
                            np.linspace(0.62, 0.78, 12)])
    rngp = np.random.RandomState(123)
    results = {}
    for arm in ARMS:
        for seed in SEEDS:
            net = train_arm(arm, seed, X, Y)
            r = {}
            r["pr_quiet"] = jac_pr(arm, net, g(quiet_t))
            r["pr_active"] = jac_pr(arm, net, g(act_t))
            # ON-SUPPORT per-band fit (the metric smoothing arms sacrifice)
            tq = np.linspace(0.42, 0.58, 40)
            ts = np.linspace(0.66, 0.84, 40)
            oq = policy_out(arm, net, g(tq))
            os_ = policy_out(arm, net, g(ts))
            r["fit_quiet"] = float(np.median(np.linalg.norm(oq - h(tq), axis=1)))
            r["fit_stroke"] = float(np.median(np.linalg.norm(os_ - h(ts), axis=1)))
            for rad in (0.1, 0.3):
                errs, dists, mcos, mnorm = [], [], [], []
                for t0 in np.concatenate([quiet_t, act_t]):
                    x0, y0 = g(t0)[0], h(t0)[0]
                    for _ in range(6):
                        u = rngp.randn(NX)
                        u /= np.linalg.norm(u)
                        out = policy_out(arm, net, (x0 + rad * u)[None, :])[0]
                        errs.append(np.linalg.norm(out - y0))
                        dists.append(np.min(np.linalg.norm(YMAN - out, axis=1)))
                        if 0.42 <= t0 <= 0.58:
                            ya = h(tau_alias(x0[0]))[0]
                            mcos.append(np.dot(out, ya)
                                        / (np.linalg.norm(out) * np.linalg.norm(ya) + 1e-9))
                            mnorm.append(np.linalg.norm(out)
                                         / (np.linalg.norm(ya) + 1e-9))
                r[f"bias_r{rad}"] = float(np.median(errs))
                r[f"distY_r{rad}"] = float(np.median(dists))
                r[f"miscos_r{rad}"] = float(np.median(mcos))
                r[f"misnorm_r{rad}"] = float(np.median(mnorm))
            if arm in ("flow", "flowgrid"):
                m, s = field_check(net)
                r["field_ev_mean"] = m
                r["field_ev_std"] = s
            results[f"{arm}_s{seed}"] = r
            print(f"{arm} s{seed}: PR q/a {r['pr_quiet']:.1f}/{r['pr_active']:.1f} | "
                  f"fitOn q/s {r['fit_quiet']:.3f}/{r['fit_stroke']:.3f} | "
                  f"bias .1/.3 {r['bias_r0.1']:.3f}/{r['bias_r0.3']:.3f} | "
                  f"distY .3 {r['distY_r0.3']:.3f} | "
                  f"miscos .3 {r['miscos_r0.3']:.2f} misnorm {r['misnorm_r0.3']:.2f}"
                  + (f" | fieldEV {r['field_ev_mean']:.2f}±{r['field_ev_std']:.2f}"
                     if "field_ev_mean" in r else ""), flush=True)
    json.dump(results, open("toyavb.json", "w"), indent=1)
    print("\n=== TABLE (mean over seeds) ===")
    for arm in ARMS:
        ks = [k for k in results if k.startswith(arm + "_s")]
        agg = {m: np.mean([results[k][m] for k in ks])
               for m in results[ks[0]] if not m.startswith("field")}
        print(f"{arm:>9}: PRq {agg['pr_quiet']:.1f} PRa {agg['pr_active']:.1f} | "
              f"fitOn q/s {agg['fit_quiet']:.3f}/{agg['fit_stroke']:.3f} | "
              f"bias@.3 {agg['bias_r0.3']:.3f} | distY@.3 {agg['distY_r0.3']:.3f} | "
              f"miscos@.3 {agg['miscos_r0.3']:.2f}")


if __name__ == "__main__":
    main()
