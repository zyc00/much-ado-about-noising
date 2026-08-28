"""Rank-collapse toy (fresh start, self-contained).

Setup: latent s ~ U[-1,1]^4. Input x (12-d) = three groups:
  A = s + eps            (linear encoding)
  B = (R s)^3 + eps      (same information, needs cube-root decode)
  C = pure N(0,1) noise  (spurious-attribution control)
Equal scales after min-max IO normalization, equal obs noise NOISE per dim, so
the Bayes predictor averages A and B; ignoring B is strictly suboptimal.
Target: y = tanh(1.5 * U s), U orthogonal (full-rank, smooth).

Hypothesis under test (pre-registered, PART CCCIV): MSE commits to A ->
input-Jacobian B-share ~ 0, feature-Jacobian effective rank ~ 4 with null
space over B, feature-covariance PR low, corrupt-A error catastrophic while
corrupt-B is free. MIP/Flow retain higher B-share / rank / PR and degrade
more gracefully under corrupt-A. C-share ~ 0 for all arms (sanity).

Arms: l2, mip (2-step flow sampler), flow8 (8-step). 3 seeds each.
Metrics -> stdout table + toyrank.json. No engineered eval rules.
"""
import json
import os

import numpy as np
import torch
import torch.nn as nn

DEV = "cuda" if torch.cuda.is_available() else "cpu"
R_LAT = 4
DIN, DOUT = 12, 4
NOISE = float(os.environ.get("NOISE", "0.05"))
N_TRAIN = int(os.environ.get("N", "20000"))
STEPS = int(os.environ.get("STEPS", "20000"))
WIDTH = int(os.environ.get("WIDTH", "64"))
BATCH = 256
SEEDS = [int(s) for s in os.environ.get("SEEDS", "0,1,2").split(",")]

_r = np.random.RandomState(7)
R_MIX = np.linalg.qr(_r.randn(R_LAT, R_LAT))[0]
U_OUT = np.linalg.qr(_r.randn(R_LAT, R_LAT))[0]


def g(s):
    return np.tanh(1.5 * s @ U_OUT.T)


def make_data(n, rng):
    s = rng.uniform(-1, 1, (n, R_LAT))
    A = s + NOISE * rng.randn(n, R_LAT)
    B = (s @ R_MIX.T) ** 3 + NOISE * rng.randn(n, R_LAT)
    C = rng.randn(n, R_LAT)
    return np.concatenate([A, B, C], 1).astype(np.float32), g(s).astype(np.float32), s


class Reg(nn.Module):
    def __init__(self):
        super().__init__()
        self.trunk = nn.Sequential(nn.Linear(DIN, WIDTH), nn.ReLU(),
                                   nn.Linear(WIDTH, WIDTH), nn.ReLU())
        self.head = nn.Linear(WIDTH, DOUT)

    def feats(self, xn):
        return self.trunk(xn)

    def forward(self, xn):
        return self.head(self.trunk(xn))


class FlowNet(nn.Module):
    def __init__(self, nsteps):
        super().__init__()
        self.nsteps = nsteps
        self.trunk = nn.Sequential(nn.Linear(DIN + DOUT + 1, WIDTH), nn.ReLU(),
                                   nn.Linear(WIDTH, WIDTH), nn.ReLU())
        self.head = nn.Linear(WIDTH, DOUT)

    def vel(self, xn, yt, t):
        return self.head(self.trunk(torch.cat([xn, yt, t], 1)))

    def feats(self, xn):
        z = torch.zeros(len(xn), DOUT + 1, device=xn.device)
        return self.trunk(torch.cat([xn, z], 1))

    def forward(self, xn):
        y = torch.zeros(len(xn), DOUT, device=xn.device)
        for k in range(self.nsteps):
            t = torch.full((len(xn), 1), k / self.nsteps, device=xn.device)
            y = y + (1.0 / self.nsteps) * self.vel(xn, y, t)
        return y


def train(arm, seed, X, Y):
    torch.manual_seed(seed)
    net = (Reg() if arm == "l2" else FlowNet(2 if arm == "mip" else 8)).to(DEV)
    opt = torch.optim.Adam(net.parameters(), lr=1e-3)
    Xt = torch.tensor(X, device=DEV)
    Yt = torch.tensor(Y, device=DEV)
    rng = np.random.RandomState(seed)
    for _ in range(STEPS):
        idx = rng.randint(0, len(Xt), BATCH)
        xb, yb = Xt[idx], Yt[idx]
        if arm == "l2":
            loss = ((net(xb) - yb) ** 2).mean()
        else:
            eps = torch.randn_like(yb)
            t = torch.rand(len(yb), 1, device=DEV)
            yt = (1 - t) * eps + t * yb
            loss = ((net.vel(xb, yt, t) - (yb - eps)) ** 2).mean()
        opt.zero_grad()
        loss.backward()
        opt.step()
    return net


def probe(net, Xn_probe):
    """Jacobian group shares, feature-Jacobian effective rank, feature PR."""
    xp = torch.tensor(Xn_probe, device=DEV, requires_grad=True)
    yp = net(xp)
    Jrows = [torch.autograd.grad(yp[:, k].sum(), xp, retain_graph=True)[0]
             for k in range(DOUT)]
    J = torch.stack(Jrows, 1)  # (n, DOUT, DIN)
    sq = (J ** 2).sum((0, 1))
    gA, gB, gC = sq[:4].sum().item(), sq[4:8].sum().item(), sq[8:].sum().item()

    eranks = []
    for i in range(16):
        xi = torch.tensor(Xn_probe[i:i + 1], device=DEV)
        Jf = torch.autograd.functional.jacobian(
            lambda v: net.feats(v), xi, vectorize=True)[0, :, 0, :]
        sv = torch.linalg.svdvals(Jf)
        p = sv ** 2
        eranks.append(((p.sum() ** 2) / (p ** 2).sum()).item())

    with torch.no_grad():
        Fall = net.feats(torch.tensor(Xn_probe, device=DEV))
        Fc = Fall - Fall.mean(0, keepdim=True)
        ev = torch.linalg.eigvalsh(Fc.T @ Fc / len(Fc))
        ev = torch.clamp(ev, min=0)
        pr = ((ev.sum() ** 2) / ((ev ** 2).sum() + 1e-12)).item()
    return dict(bshare=gB / (gA + gB), cshare=gC / (gA + gB + gC),
                jrank=float(np.mean(eranks)), fpr=pr)


def main():
    rng = np.random.RandomState(0)
    X, Y, _ = make_data(N_TRAIN, rng)
    xlo, xhi = X.min(0), X.max(0)
    ylo, yhi = Y.min(0), Y.max(0)

    def nx(x):
        return (2 * (x - xlo) / (xhi - xlo) - 1).astype(np.float32)

    def ny(y):
        return (2 * (y - ylo) / (yhi - ylo) - 1).astype(np.float32)

    Xn, Yn = nx(X), ny(Y)
    erng = np.random.RandomState(999)
    Xe, Ye, Se = make_data(4096, erng)
    Xen, Yen = nx(Xe), ny(Ye)

    def corrupt(Xr, group, mode, sev, crng):
        Xc = Xr.copy()
        sl = slice(0, 4) if group == "A" else slice(4, 8)
        if mode == "noise":
            Xc[:, sl] += sev * crng.randn(len(Xc), 4)
        else:
            Xc[:, sl] = Xc[:, sl].mean(0, keepdims=True)
        return Xc

    # oracle anchor: decode s from both groups, average
    sA = Xe[:, :4]
    sB = np.cbrt(Xe[:, 4:8]) @ R_MIX
    orc_clean = float(((ny(g((sA + sB) / 2)) - Yen) ** 2).mean())
    crng = np.random.RandomState(5)
    XcA = corrupt(Xe, "A", "noise", 0.3, crng)
    orc_cA = float(((ny(g((XcA[:, :4] + np.cbrt(XcA[:, 4:8]) @ R_MIX) / 2))
                     - Yen) ** 2).mean())
    print(f"oracle(avg A,B): clean {orc_clean:.5f} | noiseA0.3 {orc_cA:.5f}",
          flush=True)

    results = {}
    for arm in ("l2", "mip", "flow8"):
        rows = []
        for seed in SEEDS:
            net = train(arm, seed, Xn, Yn)
            with torch.no_grad():
                def err(Xr):
                    p = net(torch.tensor(nx(Xr), device=DEV)).cpu().numpy()
                    return float(((p - Yen) ** 2).mean())
                e_clean = err(Xe)
                crng = np.random.RandomState(5)
                e_nA = err(corrupt(Xe, "A", "noise", 0.3, crng))
                crng = np.random.RandomState(5)
                e_nB = err(corrupt(Xe, "B", "noise", 0.3, crng))
            m = probe(net, Xen[:512])
            m.update(clean=e_clean, noiseA=e_nA, noiseB=e_nB)
            rows.append(m)
            print(f"{arm} s{seed}: clean {e_clean:.5f} Bshare {m['bshare']:.3f} "
                  f"Cshare {m['cshare']:.3f} Jrank {m['jrank']:.2f} "
                  f"fPR {m['fpr']:.2f} | noiseA {e_nA:.5f} noiseB {e_nB:.5f}",
                  flush=True)
        results[arm] = rows

    print("\n=== TABLE (mean+-sd over seeds; errors in normalized-y MSE) ===",
          flush=True)
    hdr = ("arm", "clean", "B-share", "C-share", "featJ-erank", "feat-PR",
           "noiseA0.3", "noiseB0.3")
    print(" | ".join(f"{h:>11}" for h in hdr), flush=True)
    for arm, rows in results.items():
        def ms(k):
            v = [r[k] for r in rows]
            return f"{np.mean(v):.3f}±{np.std(v):.3f}"
        print(" | ".join([f"{arm:>11}"] + [f"{ms(k):>11}" for k in
              ("clean", "bshare", "cshare", "jrank", "fpr", "noiseA", "noiseB")]),
              flush=True)
    json.dump({"results": results, "oracle": {"clean": orc_clean,
               "noiseA": orc_cA}}, open("toyrank.json", "w"), indent=1)
    print("wrote toyrank.json", flush=True)


if __name__ == "__main__":
    main()
