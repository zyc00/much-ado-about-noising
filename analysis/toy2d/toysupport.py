"""Support-geometry toy (pre-reg PART CCCXIII). Fully deterministic, ZERO
noise. 1-D contact servo: descend to surface h (stiffness k), press to
target force f*=0.06, hold 4 steps; f > 2f* breaks (intrinsic fail).
Obs [p, h, k, f]: M1=(p,h,k) fully explains the label everywhere via the
delicate composed chart h - f*/k; M2=f is zero out of contact and gives
the robust affine chart (f - f*) in-band. Scarcity knob NEP.
Arms: l2 / mip / flow8 / l2nf (force column masked train+eval).
"""
import json
import os

import numpy as np
import torch
import torch.nn as nn

DEV = "cuda" if torch.cuda.is_available() else "cpu"
H = 8
FSTAR, FBREAK, FTOL = 0.06, 0.12, 0.02
GAIN, CLIP = 0.35, 0.08
HOLD_N = 4
STEPS = int(os.environ.get("STEPS", "30000"))
WIDTH = int(os.environ.get("WIDTH", "64"))
SEEDS = [int(s) for s in os.environ.get("SEEDS", "0,1").split(",")]
NEPS = [int(n) for n in os.environ.get("NEPS", "5,15,50,200").split(",")]


def draw_ep(rng):
    return rng.uniform(0.8, 1.8), rng.uniform(0.0, 0.3), rng.uniform(1.0, 10.0)


def force(p, h, k):
    return k * max(0.0, h - p)


def expert_action(p, h, k):
    return float(np.clip(GAIN * (h - FSTAR / k - p), -CLIP, CLIP))


def gen_episode(rng):
    p0, h, k = draw_ep(rng)
    p, hold, tail = p0, 0, H + 2
    states, acts = [], []
    for _ in range(120):
        f = force(p, h, k)
        a = expert_action(p, h, k)
        states.append([p, h, k, f])
        acts.append(a)
        p = p + a
        f2 = force(p, h, k)
        hold = hold + 1 if abs(f2 - FSTAR) <= FTOL else 0
        if hold >= HOLD_N:
            tail -= 1
            if tail <= 0:
                break
    return np.array(states, np.float32), np.array(acts, np.float32)


def build_dataset(nep, rng):
    X, Y = [], []
    for _ in range(nep):
        st, ac = gen_episode(rng)
        for t in range(len(st) - H):
            X.append(st[t])
            Y.append(ac[t:t + H])
    return np.array(X, np.float32), np.array(Y, np.float32)


class Reg(nn.Module):
    def __init__(self, hetero=False):
        super().__init__()
        self.hetero = hetero
        self.trunk = nn.Sequential(nn.Linear(4, WIDTH), nn.ReLU(),
                                   nn.Linear(WIDTH, WIDTH), nn.ReLU())
        self.head = nn.Linear(WIDTH, 2 * H if hetero else H)

    def full(self, xn):
        return self.head(self.trunk(xn))

    def forward(self, xn):
        out = self.head(self.trunk(xn))
        return out[:, :H] if self.hetero else out


class FlowNet(nn.Module):
    def __init__(self, nsteps):
        super().__init__()
        self.nsteps = nsteps
        self.trunk = nn.Sequential(nn.Linear(4 + H + 1, WIDTH), nn.ReLU(),
                                   nn.Linear(WIDTH, WIDTH), nn.ReLU())
        self.head = nn.Linear(WIDTH, H)

    def vel(self, xn, yt, t):
        return self.head(self.trunk(torch.cat([xn, yt, t], 1)))

    def forward(self, xn):
        y = torch.zeros(len(xn), H, device=xn.device)
        for j in range(self.nsteps):
            t = torch.full((len(xn), 1), j / self.nsteps, device=xn.device)
            y = y + (1.0 / self.nsteps) * self.vel(xn, y, t)
        return y


class Norm:
    def __init__(self, X, Y):
        self.xlo, self.xhi = X.min(0), X.max(0)
        self.ylo, self.yhi = Y.min(0), Y.max(0)

    def nx(self, x):
        return (2 * (x - self.xlo) / (self.xhi - self.xlo + 1e-8) - 1).astype(np.float32)

    def uy(self, yn):
        return (yn + 1) / 2 * (self.yhi - self.ylo + 1e-8) + self.ylo

    def ny(self, y):
        return (2 * (y - self.ylo) / (self.yhi - self.ylo + 1e-8) - 1).astype(np.float32)


def train(arm, seed, Xn, Yn):
    torch.manual_seed(seed)
    net = (FlowNet(2 if arm == "mip" else 8) if arm in ("mip", "flow8")
           else Reg(hetero=arm in ("hg", "ht"))).to(DEV)
    opt = torch.optim.Adam(net.parameters(), lr=1e-3)
    Xt = torch.tensor(Xn, device=DEV)
    Yt = torch.tensor(Yn, device=DEV)
    rng = np.random.RandomState(seed)
    for _ in range(STEPS):
        idx = rng.randint(0, len(Xt), min(256, len(Xt)))
        xb, yb = Xt[idx], Yt[idx]
        if arm in ("mip", "flow8"):
            eps = torch.randn_like(yb)
            t = torch.rand(len(yb), 1, device=DEV)
            yt = (1 - t) * eps + t * yb
            loss = ((net.vel(xb, yt, t) - (yb - eps)) ** 2).mean()
        elif arm in ("hg", "ht"):
            out = net.full(xb)
            mu, lv = out[:, :H], out[:, H:].clamp(-10, 2)
            r2 = (yb - mu) ** 2 * torch.exp(-lv)
            if arm == "hg":
                loss = (0.5 * (lv + r2)).mean()
            else:
                nu = 2.0
                loss = ((nu + 1) / 2 * torch.log1p(r2 / nu) + 0.5 * lv).mean()
        else:
            loss = ((net(xb) - yb) ** 2).mean()
        opt.zero_grad()
        loss.backward()
        opt.step()
    return net


def obs_of(p, h, k, arm):
    f = force(p, h, k)
    o = np.array([p, h, k, 0.0 if arm == "l2nf" else f], np.float32)
    return o


def rollout(net, norm, arm, ep, AS=1, ablate_f=False):
    p0, h, k = ep
    p, hold = p0, 0
    for _ in range(0, 160, AS):
        if arm == "expert":
            chunk = np.array([expert_action(p, h, k)])
        else:
            o = obs_of(p, h, k, arm).copy()
            if ablate_f:
                o[3] = 0.0
            on = torch.tensor(norm.nx(o[None]), device=DEV)
            with torch.no_grad():
                chunk = norm.uy(net(on)[0].cpu().numpy())
        for a in chunk[:AS]:
            p = p + float(np.clip(a, -CLIP, CLIP))
            f = force(p, h, k)
            if f > FBREAK:
                return "break"
            hold = hold + 1 if abs(f - FSTAR) <= FTOL else 0
            if hold >= HOLD_N:
                return "success"
    return "timeout"


def evaluate(net, norm, arm, eps, AS=1, ablate_f=False):
    outs = [rollout(net, norm, arm, ep, AS, ablate_f) for ep in eps]
    return {o: outs.count(o) / len(outs) for o in ("success", "break", "timeout")}


def fshare(net, norm, Xn_band):
    xp = torch.tensor(Xn_band, device=DEV, requires_grad=True)
    a0 = net(xp)[:, 0].sum()
    g = torch.autograd.grad(a0, xp)[0].pow(2).sum(0)
    return float(g[3] / (g.sum() + 1e-12))


def band_err(net, norm, arm):
    errs = []
    for k in np.linspace(1.0, 10.0, 19):
        for d in np.linspace(0.002, 0.02, 10):
            h = 0.15
            p = h - d
            o = obs_of(p, h, k, arm)
            on = torch.tensor(norm.nx(o[None]), device=DEV)
            with torch.no_grad():
                a = float(norm.uy(net(on)[0].cpu().numpy())[0])
            errs.append(abs(a - expert_action(p, h, k)))
    return float(np.mean(errs))


def main():
    erng = np.random.RandomState(4242)
    eval_eps = [draw_ep(erng) for _ in range(100)]
    ex = [rollout(None, None, "expert", ep) for ep in eval_eps]
    # expert rollout: reuse rollout with a shim
    results = {}
    if os.path.exists("toysupport.json"):
        results.update(json.load(open("toysupport.json")))
    print(f"expert SR {np.mean([o == 'success' for o in ex]):.2f}", flush=True)
    for nep in NEPS:
        drng = np.random.RandomState(1000 + nep)
        Xfull, Yfull = build_dataset(nep, drng)
        band_frac = float((Xfull[:, 3] > 0).mean())
        print(f"NEP={nep}: {len(Xfull)} samples, in-band frac {band_frac:.2f}",
              flush=True)
        arms = tuple(os.environ.get("ARMS", "l2,mip,flow8,l2nf").split(","))
        for arm in arms:
            X = Xfull.copy()
            if arm == "l2nf":
                X[:, 3] = 0.0
            norm = Norm(X, Yfull)
            Xn, Yn = norm.nx(X), norm.ny(Yfull)
            band_idx = np.where(Xfull[:, 3] > 0)[0][:256]
            for seed in SEEDS:
                net = train(arm, seed, Xn, Yn)
                r1 = evaluate(net, norm, arm, eval_eps, AS=1)
                r8 = evaluate(net, norm, arm, eval_eps, AS=8)
                rab = evaluate(net, norm, arm, eval_eps, AS=1, ablate_f=True)
                fs = fshare(net, norm, Xn[band_idx])
                be = band_err(net, norm, arm)
                key = f"nep{nep}_{arm}_s{seed}"
                results[key] = dict(sr1=r1["success"], br1=r1["break"],
                                    sr8=r8["success"], srab=rab["success"],
                                    fshare=fs, banderr=be)
                print(f"NEP={nep} {arm} s{seed}: SR1 {r1['success']:.2f} "
                      f"(brk {r1['break']:.2f}) SR8 {r8['success']:.2f} | "
                      f"SR-noF {rab['success']:.2f} | F-share {fs:.3f} | "
                      f"banderr {be:.4f}", flush=True)
    tarms = sorted({k.split("_")[1] for k in results})
    print("\n=== TABLE (mean over seeds) ===", flush=True)
    print(f"{'nep':>5} {'arm':>6} | {'SR1':>5} {'brk':>5} {'SR8':>5} "
          f"{'SRnoF':>5} | {'Fshare':>6} {'bandErr':>7}", flush=True)
    for nep in NEPS:
        for arm in tarms:
            rows = [results[k] for k in (f"nep{nep}_{arm}_s{s}" for s in SEEDS)
                    if k in results]
            if not rows:
                continue
            m = {kk: np.mean([r[kk] for r in rows]) for kk in rows[0]}
            print(f"{nep:>5} {arm:>6} | {m['sr1']:5.2f} {m['br1']:5.2f} "
                  f"{m['sr8']:5.2f} {m['srab']:5.2f} | {m['fshare']:6.3f} "
                  f"{m['banderr']:7.4f}", flush=True)
    json.dump(results, open("toysupport.json", "w"), indent=1)
    print("wrote toysupport.json", flush=True)


if __name__ == "__main__":
    main()
