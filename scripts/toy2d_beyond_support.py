"""Anti-MIP scenarios in the corridor toy (clean labels, h=48/3k).

S1 beyond-support starts: train y0 in [-15,15], evaluate from far starts
|y0| up to 45. True field is the capped linear servo everywhere; the
correct far response = scaled-up demonstrated correction. Gate |y|<1mm at
x=40 (x advances ~4mm/step, so y must be corrected fast).
S2 tiny-data tax: K in {3,5,10,30} demos, fresh in-support starts.
Also: response profile a_y(y0) at x=5 per method (mechanism), and an
L2-ensemble-of-2 control on the sigma=2 label-noise cell (tests whether
MIP's denoise edge is implicit ensembling)."""
import pickle

import numpy as np
import torch
import torch.nn as nn

dev = "cuda" if torch.cuda.is_available() else "cpu"
KP, CAP, TOL = 0.3, 4.0, 1.0


def servo(p, t):
    return np.clip(KP * (t - p), -CAP, CAP)


def collect(y0):
    p = np.array([0.0, y0])
    rows = []
    for _ in range(60):
        a = servo(p, np.array([50.0, 0.0]))
        rows.append([*p, y0, *a])
        p = p + a
        if p[0] >= 45:
            break
    return np.array(rows)


def mlp(inp, out, h=48):
    return nn.Sequential(nn.Linear(inp, h), nn.SiLU(),
                         nn.Linear(h, h), nn.SiLU(), nn.Linear(h, out))


def build(K, sigma, seed):
    ys = np.linspace(-15, 15, K)
    data = np.concatenate([collect(y) for y in ys])
    rng = np.random.RandomState(3000 + seed)
    S = torch.tensor(data[:, :3], dtype=torch.float32, device=dev)
    A = torch.tensor(data[:, 3:5] + rng.randn(len(data), 2) * sigma,
                     dtype=torch.float32, device=dev)
    return S, A, S.mean(0), S.std(0) + 1e-6


def train(kind, S, A, mu, sd, seed, steps=3000, h=48):
    torch.manual_seed(seed)
    asd = float(A.std())
    if kind == "L2":
        net = mlp(3, 2, h).to(dev)
    elif kind == "MIP":
        net = mlp(5, 2, h).to(dev)
    else:
        net = mlp(6, 2, h).to(dev)
    opt = torch.optim.AdamW(net.parameters(), lr=1e-3, weight_decay=1e-5)
    for _ in range(steps):
        i = torch.randint(0, len(S), (256,), device=dev)
        s, a = (S[i] - mu) / sd, A[i]
        if kind == "L2":
            loss = ((net(s) - a) ** 2).mean()
        elif kind == "MIP":
            z = torch.randn_like(a) * asd
            y0 = net(torch.cat([z / asd, s], 1))
            y1 = net(torch.cat([y0.detach() / asd, s], 1))
            loss = ((y0 - a) ** 2).mean() + ((y1 - a) ** 2).mean()
        else:
            z = torch.randn_like(a) * asd
            t = torch.rand(len(a), 1, device=dev)
            at = (1 - t) * z + t * a
            loss = ((net(torch.cat([at / asd, t, s], 1)) - (a - z)) ** 2
                    ).mean()
        opt.zero_grad(); loss.backward(); opt.step()

    def f(s):
        sn = (s - mu) / sd
        if kind == "L2":
            return net(sn)
        if kind == "MIP":
            z = torch.randn(s.shape[0], 2, device=dev)
            y0 = net(torch.cat([z, sn], 1))
            return net(torch.cat([y0 / asd, sn], 1))
        a = torch.randn(s.shape[0], 2, device=dev) * asd
        for k in range(10):
            t = torch.full((s.shape[0], 1), k / 10.0, device=dev)
            a = a + 0.1 * net(torch.cat([a / asd, t, sn], 1))
        return a
    return f


def roll(f, y0, nmax=60):
    p = np.array([0.0, y0])
    n = 0
    while p[0] < 40.0 and n < nmax and abs(p[1]) < 200:
        s = torch.tensor([[*p, y0]], dtype=torch.float32, device=dev)
        with torch.no_grad():
            a = f(s)[0].cpu().numpy()
        p = p + a
        n += 1
    return (p[0] >= 40.0 and abs(p[1]) < TOL), abs(p[1])


SEEDS = [1, 2, 3]
METH = ["L2", "MIP", "Flow"]
print("== S1: beyond-support starts (train |y0|<=15, clean, h48/3k) ==")
FAR = [10, 20, 30, 40]
res_far = {}
prof = {}
for name in METH:
    fs = []
    for sd in SEEDS:
        S, A, mu, sdv = build(30, 0.0, sd)
        fs.append(train(name, S, A, mu, sdv, sd))
    for y0a in FAR:
        oks = []
        for f in fs:
            for sgn in (1, -1):
                ok, _ = roll(f, sgn * y0a)
                oks.append(ok)
        res_far[(name, y0a)] = np.mean(oks)
        print(f"  |y0|={y0a:2d} {name:5s} SR {np.mean(oks):.2f}", flush=True)
    # response profile a_y at x=5 over y0 grid (seed-1 model, 20-sample avg)
    ys = np.linspace(-45, 45, 61)
    av = []
    for y in ys:
        s = torch.tensor([[5.0, y, y]] * 20, dtype=torch.float32,
                         device=dev)
        with torch.no_grad():
            av.append(float(fs[0](s)[:, 1].mean()))
    prof[name] = (ys, np.array(av))

print("== S2: tiny-data tax (clean, fresh starts U(-15,15)) ==")
rngE = np.random.RandomState(9)
YE = rngE.uniform(-15, 15, 40)
res_k = {}
for K in [3, 5, 10, 30]:
    for name in METH:
        oks = []
        for sd in SEEDS:
            S, A, mu, sdv = build(K, 0.0, sd)
            f = train(name, S, A, mu, sdv, sd)
            oks += [roll(f, y)[0] for y in YE]
        res_k[(name, K)] = np.mean(oks)
        print(f"  K={K:2d} {name:5s} SR {np.mean(oks):.2f}", flush=True)

print("== control: L2-ensemble-of-2 on sigma=2 label noise ==")
oks, oks1 = [], []
for sd in SEEDS:
    S, A, mu, sdv = build(30, 2.0, sd)
    f1 = train("L2", S, A, mu, sdv, sd)
    f2 = train("L2", S, A, mu, sdv, sd + 50)
    f_ens = lambda s: 0.5 * (f1(s) + f2(s))
    oks += [roll(f_ens, y)[0] for y in YE]
    oks1 += [roll(f1, y)[0] for y in YE]
print(f"  L2 single SR {np.mean(oks1):.2f}  L2-ens2 SR {np.mean(oks):.2f}",
      flush=True)

with open("analysis/toy2d_beyond.pkl", "wb") as fh:
    pickle.dump({"res_far": res_far, "prof": prof, "res_k": res_k}, fh)
print("SAVED")
