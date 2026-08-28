"""2D toy for the no-free-lunch designs.

Task (2D plane): start (0,0) -> anchor A=(60,g), g~U(-15,15) -> merge to
S=(120,0) -> precision approach to goal line x=158; success = |y| < 1 mm
at the crossing. Clean capped servo (kp=0.3, cap=4), obs=(x,y,g),
action=(dx,dy). No absorbing failures: the only way to fail is imprecision
at the gate.

Design 1 (label noise): training targets a := a_clean + N(0, sigma^2 I),
states stay clean. L2 = conditional mean = optimal denoiser; Flow fits the
conditional distribution and re-injects the noise at inference.
Design 2 (capacity/steps starvation): clean data, shrink hidden width and
training steps.
"""
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn

dev = "cuda" if torch.cuda.is_available() else "cpu"
KP, CAP = 0.3, 4.0
A_X, S_PT, G_X, TOL = 60.0, np.array([120.0, 0.0]), 158.0, 1.0
K_DEMO = 30
G_TRAIN = np.linspace(-15, 15, K_DEMO)


def servo(p, t):
    return np.clip(KP * (t - p), -CAP, CAP)


def collect(g):
    p = np.zeros(2)
    phase = 1
    rows = []
    for _ in range(300):
        t = (np.array([A_X, g]) if phase == 1 else
             S_PT if phase == 2 else np.array([170.0, 0.0]))
        a = servo(p, t)
        rows.append([*p, g, *a])
        p = p + a
        if phase == 1 and np.linalg.norm(p - np.array([A_X, g])) < 3:
            phase = 2
        elif phase == 2 and np.linalg.norm(p - S_PT) < 3:
            phase = 3
        if p[0] >= 165:
            break
    return np.array(rows)


DEMOS = [collect(g) for g in G_TRAIN]
DATA = np.concatenate(DEMOS)


def mlp(inp, out, h):
    return nn.Sequential(nn.Linear(inp, h), nn.SiLU(),
                         nn.Linear(h, h), nn.SiLU(), nn.Linear(h, out))


def make_tensors(sigma, seed):
    rng = np.random.RandomState(1000 + seed)
    S = torch.tensor(DATA[:, :3], dtype=torch.float32, device=dev)
    A = torch.tensor(DATA[:, 3:5] + rng.randn(len(DATA), 2) * sigma,
                     dtype=torch.float32, device=dev)
    mu, sd = S.mean(0), S.std(0) + 1e-6
    return S, A, mu, sd, A.std()


def train_l2(S, A, mu, sd, h, steps, seed):
    torch.manual_seed(seed)
    net = mlp(3, 2, h).to(dev)
    opt = torch.optim.AdamW(net.parameters(), lr=1e-3, weight_decay=1e-5)
    for _ in range(steps):
        i = torch.randint(0, len(S), (256,), device=dev)
        loss = ((net((S[i] - mu) / sd) - A[i]) ** 2).mean()
        opt.zero_grad(); loss.backward(); opt.step()
    return lambda s: net((s - mu) / sd)


def train_mip(S, A, mu, sd, h, steps, seed):
    torch.manual_seed(seed)
    asd = float(A.std())
    net = mlp(5, 2, h).to(dev)
    opt = torch.optim.AdamW(net.parameters(), lr=1e-3, weight_decay=1e-5)
    for _ in range(steps):
        i = torch.randint(0, len(S), (256,), device=dev)
        s, a = (S[i] - mu) / sd, A[i]
        z = torch.randn_like(a) * asd
        y0 = net(torch.cat([z / asd, s], 1))
        y1 = net(torch.cat([y0.detach() / asd, s], 1))
        loss = ((y0 - a) ** 2).mean() + ((y1 - a) ** 2).mean()
        opt.zero_grad(); loss.backward(); opt.step()

    def f(s):
        sn = (s - mu) / sd
        z = torch.randn(s.shape[0], 2, device=dev)
        y0 = net(torch.cat([z, sn], 1))
        return net(torch.cat([y0 / asd, sn], 1))
    return f


def train_flow(S, A, mu, sd, h, steps, seed):
    torch.manual_seed(seed)
    asd = float(A.std())
    net = mlp(6, 2, h).to(dev)
    opt = torch.optim.AdamW(net.parameters(), lr=1e-3, weight_decay=1e-5)
    for _ in range(steps):
        i = torch.randint(0, len(S), (256,), device=dev)
        s, a = (S[i] - mu) / sd, A[i]
        z = torch.randn_like(a) * asd
        t = torch.rand(len(a), 1, device=dev)
        at = (1 - t) * z + t * a
        v = net(torch.cat([at / asd, t, s], 1))
        loss = ((v - (a - z)) ** 2).mean()
        opt.zero_grad(); loss.backward(); opt.step()

    def f(s):
        sn = (s - mu) / sd
        a = torch.randn(s.shape[0], 2, device=dev) * asd
        for k in range(10):
            t = torch.full((s.shape[0], 1), k / 10.0, device=dev)
            a = a + 0.1 * net(torch.cat([a / asd, t, sn], 1))
        return a
    return f


TRAINERS = {"L2": train_l2, "MIP": train_mip, "Flow": train_flow}


def rollout(f, g, nmax=300):
    p = np.zeros(2)
    tr = [p.copy()]
    while len(tr) < nmax and p[0] < G_X and abs(p[1]) < 120:
        s = torch.tensor([[*p, g]], dtype=torch.float32, device=dev)
        with torch.no_grad():
            a = f(s)[0].cpu().numpy()
        p = p + a
        tr.append(p.copy())
    ok = p[0] >= G_X and abs(p[1]) < TOL
    return ok, abs(p[1]) if p[0] >= G_X else np.nan, np.array(tr)


def eval_cell(f):
    oks, ys, trs = [], [], []
    for g in G_TRAIN:
        ok, y, tr = rollout(f, g)
        oks.append(ok); ys.append(y); trs.append(tr)
    return np.mean(oks), np.nanmean(ys), trs


# ---------- Design 1: label-noise sweep ----------
import os
RUN_OLD = not os.environ.get("SKIP_D12")
SIGMAS = [0.0, 0.5, 1.0, 2.0] if RUN_OLD else []
SEEDS = [1, 2, 3]
res1 = {}
tr_show = {}
print("== Design 1: Gaussian action-label noise, h=256, 12k steps ==")
for sig in SIGMAS:
    for name, tr in TRAINERS.items():
        srs, yms = [], []
        for sd in SEEDS:
            S, A, mu, sdv, _ = make_tensors(sig, sd)
            f = tr(S, A, mu, sdv, 256, 12000, sd)
            sr, ym, trs = eval_cell(f)
            srs.append(sr); yms.append(ym)
            if sd == 1:
                tr_show[(name, sig)] = trs
        res1[(name, sig)] = (np.mean(srs), np.mean(yms))
        print(f"  sigma={sig:3.1f} {name:5s} SR@1mm {np.mean(srs):.2f}  "
              f"|y|@goal {np.mean(yms):5.2f}mm", flush=True)

# ---------- Design 2: capacity / steps starvation (clean data) ----------
print("== Design 2: clean data, capacity x steps grid ==")
HS = [16, 48, 256] if RUN_OLD else []
STEPS = [2000, 6000, 12000]
res2 = {}
for h in HS:
    for st in STEPS:
        for name, tr in TRAINERS.items():
            srs = []
            for sd in SEEDS:
                S, A, mu, sdv, _ = make_tensors(0.0, sd)
                f = tr(S, A, mu, sdv, h, st, sd)
                sr, _, _ = eval_cell(f)
                srs.append(sr)
            res2[(name, h, st)] = np.mean(srs)
            print(f"  h={h:3d} steps={st:5d} {name:5s} "
                  f"SR {np.mean(srs):.2f}", flush=True)

import pickle
if RUN_OLD:
    with open("analysis/toy2d_nfl.pkl", "wb") as fh:
        pickle.dump({"res1": res1, "res2": res2,
                     "tr_show": {k: v for k, v in tr_show.items()
                                 if k[1] in (0.0, 2.0)}}, fh)
    print("SAVED analysis/toy2d_nfl.pkl")


# ---------- Design 1b: SHORT-HORIZON precision corridor ----------
# start (0, y0), y0~U(-15,15); goal line x=40; ~15 steps; tolerance 1 mm.
# No fan/merge, no room for compounding: isolates denoise vs re-inject.
def collect_corr(y0):
    p = np.array([0.0, y0])
    rows = []
    for _ in range(60):
        a = servo(p, np.array([50.0, 0.0]))
        rows.append([*p, y0, *a])
        p = p + a
        if p[0] >= 45:
            break
    return np.array(rows)


Y_TRAIN = np.linspace(-15, 15, K_DEMO)
CDATA = np.concatenate([collect_corr(y) for y in Y_TRAIN])


def make_ct(sigma, seed):
    rng = np.random.RandomState(2000 + seed)
    S = torch.tensor(CDATA[:, :3], dtype=torch.float32, device=dev)
    A = torch.tensor(CDATA[:, 3:5] + rng.randn(len(CDATA), 2) * sigma,
                     dtype=torch.float32, device=dev)
    return S, A, S.mean(0), S.std(0) + 1e-6


def rollout_corr(f, y0, nmax=60):
    p = np.array([0.0, y0])
    while p[0] < 40.0 and len(str(0)) and nmax > 0:
        s = torch.tensor([[*p, y0]], dtype=torch.float32, device=dev)
        with torch.no_grad():
            a = f(s)[0].cpu().numpy()
        p = p + a
        nmax -= 1
    return abs(p[1]) < TOL, abs(p[1])


CH = int(os.environ.get("CORR_H", "256"))
CS = int(os.environ.get("CORR_STEPS", "12000"))
print(f"== Design 1b: short corridor (~15 steps), label-noise sweep, "
      f"h={CH}/{CS} ==")
res1b = {}
for sig in [0.0, 1.0, 2.0, 3.0]:
    for name, tr in TRAINERS.items():
        srs, ys = [], []
        for sd in SEEDS:
            S, A, mu, sdv = make_ct(sig, sd)
            f = tr(S, A, mu, sdv, CH, CS, sd)
            for y0 in Y_TRAIN:
                ok, ay = rollout_corr(f, y0)
                srs.append(ok); ys.append(ay)
        res1b[(name, sig)] = (np.mean(srs), np.mean(ys),
                             np.percentile(ys, 90))
        print(f"  sigma={sig:3.1f} {name:5s} SR@1mm {np.mean(srs):.2f}  "
              f"|y| mean {np.mean(ys):5.2f} p90 {np.percentile(ys, 90):5.2f}",
              flush=True)

# mechanism: sampled-action spread at a fixed mid-corridor state
print("== action spread at state (20, 3, y0=3), 200 samples, sigma=2 ==")
spread = {}
for name, tr in TRAINERS.items():
    S, A, mu, sdv = make_ct(2.0, 1)
    f = tr(S, A, mu, sdv, CH, CS, 1)
    s = torch.tensor([[20.0, 3.0, 3.0]] * 200, dtype=torch.float32,
                     device=dev)
    with torch.no_grad():
        aa = f(s).cpu().numpy()
    spread[name] = aa
    print(f"  {name:5s} std_dx {aa[:,0].std():.3f} std_dy "
          f"{aa[:,1].std():.3f}  (label noise 2.0)", flush=True)

# ---------- Design 2b: 3D-toy training-budget cliff (clean data) ----------
A3_X, S3, I3 = 60.0, np.array([120.0, 0.0, 0.0]), np.array(
    [120.0, 0.0, -40.0])


def collect3(g):
    p = np.zeros(3)
    ph = 1
    rows = []
    for _ in range(400):
        t = (np.array([A3_X, g, 0.0]) if ph == 1 else S3 if ph == 2 else I3)
        a = np.clip(KP * (t - p), -CAP, CAP)
        rows.append([*p, g, *a])
        p = p + a
        if ph == 1 and np.linalg.norm(p - np.array([A3_X, g, 0.0])) < 3:
            ph = 2
        elif ph == 2 and np.linalg.norm(p - S3) < 3:
            ph = 3
        if p[2] <= -38:
            break
    return np.array(rows)


D3 = np.concatenate([collect3(g) for g in np.linspace(-15, 15, K_DEMO)])
S3T = torch.tensor(D3[:, :4], dtype=torch.float32, device=dev)
A3T = torch.tensor(D3[:, 4:7], dtype=torch.float32, device=dev)
MU3, SD3 = S3T.mean(0), S3T.std(0) + 1e-6


def train3(kind, h, steps, seed):
    torch.manual_seed(seed)
    asd = float(A3T.std())
    if kind == "L2":
        net = mlp(4, 3, h).to(dev)
    elif kind == "MIP":
        net = mlp(7, 3, h).to(dev)
    else:
        net = mlp(8, 3, h).to(dev)
    opt = torch.optim.AdamW(net.parameters(), lr=1e-3, weight_decay=1e-5)
    for _ in range(steps):
        i = torch.randint(0, len(S3T), (256,), device=dev)
        s, a = (S3T[i] - MU3) / SD3, A3T[i]
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
            loss = ((net(torch.cat([at / asd, t, s], 1)) - (a - z))
                    ** 2).mean()
        opt.zero_grad(); loss.backward(); opt.step()

    def f(s):
        sn = (s - MU3) / SD3
        if kind == "L2":
            return net(sn)
        if kind == "MIP":
            z = torch.randn(s.shape[0], 3, device=dev)
            y0 = net(torch.cat([z, sn], 1))
            return net(torch.cat([y0 / asd, sn], 1))
        a = torch.randn(s.shape[0], 3, device=dev) * asd
        for k in range(10):
            t = torch.full((s.shape[0], 1), k / 10.0, device=dev)
            a = a + 0.1 * net(torch.cat([a / asd, t, sn], 1))
        return a
    return f


def roll3(f, g, nmax=420):
    p = np.zeros(3)
    n = 0
    while (n < nmax and p[2] > -38 and abs(p[1]) < 80 and -20 < p[0] < 200
           and p[2] < 30):
        s = torch.tensor([[*p, g]], dtype=torch.float32, device=dev)
        with torch.no_grad():
            a = f(s)[0].cpu().numpy()
        p = p + a
        n += 1
    return p[2] <= -38 and np.linalg.norm(p[:2] - S3[:2]) < 4.0


print("== Design 2b: 3D toy, h=64, training-budget sweep (clean, "
      "training anchors) ==")
res2b = {}
D2B = [] if os.environ.get("SKIP_D2B") else [3000, 5000, 7000, 9000, 12000]
for st in D2B:
    for name in ["L2", "MIP", "Flow"]:
        srs = []
        for sd in SEEDS:
            f = train3(name, 64, st, sd)
            srs.append(np.mean([roll3(f, g)
                                for g in np.linspace(-15, 15, K_DEMO)]))
        res2b[(name, st)] = np.mean(srs)
        print(f"  steps={st:5d} {name:5s} SR {np.mean(srs):.2f}", flush=True)

import pickle
with open(os.environ.get("OUT2", "analysis/toy2d_nfl2.pkl"), "wb") as fh:
    pickle.dump({"res1b": res1b, "spread": spread, "res2b": res2b}, fh)
print("SAVED analysis/toy2d_nfl2.pkl")
