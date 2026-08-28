"""Phase/distractor toy — script-data-faithful (pre-reg: PART CCCVI).

approach (descend to A) -> settle (hold; progress latch c: 0->1 over K
steps while near A) -> transit (diagonal to B, passes settle's height
band). Success intrinsic: c>=1 and reach B. Obs cells: DISTR adds 8
redundant height encodings d_j = h_j - y (base_relz/tool_relz analog);
CUR = [x, y, c] only. Eval: impulse kick at a random mid-hold step.
Arms l2 / mip(2-step) / flow8 share the FlowNet trunk for generative.
Instruments: per-phase feature-Jacobian PR, settle column-gain shares,
kicked-state kNN phase composition, transit-cos of post-kick action.
"""
import json
import os

import numpy as np
import torch
import torch.nn as nn

DEV = "cuda" if torch.cuda.is_available() else "cpu"
A = np.array([0.0, 0.0])
B = np.array([0.5, 0.5])
K_HOLD = int(os.environ.get("KHOLD", "15"))
NDIST = int(os.environ.get("NDIST", "8"))
DNOISE = float(os.environ.get("DNOISE", "0.005"))
ONOISE = float(os.environ.get("ONOISE", "0.005"))
EXNOISE = float(os.environ.get("EXNOISE", "0.01"))
GSAT = float(os.environ.get("GSAT", "0.33"))
SEAT = float(os.environ.get("SEAT", "0.03"))
H = 8
STEPS = int(os.environ.get("STEPS", "20000"))
WIDTH = int(os.environ.get("WIDTH", "64"))
NEP = int(os.environ.get("NEP", "200"))
SEEDS = [int(s) for s in os.environ.get("SEEDS", "0,1,2").split(",")]
EPLEN = 140
GAIN, CLIP, HOLD_R = 0.25, 0.10, 0.06


def expert_action(p, c):
    # during the hold the tool is SEATED: target creeps down by SEAT over the wait
    tgt = (A + np.array([0.0, -SEAT * min(c, 1.0)])) if c < 1.0 else B
    return np.clip(GAIN * (tgt - p), -CLIP, CLIP)


def step_c(p, c):
    if c < 1.0 and np.linalg.norm(p - A) < HOLD_R:
        return min(1.0, c + 1.0 / K_HOLD)
    return c


def gen_episode(rng):
    h = 0.3 + 0.4 * rng.rand(NDIST) + rng.uniform(-0.05, 0.05, NDIST)
    p = np.array([rng.uniform(-0.1, 0.1), rng.uniform(0.7, 0.9)])
    c = 0.0
    states, acts = [], []
    tail = H + 4  # post-success hold-at-B frames so the H-cut doesn't starve transit
    for _ in range(EPLEN):
        a = expert_action(p, c)
        states.append(np.concatenate([p, [c], h]))
        acts.append(a)
        p = p + a + EXNOISE * rng.randn(2)
        c = step_c(p, c)
        if c >= 1.0 and np.linalg.norm(p - B) < 0.05:
            tail -= 1
            if tail <= 0:
                break
    return np.array(states), np.array(acts)


def phase_of(state):
    c = state[2]
    if c <= 0.0:
        return "approach"
    return "settle" if c < 1.0 else "transit"


def build_obs(states, cell, rng):
    p, c, h = states[:, :2], states[:, 2:3], states[:, 3:]
    g = np.minimum(c / GSAT, 1.0)  # saturating latch obs (grip-qpos analog)
    if cell == "CUR":
        o = np.concatenate([p, g], 1)
    else:
        d = h - p[:, 1:2]
        o = np.concatenate([p, g, d], 1)
    return (o + ONOISE * rng.randn(*o.shape)).astype(np.float32)


def build_dataset(cell, rng):
    X, Y, PH = [], [], []
    for _ in range(NEP):
        st, ac = gen_episode(rng)
        obs = build_obs(st, cell, rng)
        for t in range(len(st) - H):
            X.append(obs[t])
            Y.append(ac[t:t + H].reshape(-1))
            PH.append(phase_of(st[t]))
    return np.array(X, np.float32), np.array(Y, np.float32), np.array(PH)


class Reg(nn.Module):
    def __init__(self, din):
        super().__init__()
        self.trunk = nn.Sequential(nn.Linear(din, WIDTH), nn.ReLU(),
                                   nn.Linear(WIDTH, WIDTH), nn.ReLU())
        self.head = nn.Linear(WIDTH, 2 * H)

    def feats(self, xn):
        return self.trunk(xn)

    def forward(self, xn):
        return self.head(self.trunk(xn))


class FlowNet(nn.Module):
    def __init__(self, din, nsteps):
        super().__init__()
        self.nsteps = nsteps
        self.trunk = nn.Sequential(nn.Linear(din + 2 * H + 1, WIDTH), nn.ReLU(),
                                   nn.Linear(WIDTH, WIDTH), nn.ReLU())
        self.head = nn.Linear(WIDTH, 2 * H)

    def vel(self, xn, yt, t):
        return self.head(self.trunk(torch.cat([xn, yt, t], 1)))

    def feats(self, xn):
        z = torch.zeros(len(xn), 2 * H + 1, device=xn.device)
        return self.trunk(torch.cat([xn, z], 1))

    def forward(self, xn):
        y = torch.zeros(len(xn), 2 * H, device=xn.device)
        for k in range(self.nsteps):
            t = torch.full((len(xn), 1), k / self.nsteps, device=xn.device)
            y = y + (1.0 / self.nsteps) * self.vel(xn, y, t)
        return y


class Norm:
    def __init__(self, X, Y):
        self.xlo, self.xhi = X.min(0), X.max(0)
        self.ylo, self.yhi = Y.min(0), Y.max(0)

    def nx(self, x):
        return (2 * (x - self.xlo) / (self.xhi - self.xlo + 1e-8) - 1).astype(np.float32)

    def ny(self, y):
        return (2 * (y - self.ylo) / (self.yhi - self.ylo + 1e-8) - 1).astype(np.float32)

    def uy(self, yn):
        return (yn + 1) / 2 * (self.yhi - self.ylo + 1e-8) + self.ylo


def train(arm, seed, Xn, Yn, din):
    torch.manual_seed(seed)
    net = (Reg(din) if arm == "l2" else FlowNet(din, 2 if arm == "mip" else 8)).to(DEV)
    opt = torch.optim.Adam(net.parameters(), lr=1e-3)
    Xt = torch.tensor(Xn, device=DEV)
    Yt = torch.tensor(Yn, device=DEV)
    rng = np.random.RandomState(seed)
    for _ in range(STEPS):
        idx = rng.randint(0, len(Xt), 256)
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


def policy_chunk(net, norm, state, cell, rng):
    o = build_obs(state[None], cell, rng)
    on = torch.tensor(norm.nx(o), device=DEV)
    with torch.no_grad():
        yn = net(on)[0].cpu().numpy()
    return norm.uy(yn).reshape(H, 2)


def rollout(net, norm, cell, rng, kick=0.0, expert=False):
    h = 0.3 + 0.4 * rng.rand(NDIST) + rng.uniform(-0.05, 0.05, NDIST)
    p = np.array([rng.uniform(-0.1, 0.1), rng.uniform(0.7, 0.9)])
    c = 0.0
    kick_at_c = rng.uniform(0.2, 0.8)
    kicked = False
    tcos, exit_hold = np.nan, False
    budget = 320  # actions; nominal episode ~140
    while budget > 0:
        state = np.concatenate([p, [c], h])
        if expert:
            chunk = np.array([expert_action(p, c)])
        else:
            chunk = policy_chunk(net, norm, state, cell, rng)
        for j, a in enumerate(chunk):
            if kick > 0 and not kicked and 0 < c >= kick_at_c and c < 1.0:
                th = rng.uniform(0, 2 * np.pi)
                p = p + kick * np.array([np.cos(th), np.sin(th)])
                kicked = True
                if not expert:
                    st2 = np.concatenate([p, [c], h])
                    a2 = policy_chunk(net, norm, st2, cell, rng)[0]
                    tdir = (B - A) / np.linalg.norm(B - A)
                    n = np.linalg.norm(a2)
                    tcos = float(a2 @ tdir / n) if n > 1e-6 else 0.0
                break  # replan on the kicked state
            a = np.clip(a, -CLIP, CLIP)
            p = p + a
            c = step_c(p, c)
            budget -= 1
            if 0 < c < 1.0 and np.linalg.norm(p - A) > 0.15:
                exit_hold = True
            if c >= 1.0 and np.linalg.norm(p - B) < 0.05:
                return True, tcos, exit_hold
            if budget <= 0:
                break
    return False, tcos, exit_hold


def evaluate(net, norm, cell, kick, n_ep=50, seed=7777, expert=False):
    rng = np.random.RandomState(seed + int(kick * 1000))
    res = [rollout(net, norm, cell, rng, kick, expert) for _ in range(n_ep)]
    sr = float(np.mean([r[0] for r in res]))
    tc = [r[1] for r in res if np.isfinite(r[1])]
    ex = float(np.mean([r[2] for r in res]))
    return sr, (float(np.mean(tc)) if tc else np.nan), ex


def probe(net, norm, Xn, PH, cell):
    out = {}
    for ph in ("approach", "settle", "transit"):
        sel = np.where(PH == ph)[0]
        idx = sel[np.linspace(0, len(sel) - 1, 20).astype(int)]
        prs = []
        for i in idx:
            xi = torch.tensor(Xn[i:i + 1], device=DEV)
            Jf = torch.autograd.functional.jacobian(
                lambda v: net.feats(v), xi, vectorize=True)[0, :, 0, :]
            sv = torch.linalg.svdvals(Jf) ** 2
            prs.append(((sv.sum() ** 2) / (sv ** 2).sum()).item())
        out[f"PR_{ph}"] = float(np.mean(prs))
        if ph == "settle":
            xg = torch.tensor(Xn[idx], device=DEV, requires_grad=True)
            F = net.feats(xg)
            gsum = torch.autograd.grad(F.pow(2).sum(), xg)[0].pow(2).sum(0)
            g = gsum.detach().cpu().numpy()
            tot = g.sum()
            out["gain_x"] = float(g[0] / tot)
            out["gain_y"] = float(g[1] / tot)
            out["gain_c"] = float(g[2] / tot)
            out["gain_dist"] = float(g[3:].sum() / tot) if cell == "DISTR" else 0.0
    return out


def knn_phase(net, norm, Xn, PH, cell, rng):
    """feature-kNN phase composition of kicked settle states."""
    with torch.no_grad():
        Fb = net.feats(torch.tensor(Xn, device=DEV)).cpu().numpy()
    sel = np.where(PH == "settle")[0]
    qs = []
    for i in sel[np.linspace(0, len(sel) - 1, 40).astype(int)]:
        x = Xn[i].copy()
        # kick in raw obs space: perturb x,y by 0.04 (normalized via scale)
        sc = 2 / (norm.xhi[:2] - norm.xlo[:2] + 1e-8)
        th = rng.uniform(0, 2 * np.pi)
        x[:2] += 0.04 * np.array([np.cos(th), np.sin(th)]) * sc
        qs.append(x)
    with torch.no_grad():
        Fq = net.feats(torch.tensor(np.array(qs, np.float32), device=DEV)).cpu().numpy()
    frac_transit = []
    for f in Fq:
        d = ((Fb - f) ** 2).sum(1)
        nn10 = np.argsort(d)[:10]
        frac_transit.append(np.mean(PH[nn10] == "transit"))
    return float(np.mean(frac_transit))


def main():
    results = {}
    cells = tuple(os.environ.get("CELLS", "DISTR,CUR").split(","))
    for cell in cells:
        drng = np.random.RandomState(0)
        X, Y, PH = build_dataset(cell, drng)
        norm = Norm(X, Y)
        Xn, Yn = norm.nx(X), norm.ny(Y)
        print(f"[{cell}] dataset {len(X)} samples | phases: "
              f"{[(p, int((PH == p).sum())) for p in ('approach', 'settle', 'transit')]}",
              flush=True)
        e_sr = [evaluate(None, norm, cell, k, expert=True)[0]
                for k in (0.0, 0.02, 0.04, 0.06)]
        print(f"[{cell}] EXPERT anchor SR clean/.02/.04/.06 = "
              + "/".join(f"{s:.2f}" for s in e_sr), flush=True)
        for arm in ("l2", "mip", "flow8"):
            rows = []
            for seed in SEEDS:
                net = train(arm, seed, Xn, Yn, X.shape[1])
                r = {}
                for k in (0.0, 0.02, 0.04, 0.06):
                    sr, tc, ex = evaluate(net, norm, cell, k)
                    r[f"sr{k}"] = sr
                    if k == 0.04:
                        r["tcos"], r["exit"] = tc, ex
                r.update(probe(net, norm, Xn, PH, cell))
                r["knn_transit"] = knn_phase(net, norm, Xn, PH, cell,
                                             np.random.RandomState(3))
                rows.append(r)
                print(f"[{cell}] {arm} s{seed}: SR {r['sr0.0']:.2f}/"
                      f"{r['sr0.02']:.2f}/{r['sr0.04']:.2f}/{r['sr0.06']:.2f} "
                      f"tcos {r['tcos']:.2f} exit {r['exit']:.2f} | "
                      f"PR a/s/t {r['PR_approach']:.1f}/{r['PR_settle']:.1f}/"
                      f"{r['PR_transit']:.1f} | settle gains x/y/c/d "
                      f"{r['gain_x']:.2f}/{r['gain_y']:.2f}/{r['gain_c']:.2f}/"
                      f"{r['gain_dist']:.2f} | knnT {r['knn_transit']:.2f}",
                      flush=True)
            results[f"{cell}_{arm}"] = rows
        results[f"{cell}_expert"] = e_sr

    print("\n=== TABLE (mean over seeds) ===", flush=True)
    print(f"{'cell/arm':>13} | {'clean':>5} {'k.02':>5} {'k.04':>5} {'k.06':>5} "
          f"| {'tcos':>5} {'knnT':>5} | {'PRset':>5} {'d-share':>7}", flush=True)
    for cell in cells:
        e = results[f"{cell}_expert"]
        print(f"{cell + '/expert':>13} | " + " ".join(f"{s:5.2f}" for s in e)
              + " |     -     - |     -       -", flush=True)
        for arm in ("l2", "mip", "flow8"):
            rows = results[f"{cell}_{arm}"]
            m = {k: np.mean([r[k] for r in rows]) for k in rows[0]}
            print(f"{cell + '/' + arm:>13} | {m['sr0.0']:5.2f} {m['sr0.02']:5.2f} "
                  f"{m['sr0.04']:5.2f} {m['sr0.06']:5.2f} | {m['tcos']:5.2f} "
                  f"{m['knn_transit']:5.2f} | {m['PR_settle']:5.1f} "
                  f"{m['gain_dist']:7.2f}", flush=True)
    json.dump(results, open("toyphase.json", "w"), indent=1, default=float)
    print("wrote toyphase.json", flush=True)


if __name__ == "__main__":
    main()
