"""Collapse-note toy (pre-reg PART CCCXIV): miniaturize the funding-collapse
causal chain. Clean deterministic data. approach -> grasp window (K_HOLD=6 <
H=8; latch g + seat micro-motion encode the schedule redundantly; |x|>XTOL
while 0<g<1 breaks the grasp) -> stroke to B through the same height band.
Cells DISTR [x,y,g,d1..8] / CUR [x,y,g]. Arms l2/hg/ht/mip.
Instruments: quiet/loud PR at snapshots, settle grad-share trajectory,
basin sweep, kicked kNN stroke-fraction, settle column gains."""
import copy
import json
import os

import numpy as np
import torch
import torch.nn as nn

DEV = "cuda" if torch.cuda.is_available() else "cpu"
A = np.array([0.0, 0.0])
B = np.array([0.6, 0.5])
H = 8
K_HOLD = 6
SEAT = 0.03
XTOL = 0.03
GAIN, CLIP = 0.3, 0.08
NDIST = int(os.environ.get("NDIST", "8"))
NEP = int(os.environ.get("NEP", "150"))
STEPS = int(os.environ.get("STEPS", "40000"))
WIDTH = int(os.environ.get("WIDTH", "128"))
SEEDS = [int(s) for s in os.environ.get("SEEDS", "0,1").split(",")]
SNAPS = (5000, 20000)


def expert_action(p, g):
    if g >= 1.0:
        tgt = B
    elif g > 0.0:
        tgt = A + np.array([0.0, -SEAT * g])
    else:
        tgt = A
    return np.clip(GAIN * (tgt - p), -CLIP, CLIP)


def step_g(p, g):
    if g < 1.0 and np.linalg.norm(p - A) < 0.06:
        return min(1.0, g + 1.0 / K_HOLD)
    return g


def gen_episode(rng):
    h = 0.3 + 0.5 * rng.rand(NDIST)
    p = np.array([rng.uniform(-0.06, 0.06), rng.uniform(0.5, 1.0)])
    g = 0.0
    states, tail = [], H + 3
    for _ in range(120):
        a = expert_action(p, g)
        states.append((p.copy(), g, a.copy()))
        p = p + a
        g = step_g(p, g)
        if g >= 1.0 and np.linalg.norm(p - B) < 0.05:
            tail -= 1
            if tail <= 0:
                break
    return states, h


def phase_of(g, p):
    if g <= 0.0:
        return "approach"
    return "settle" if g < 1.0 else "stroke"


def build_obs(p, g, h, cell):
    if cell == "CUR":
        return np.array([p[0], p[1], g], np.float32)
    return np.concatenate([[p[0], p[1], g], h - p[1]]).astype(np.float32)


def build_dataset(cell, rng):
    X, Y, PH, YY = [], [], [], []
    for _ in range(NEP):
        states, h = gen_episode(rng)
        obs = [build_obs(p, g, h, cell) for p, g, a in states]
        for t in range(len(states) - H):
            X.append(obs[t])
            Y.append(np.concatenate([states[t + j][2] for j in range(H)]))
            PH.append(phase_of(states[t][1], states[t][0]))
            YY.append(states[t][0][1])
    return (np.array(X, np.float32), np.array(Y, np.float32), np.array(PH),
            np.array(YY, np.float32))


class Reg(nn.Module):
    def __init__(self, din, hetero=False):
        super().__init__()
        self.hetero = hetero
        self.trunk = nn.Sequential(nn.Linear(din, WIDTH), nn.ReLU(),
                                   nn.Linear(WIDTH, WIDTH), nn.ReLU())
        self.head = nn.Linear(WIDTH, 4 * H if hetero else 2 * H)

    def feats(self, xn):
        return self.trunk(xn)

    def full(self, xn):
        return self.head(self.trunk(xn))

    def forward(self, xn):
        out = self.head(self.trunk(xn))
        return out[:, :2 * H] if self.hetero else out


class MipNet(nn.Module):
    def __init__(self, din):
        super().__init__()
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
        for j in range(2):
            t = torch.full((len(xn), 1), j / 2, device=xn.device)
            y = y + 0.5 * self.vel(xn, y, t)
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


def loss_fn(arm, net, xb, yb, qn_mask=None):
    if arm == "l2qn":
        yb = yb + 0.1 * qn_mask * torch.randn_like(yb)
        return ((net(xb) - yb) ** 2).mean()
    if arm in ("mip", "mipng"):
        eps = torch.randn_like(yb)
        t = torch.rand(len(yb), 1, device=DEV)
        yt = (1 - t) * eps + t * yb
        return ((net.vel(xb, yt, t) - (yb - eps)) ** 2).mean()
    if arm in ("hg", "ht"):
        out = net.full(xb)
        mu, lv = out[:, :2 * H], out[:, 2 * H:].clamp(-10, 2)
        r2 = (yb - mu) ** 2 * torch.exp(-lv)
        if arm == "hg":
            return (0.5 * (lv + r2)).mean()
        return (1.5 * torch.log1p(r2 / 2.0) + 0.5 * lv).mean()
    return ((net(xb) - yb) ** 2).mean()


def train(arm, seed, Xn, Yn, din, PH=None):
    torch.manual_seed(seed)
    net = (MipNet(din) if arm in ("mip", "mipng")
           else Reg(din, hetero=arm in ("hg", "ht"))).to(DEV)
    opt = torch.optim.Adam(net.parameters(), lr=1e-3)
    Xt = torch.tensor(Xn, device=DEV)
    Yt = torch.tensor(Yn, device=DEV)
    rng = np.random.RandomState(seed)
    settle = torch.tensor((PH == "settle").astype(np.float32)[:, None],
                          device=DEV) if PH is not None else None
    snaps = {}
    for it in range(STEPS):
        idx = rng.randint(0, len(Xt), 256)
        loss = loss_fn(arm, net, Xt[idx], Yt[idx],
                       settle[idx] if settle is not None else None)
        opt.zero_grad()
        loss.backward()
        opt.step()
        if it + 1 in SNAPS:
            snaps[it + 1] = copy.deepcopy(net)
    snaps[STEPS] = net
    return snaps


def pr_of(net, Xn_states):
    prs = []
    for i in range(min(20, len(Xn_states))):
        xi = torch.tensor(Xn_states[i:i + 1], device=DEV)
        Jf = torch.autograd.functional.jacobian(
            lambda v: net.feats(v), xi, vectorize=True)[0, :, 0, :]
        sv = torch.linalg.svdvals(Jf) ** 2
        prs.append(((sv.sum() ** 2) / (sv ** 2).sum()).item())
    return float(np.mean(prs))


def grad_share(arm, net, norm, Xn, Yn, PH):
    """settle-sample share of per-sample gradient weight (residual-based)."""
    with torch.no_grad():
        Xt = torch.tensor(Xn, device=DEV)
        Yt = torch.tensor(Yn, device=DEV)
        if arm in ("hg", "ht"):
            out = net.full(Xt)
            mu, lv = out[:, :2 * H], out[:, 2 * H:].clamp(-10, 2)
            w = ((Yt - mu).abs() * torch.exp(-lv)).mean(1)
        elif arm == "mip":
            w = (net(Xt) - Yt).abs().mean(1)
        else:
            w = (net(Xt) - Yt).abs().mean(1)
        w = (w / (w.mean() + 1e-12)).cpu().numpy()
    return float(w[PH == "settle"].mean())


def col_shares(net, Xn_settle, cell):
    xg = torch.tensor(Xn_settle, device=DEV, requires_grad=True)
    F = net.feats(xg)
    g = torch.autograd.grad(F.pow(2).sum(), xg)[0].pow(2).sum(0)
    g = (g / g.sum()).detach().cpu().numpy()
    return dict(x=float(g[0]), y=float(g[1]), g=float(g[2]),
                d=float(g[3:].sum()) if cell == "DISTR" else 0.0)


def basin(net, norm, Xn, PH, YY):
    """interp settle anchor -> matched-y stroke obs; alpha where |a_pos|
    exceeds half the stroke magnitude."""
    si = np.where(PH == "settle")[0]
    ki = np.where(PH == "stroke")[0]
    snaps = []
    for i in si[np.linspace(0, len(si) - 1, 12).astype(int)]:
        j = ki[np.argmin(np.abs(YY[ki] - YY[i]))]
        o0, o1 = Xn[i], Xn[j]
        with torch.no_grad():
            mags = []
            for al in np.linspace(0, 1, 11):
                on = torch.tensor(((1 - al) * o0 + al * o1)[None], device=DEV)
                a = norm.uy(net(on)[0].cpu().numpy()).reshape(H, 2)[0]
                mags.append(np.linalg.norm(a))
        mags = np.array(mags)
        ref = mags[-1]
        above = np.where(mags > 0.5 * ref)[0]
        snaps.append(above[0] / 10 if len(above) else 1.0)
    return float(np.mean(snaps))


def knn_stroke(net, norm, Xn, PH, kick=0.03):
    with torch.no_grad():
        Fb = net.feats(torch.tensor(Xn, device=DEV)).cpu().numpy()
    si = np.where(PH == "settle")[0]
    qs = []
    sc = 2 / (norm.xhi[1] - norm.xlo[1] + 1e-8)
    for i in si[np.linspace(0, len(si) - 1, 30).astype(int)]:
        q = Xn[i].copy()
        q[1] += kick * sc  # pure +y kick in normalized units
        qs.append(q)
    with torch.no_grad():
        Fq = net.feats(torch.tensor(np.array(qs, np.float32), device=DEV)).cpu().numpy()
    fr = []
    for f in Fq:
        nn10 = np.argsort(((Fb - f) ** 2).sum(1))[:10]
        fr.append(np.mean(PH[nn10] == "stroke"))
    return float(np.mean(fr))


def rollout(net, norm, cell, ep_rng, kick=0.0, AS=8, expert=False, arm=""):
    h = 0.3 + 0.5 * ep_rng.rand(NDIST)
    p = np.array([ep_rng.uniform(-0.06, 0.06), ep_rng.uniform(0.5, 1.0)])
    g = 0.0
    kick_at = ep_rng.uniform(0.2, 0.8)
    kicked = False
    budget = 200
    while budget > 0:
        if expert:
            chunk = np.array([expert_action(p, g)])
        else:
            o = norm.nx(build_obs(p, g, h, cell)[None])
            if arm.endswith("ng"):
                o[:, 2] = 0.0
            on = torch.tensor(o, device=DEV)
            with torch.no_grad():
                chunk = norm.uy(net(on)[0].cpu().numpy()).reshape(H, 2)
        for a in chunk[:AS]:
            if kick > 0 and not kicked and kick_at <= g < 1.0:
                p = p + np.array([0.0, kick * ep_rng.choice([-1.0, 1.0])])
                kicked = True
                break  # replan on the kicked state
            p = p + np.clip(a, -CLIP, CLIP)
            g = step_g(p, g)
            budget -= 1
            if 0.0 < g < 1.0 and abs(p[0] - A[0]) > XTOL:
                return "break"
            if g >= 1.0 and np.linalg.norm(p - B) < 0.05:
                return "success"
            if budget <= 0:
                break
    return "timeout"


def evaluate(net, norm, cell, kick, AS, n=80, expert=False, arm=""):
    rng = np.random.RandomState(9000 + int(kick * 1000) + AS)
    outs = [rollout(net, norm, cell, rng, kick, AS, expert, arm) for _ in range(n)]
    return {o: outs.count(o) / len(outs) for o in ("success", "break", "timeout")}


def main():
    results = {}
    if os.path.exists("toycollapse.json"):
        results.update(json.load(open("toycollapse.json")))
    for cell in tuple(os.environ.get("CELLS", "DISTR,CUR").split(",")):
        drng = np.random.RandomState(0)
        X, Y, PH, YY = build_dataset(cell, drng)
        norm = Norm(X, Y)
        Xn, Yn = norm.nx(X), norm.ny(Y)
        si = np.where(PH == "settle")[0][:200]
        li = np.where(PH == "stroke")[0][:200]
        print(f"[{cell}] {len(X)} samples | "
              f"{[(ph, int((PH == ph).sum())) for ph in ('approach', 'settle', 'stroke')]}",
              flush=True)
        e = evaluate(None, None, cell, 0.04, 8, expert=True)
        print(f"[{cell}] expert kicked(.04): {e}", flush=True)
        arms = tuple(os.environ.get("ARMS", "l2,hg,ht,mip").split(","))
        for arm in arms:
            Xa = Xn.copy()
            if arm.endswith("ng"):
                Xa[:, 2] = 0.0  # latch column masked (height-only chart)
            for seed in SEEDS:
                snaps = train(arm, seed, Xa, Yn, X.shape[1], PH)
                net = snaps[STEPS]
                r = {}
                for st, sn in snaps.items():
                    r[f"prq_{st}"] = pr_of(sn, Xa[si])
                    r[f"prl_{st}"] = pr_of(sn, Xa[li])
                    r[f"gsh_{st}"] = grad_share(arm, sn, norm, Xa, Yn, PH)
                r.update({f"cg_{k}": v for k, v in
                          col_shares(net, Xa[si[:64]], cell).items()})
                r["basin"] = basin(net, norm, Xa, PH, YY)
                r["knnS"] = knn_stroke(net, norm, Xa, PH)
                for kick in (0.0, 0.02, 0.04):
                    for AS in (8, 1):
                        ev = evaluate(net, norm, cell, kick, AS, arm=arm)
                        r[f"sr_k{kick}_as{AS}"] = ev["success"]
                        r[f"br_k{kick}_as{AS}"] = ev["break"]
                results[f"{cell}_{arm}_s{seed}"] = r
                print(f"[{cell}] {arm} s{seed}: SR(as8) "
                      f"{r['sr_k0.0_as8']:.2f}/{r['sr_k0.02_as8']:.2f}/{r['sr_k0.04_as8']:.2f} "
                      f"br(.04) {r['br_k0.04_as8']:.2f} | as1(.04) {r['sr_k0.04_as1']:.2f} | "
                      f"PRq/l@end {r[f'prq_{STEPS}']:.1f}/{r[f'prl_{STEPS}']:.1f} "
                      f"gshS {r[f'gsh_{min(snaps)}']:.2f}->{r[f'gsh_{STEPS}']:.2f} | "
                      f"basin {r['basin']:.2f} knnS {r['knnS']:.2f} | "
                      f"cg y+d {r['cg_y'] + r['cg_d']:.2f} g {r['cg_g']:.2f}",
                      flush=True)
    json.dump(results, open("toycollapse.json", "w"), indent=1)
    print("wrote toycollapse.json", flush=True)


if __name__ == "__main__":
    main()
