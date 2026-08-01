"""HARDER TOY: 2D insertion, 64-dim obs, condition-dependent plan, weak recovery
residual (lambda_rec), narrow tube, action chunks (H=8), closed-loop insertion SR.
Models: MSE, MIP (t=0.9 aux w=100), rectified FLOW (10-step), PDS (MSE + PD-chunk reg).
Regimes: easy (50k, sig .03, lam .2) [A only], hard (5k, .01, .1) [A/B/C],
veryhard (2k, .005, .05) [A only]. Outputs TOYINS-prefixed tables + figures.
"""
import numpy as np, torch, torch.nn as nn, os, json
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

torch.manual_seed(0); np.random.seed(0)
dev = "cuda" if torch.cuda.is_available() else "cpu"
OUT = "analysis/toyins"; os.makedirs(OUT, exist_ok=True)
H = 8; DP = 0.02

# ---------- plan geometry ----------
def xhole(xi): return 0.25 * xi[..., 0]
def phi(xi): return np.pi * xi[..., 1]
def env_(p):
    # 1 for p<0.5, smooth cosine decay to 0 at p>=0.85 (nominal converges to hole
    # BEFORE the slot narrows -- otherwise the expert itself collides)
    u = np.clip((p - 0.5) / 0.35, 0, 1)
    return 0.5 * (1 + np.cos(np.pi * u))
def denv_(p):
    u = np.clip((p - 0.5) / 0.35, 0, 1)
    return np.where((p > 0.5) & (p < 0.85), -0.5 * np.pi * np.sin(np.pi * u) / 0.35, 0.0)
def wave(p, xi): return 0.15 * np.sin(2 * np.pi * p + phi(xi)) + 0.05 * np.sin(6 * np.pi * p)
def dwave(p, xi):
    return 0.15 * 2 * np.pi * np.cos(2 * np.pi * p + phi(xi)) + 0.05 * 6 * np.pi * np.cos(6 * np.pi * p)
def cx(p, xi): return xhole(xi) + env_(p) * wave(p, xi)
def cline(p, xi): return np.stack([cx(p, xi), p], -1)
def cdot(p, xi):
    dx = denv_(p) * wave(p, xi) + env_(p) * dwave(p, xi)
    return np.stack([dx, np.ones_like(p)], -1)
def TN(p, xi):
    d = cdot(p, xi); T = d / np.linalg.norm(d, axis=-1, keepdims=True)
    return T, np.stack([-T[..., 1], T[..., 0]], -1)
def vt(p): return 0.025 * (1 + 0.2 * np.cos(2 * np.pi * p))
def kA(p): return 0.25 * np.ones_like(p)
def kC(p):
    k = 0.25 * np.tanh(5 * np.sin(6 * np.pi * p))
    return np.where(p > 0.85, 0.35, k)
def gB(p): return 1 / (1 + np.exp(-(p - 0.85) / 0.03))
def kdata(variant, p):
    return {"A": kA, "C": kC}.get(variant, lambda q: gB(q) * 0.35)(p)
def vn(variant, p, n, rng):
    if variant == "B":
        return -gB(p) * 0.35 * n + (1 - gB(p)) * rng.randn(*np.shape(p)) * 0.35 * 0.01
    return -kdata(variant, p) * n

def chunk(variant, p, n, xi, lam, rng):
    # (B,H,2) expert chunk; current deviation n applied at each h
    outs = []
    for h in range(H):
        ph = np.clip(p + h * DP, 0, 1)
        T, N = TN(ph, xi)
        a = vt(ph)[:, None] * T + lam * vn(variant, ph, n, rng)[:, None] * N
        outs.append(a)
    return np.stack(outs, 1)

# ---------- observation (64-d) ----------
rngW = np.random.RandomState(99)
W1 = rngW.randn(3, 32) * 1.5; B1 = rngW.randn(32)
OFFS = rngW.randn(4, 2) * 0.05
def rot(th):
    c, s = np.cos(th), np.sin(th)
    return np.stack([np.stack([c, -s], -1), np.stack([s, c], -1)], -2)
def make_obs(q, xi, n, p, rng):
    B = len(q)
    R = rot(0.5 * xi[:, 1])
    kps = (q[:, None, :] + np.einsum('bij,kj->bki', R, OFFS)).reshape(B, 8)
    rel = q - np.stack([xhole(xi), np.ones(B)], -1)
    z = q[:, 1]
    four = np.concatenate([np.stack([np.sin(k * np.pi * z), np.cos(k * np.pi * z)], -1)
                           for k in range(1, 5)], -1) + 0.05 * rng.randn(B, 8)
    ncode = (n + 0.02 * rng.randn(B))[:, None]
    nuis = np.tanh(np.stack([xi[:, 0], xi[:, 1], p], -1) @ W1 + B1) + 0.1 * rng.randn(B, 32)
    pad = 0.1 * rng.randn(B, 9)
    return np.concatenate([q, xi, kps, rel, four, ncode, nuis, pad], -1).astype(np.float32)

def gen(variant, ns, sig, lam, rng):
    xi = rng.uniform(-1, 1, (ns, 2))
    p = rng.rand(ns)
    n = np.clip(rng.randn(ns) * sig, -3 * sig, 3 * sig)
    T, N = TN(p, xi)
    q = cline(p, xi) + n[:, None] * N
    o = make_obs(q, xi, n, p, rng)
    A = chunk(variant, p, n, xi, lam, rng)
    return o, A.reshape(ns, -1).astype(np.float32), q, xi, p, n

# ---------- models ----------
class MLP(nn.Module):
    def __init__(self, din, dout, w=256, depth=4):
        super().__init__()
        L = []; d = din
        for _ in range(depth): L += [nn.Linear(d, w), nn.SiLU()]; d = w
        L += [nn.Linear(d, dout)]
        self.net = nn.Sequential(*L)
    def forward(self, x): return self.net(x)

def train_model(kind, O, A, variant, sig, lam, steps=25000, bs=512, snap_at=None, pd_k=0.025):
    a_std = float(A.std()); An = A / a_std
    O_t = torch.tensor(O, device=dev); A_t = torch.tensor(An, device=dev)
    dout = 2 * H
    din = {"mse": 64, "mip": 64 + dout + 1, "flow": 64 + dout + 1, "pds": 64}[kind]
    torch.manual_seed(0)
    net = MLP(din, dout).to(dev)
    opt = torch.optim.Adam(net.parameters(), lr=1e-3)
    sch = torch.optim.lr_scheduler.CosineAnnealingLR(opt, steps)
    snaps = {}
    # pds precomputes paired data lazily per batch (needs geometry regen)
    for it in range(1, steps + 1):
        idx = torch.randint(0, len(O_t), (bs,), device=dev)
        o, a = O_t[idx], A_t[idx]
        if kind in ("mse", "pds"):
            loss = ((net(o) - a) ** 2).mean()
            if kind == "pds":
                # PD-chunk reg on synthetic normal pushes (fresh states each batch)
                rngp = np.random.RandomState(it)
                B2 = 128
                xi = rngp.uniform(-1, 1, (B2, 2)); p = rngp.rand(B2)
                nn_ = np.zeros(B2)
                T, N = TN(p, xi)
                q0 = cline(p, xi)
                dn = rngp.uniform(1, 4, B2) * sig * np.where(rngp.rand(B2) > .5, 1, -1)
                noise = np.random.RandomState(it + 1)
                # same noise pack for both obs
                st = noise.get_state()
                o0 = make_obs(q0, xi, nn_, p, np.random.RandomState(it + 1))
                o1 = make_obs(q0 + dn[:, None] * N, xi, dn, p, np.random.RandomState(it + 1))
                f0 = net(torch.tensor(o0, device=dev)); f1 = net(torch.tensor(o1, device=dev))
                dfa = (f1 - f0).reshape(B2, H, 2) * a_std
                Nt = torch.tensor(N, dtype=torch.float32, device=dev)
                dn_t = torch.tensor(dn, dtype=torch.float32, device=dev)
                dnorm = torch.einsum('bhi,bi->bh', dfa, Nt)
                tgt = (-pd_k * dn_t)[:, None].expand(-1, H)
                loss = loss + 100 * ((dnorm - tgt) ** 2).mean() / a_std ** 2
        elif kind == "mip":
            z0 = torch.zeros(bs, dout, device=dev)
            main = ((net(torch.cat([o, z0, z0[:, :1]], 1)) - a) ** 2).mean()
            u = a + 0.1 * torch.randn_like(a)
            tt = torch.full((bs, 1), 0.9, device=dev)
            aux = ((net(torch.cat([o, u, tt], 1)) - a) ** 2).mean()
            loss = main + 100 * aux
        else:
            z = torch.randn(bs, dout, device=dev)
            tt = torch.rand(bs, 1, device=dev)
            yt = (1 - tt) * z + tt * a
            loss = ((net(torch.cat([o, yt, tt], 1)) - (a - z)) ** 2).mean()
        opt.zero_grad(); loss.backward(); opt.step(); sch.step()
        if snap_at and it in snap_at:
            snaps[it] = {k: v.detach().clone() for k, v in net.state_dict().items()}
    return net, a_std, snaps

def deploy(kind, net, a_std, O, z_samples=4, view="main"):
    O_t = torch.tensor(np.asarray(O, np.float32), device=dev)
    B = len(O_t); dout = 2 * H
    with torch.no_grad():
        if kind in ("mse", "pds"):
            out = net(O_t)
        elif kind == "mip":
            if view == "main":
                z0 = torch.zeros(B, dout, device=dev)
                out = net(torch.cat([O_t, z0, z0[:, :1]], 1))
            else:
                out = view  # handled externally
        else:
            outs = []
            for k in range(z_samples):
                g = torch.Generator(device="cpu").manual_seed(k)
                y = torch.randn(B, dout, generator=g).to(dev)
                for i in range(10):
                    tt = torch.full((B, 1), i / 10, device=dev)
                    y = y + 0.1 * net(torch.cat([O_t, y, tt], 1))
                outs.append(y)
            out = torch.stack(outs).mean(0)
    return (out.cpu().numpy() * a_std).reshape(B, H, 2)

# ---------- measurements ----------
def Kcurve(kind, net, a_std, variant, lam, which="first", nbins=20, reps=4):
    ps = np.linspace(0.03, 0.97, nbins)
    rng = np.random.RandomState(11)
    xis = rng.uniform(-1, 1, (8, 2))
    Ks = []
    for p0 in ps:
        slopes = []
        for xi0 in xis:
            nv = np.linspace(-0.20, 0.20, 21)
            pv = np.full_like(nv, p0)
            xiv = np.tile(xi0, (len(nv), 1))
            T, N = TN(pv, xiv)
            q = cline(pv, xiv) + nv[:, None] * N
            ans = []
            for rep in range(reps):
                o = make_obs(q, xiv, nv, pv, np.random.RandomState(100 + rep))
                A = deploy(kind, net, a_std, o)
                if which == "first": a0 = A[:, 0]
                elif which == "mean": a0 = A.mean(1)
                else: a0 = A[:, -1]
                ans.append(np.einsum('ij,ij->i', a0, N))
            an = np.mean(ans, 0)
            X = np.stack([nv, np.ones_like(nv)], 1)
            slopes.append(-np.linalg.lstsq(X, an, rcond=None)[0][0])
        Ks.append(np.mean(slopes))
    return ps, np.array(Ks) / lam  # normalize by lambda_rec to compare with K_data

def reportK(tagline, variant, ps, Km):
    Kd = kdata(variant, ps)
    cos = float(Km @ Kd) / (np.linalg.norm(Km) * np.linalg.norm(Kd) + 1e-12)
    gn = float(Km @ Kd) / (np.linalg.norm(Kd) ** 2 + 1e-12)
    msk = np.abs(Kd) > 0.02
    sg = float(np.mean(np.sign(Km[msk]) == np.sign(Kd[msk]))) if msk.any() else np.nan
    r2 = 1 - np.sum((Km - Kd) ** 2) / (np.sum((Kd - Kd.mean()) ** 2) + 1e-12)
    extra = ""
    if variant == "B": extra = f" Kzero={Km[ps<0.7].mean():+.3f}"
    if variant == "C": extra = f" Kout={Km[kdata('C',ps)<-0.02].mean():+.3f}"
    print(f"TOYINS-K {tagline} cos={cos:+.3f} gain={gn:.3f} sign={sg:.2f} R2={r2:.3f} "
          f"MAE={np.abs(Km-Kd).mean():.3f}{extra}", flush=True)
    return gn

def closedloop(tag, variant, kind, net, a_std, lam, n_ep=300, exec_h=2):
    rng = np.random.RandomState(21)
    PG = np.linspace(0, 1, 800)
    succ = coll = blow = tout = 0; fin, mx = [], []
    for ep in range(n_ep):
        xi = rng.uniform(-1, 1, 2)
        cgrid = cline(PG, np.tile(xi, (len(PG), 1)))
        n0 = rng.uniform(-0.12, 0.12)
        T0, N0 = TN(np.zeros(1), xi[None])
        q = (cline(np.zeros(1), xi[None]) + n0 * N0)[0]
        maxn = abs(n0); done = None
        t = 0
        while t < 300:
            d = np.linalg.norm(cgrid - q, axis=1); i = d.argmin()
            p = PG[i]; _, N = TN(np.array([p]), xi[None])
            n = float((q - cgrid[i]) @ N[0])
            maxn = max(maxn, abs(n))
            if kind == "expert":
                A = chunk(variant, np.array([p]), np.array([n]), xi[None], lam, rng)[0]
            else:
                o = make_obs(q[None], xi[None], np.array([n]), np.array([p]), rng)
                A = deploy(kind, net, a_std, o)[0]
            for h in range(exec_h):
                q = q + A[h] + rng.randn(2) * 0.002
                t += 1
                d = np.linalg.norm(cgrid - q, axis=1); i = d.argmin()
                p = PG[i]; _, N = TN(np.array([p]), xi[None])
                n = float((q - cgrid[i]) @ N[0])
                maxn = max(maxn, abs(n))
                xh = xhole(xi[None])[0]
                wp = 0.20 if p < 0.6 else max(0.035, 0.20 - (p - 0.6) / 0.4 * (0.20 - 0.035))
                if p > 0.55 and abs(q[0] - xh) > wp: done = "coll"; break
                if abs(n) > 0.25: done = "blow"; break
                if p > 0.98 and abs(q[0] - xh) < 0.03: done = "succ"; break
            if done: break
        if done is None: done = "tout"
        if done == "succ": succ += 1
        elif done == "coll": coll += 1
        elif done == "blow": blow += 1
        else: tout += 1
        fin.append(abs(n)); mx.append(maxn)
    print(f"TOYINS-CL {tag} SR={succ/n_ep:.2f} coll={coll/n_ep:.2f} blow={blow/n_ep:.2f} "
          f"tout={tout/n_ep:.2f} final|n|_p50={np.median(fin):.3f} max|n|_p50={np.median(mx):.3f} "
          f"cross.15={np.mean(np.array(mx)>0.15):.2f}", flush=True)
    return succ / n_ep

def jac(tag, variant, kind, net, a_std, sig, lam):
    rng = np.random.RandomState(31)
    for region, noff in [("edge", 2 * sig), ("off2", 0.04), ("off4", 0.08)]:
        pv = rng.rand(100) * 0.9 + 0.05
        xiv = rng.uniform(-1, 1, (100, 2))
        sgn = np.where(rng.rand(100) > .5, 1., -1.)
        nv = noff * sgn
        T, N = TN(pv, xiv)
        q0 = cline(pv, xiv) + nv[:, None] * N
        eps = 0.01
        o0 = make_obs(q0, xiv, nv, pv, np.random.RandomState(41))
        oN = make_obs(q0 + eps * N, xiv, nv + eps, pv, np.random.RandomState(41))
        oT = make_obs(q0 + eps * T, xiv, nv, pv, np.random.RandomState(41))
        A0 = deploy(kind, net, a_std, o0); AN = deploy(kind, net, a_std, oN); AT = deploy(kind, net, a_std, oT)
        dN = (AN - A0)[:, 0] / eps; dT = (AT - A0)[:, 0] / eps
        rj = -np.einsum('ij,ij->i', dN, N) / lam
        rj_chunk = -np.einsum('bhi,bi->bh', (AN - A0) / eps, N) / lam
        cc = np.median(np.std(rj_chunk, axis=1) / (np.abs(np.median(rj_chunk, axis=1)) + 1e-6))
        print(f"TOYINS-J {tag} region={region} |Jn|={np.median(np.linalg.norm(dN,axis=1)):.3f} "
              f"|Jt|={np.median(np.linalg.norm(dT,axis=1)):.3f} rJ/lam={np.median(rj):+.3f} "
              f"frac>0={(rj>0).mean():.2f} chunk_cv={cc:.2f}", flush=True)

# ---------- run grid ----------
REGIMES = {"easy": (50000, 0.03, 0.2, ["A"]), "hard": (5000, 0.01, 0.1, ["A", "B", "C"]),
           "vhard": (2000, 0.005, 0.05, ["A"])}
KINDS = ["mse", "mip", "flow", "pds"]
summary = []
for reg, (NS, sig, lam, variants) in REGIMES.items():
    for variant in variants:
        rng = np.random.RandomState(1)
        O, A, q, xi, p, n = gen(variant, NS, sig, lam, rng)
        Oh, Ah, *_ = gen(variant, 4000, sig, lam, np.random.RandomState(2))
        closedloop(f"{reg} {variant} EXPERT", variant, "expert", None, 1.0, lam, n_ep=200)
        curves = {}
        for kind in KINDS:
            if kind == "pds" and reg != "hard": continue
            snap_at = [1000, 5000, 10000, 25000] if (kind == "mip" and reg == "hard") else None
            net, a_std, snaps = train_model(kind, O, A, variant, sig, lam, snap_at=snap_at, pd_k=lam * 0.25)
            pr = deploy(kind, net, a_std, Oh[:2000]).reshape(2000, -1)
            he = float(np.sqrt(((pr - Ah[:2000]) ** 2).sum(1)).mean())
            prt = deploy(kind, net, a_std, O[:2000]).reshape(2000, -1)
            te = float(np.sqrt(((prt - A[:2000]) ** 2).sum(1)).mean())
            print(f"TOYINS-FIT {reg} {variant} {kind} trainErr={te:.5f} heldErr={he:.5f}", flush=True)
            gains = {}
            for which in ["first", "mean"]:
                ps, Km = Kcurve(kind, net, a_std, variant, lam, which)
                gains[which] = reportK(f"{reg} {variant} {kind} {which}", variant, ps, Km)
                if which == "first": curves[kind] = (ps, Km)
            sr = closedloop(f"{reg} {variant} {kind}", variant, kind, net, a_std, lam)
            jac(f"{reg} {variant} {kind}", variant, kind, net, a_std, sig, lam)
            summary.append(dict(reg=reg, var=variant, kind=kind, sr=sr, gain=gains["first"], held=he))
            # F: flow K_recon across t
            if kind == "flow" and reg == "hard":
                for t0 in [0.1, 0.5, 0.9]:
                    rngf = np.random.RandomState(51)
                    ps2 = np.linspace(0.05, 0.95, 12); Kr = []
                    xis = rngf.uniform(-1, 1, (6, 2))
                    for p0 in ps2:
                        sl = []
                        for xi0 in xis:
                            nv = np.linspace(-0.15, 0.15, 13)
                            pv = np.full_like(nv, p0); xiv = np.tile(xi0, (len(nv), 1))
                            T, N = TN(pv, xiv)
                            qv = cline(pv, xiv) + nv[:, None] * N
                            o = make_obs(qv, xiv, nv, pv, np.random.RandomState(61))
                            Adata = chunk(variant, pv, nv, xiv, lam, np.random.RandomState(62)).reshape(len(nv), -1) / a_std
                            g = torch.Generator(device="cpu").manual_seed(0)
                            z = torch.randn(len(nv), 2 * H, generator=g).numpy()
                            yt = (1 - t0) * z + t0 * Adata
                            inp = torch.tensor(np.concatenate([o, yt, np.full((len(nv), 1), t0)], 1), dtype=torch.float32, device=dev)
                            with torch.no_grad():
                                v = net(inp).cpu().numpy()
                            y1 = (yt + (1 - t0) * v) * a_std
                            a0 = y1.reshape(len(nv), H, 2)[:, 0]
                            an = np.einsum('ij,ij->i', a0, N)
                            X = np.stack([nv, np.ones_like(nv)], 1)
                            sl.append(-np.linalg.lstsq(X, an, rcond=None)[0][0] / lam)
                        Kr.append(np.mean(sl))
                    Kd = kdata(variant, ps2)
                    cr = float(np.array(Kr) @ Kd) / (np.linalg.norm(Kr) * np.linalg.norm(Kd) + 1e-12)
                    gr = float(np.array(Kr) @ Kd) / (np.linalg.norm(Kd) ** 2 + 1e-12)
                    print(f"TOYINS-FLOWT {reg} {variant} t={t0} cos(Krecon,Kd)={cr:+.2f} gain={gr:.2f}", flush=True)
            # F: MIP transfer
            if kind == "mip" and reg == "hard" and snaps:
                for ck, sd in snaps.items():
                    net.load_state_dict(sd)
                    for view in ["main", "aux", "aux_ufix"]:
                        ps2 = np.linspace(0.05, 0.95, 12); Ks = []
                        rngm = np.random.RandomState(71)
                        xis = rngm.uniform(-1, 1, (6, 2))
                        for p0 in ps2:
                            sl = []
                            for xi0 in xis:
                                nv = np.linspace(-0.15, 0.15, 13)
                                pv = np.full_like(nv, p0); xiv = np.tile(xi0, (len(nv), 1))
                                T, N = TN(pv, xiv)
                                qv = cline(pv, xiv) + nv[:, None] * N
                                o = torch.tensor(make_obs(qv, xiv, nv, pv, np.random.RandomState(81)), device=dev)
                                if view == "main":
                                    z0 = torch.zeros(len(nv), 2 * H, device=dev)
                                    inp = torch.cat([o, z0, z0[:, :1]], 1)
                                else:
                                    nn_u = nv if view == "aux" else np.zeros_like(nv)
                                    Au = chunk(variant, pv, nn_u, xiv, lam, np.random.RandomState(82)).reshape(len(nv), -1) / a_std
                                    u = torch.tensor(Au.astype(np.float32), device=dev) + 0.1 * torch.randn(len(nv), 2 * H, device=dev)
                                    inp = torch.cat([o, u, torch.full((len(nv), 1), 0.9, device=dev)], 1)
                                with torch.no_grad():
                                    a0 = (net(inp).cpu().numpy() * a_std).reshape(len(nv), H, 2)[:, 0]
                                an = np.einsum('ij,ij->i', a0, N)
                                X = np.stack([nv, np.ones_like(nv)], 1)
                                sl.append(-np.linalg.lstsq(X, an, rcond=None)[0][0] / lam)
                            Ks.append(np.mean(sl))
                        Kd = kdata(variant, ps2)
                        gn = float(np.array(Ks) @ Kd) / (np.linalg.norm(Kd) ** 2 + 1e-12)
                        cs = float(np.array(Ks) @ Kd) / (np.linalg.norm(Ks) * np.linalg.norm(Kd) + 1e-12)
                        print(f"TOYINS-TRANSFER {variant} ck={ck} view={view} cos={cs:+.3f} gain={gn:.3f}", flush=True)
                net.load_state_dict(snaps[max(snaps)])
        # figure per (reg, variant)
        plt.figure(figsize=(7, 4))
        psx = None
        for kind, c in [("mse", "tab:red"), ("mip", "tab:blue"), ("flow", "tab:green"), ("pds", "tab:orange")]:
            if kind in curves:
                psx = curves[kind][0]
                plt.plot(*curves[kind], c=c, label=f'K_{kind}/lam')
        if psx is not None:
            plt.plot(psx, kdata(variant, psx), 'k-', lw=2, label='K_data')
        plt.axhline(0, color='gray', lw=.5); plt.legend(); plt.xlabel('p'); plt.ylabel('K/lam')
        plt.title(f'{reg} variant {variant}')
        plt.tight_layout(); plt.savefig(f"{OUT}/K_{reg}_{variant}.png", dpi=140); plt.close()

json.dump(summary, open(f"{OUT}/summary.json", "w"), indent=1)
# SR vs gain scatter
plt.figure(figsize=(5.5, 4.5))
for s in summary:
    m = {"mse": "o", "mip": "s", "flow": "^", "pds": "D"}[s["kind"]]
    plt.scatter(s["gain"], s["sr"], marker=m, s=60, label=f'{s["reg"]}-{s["var"]}-{s["kind"]}')
plt.xlabel("K gain vs K_data (first action)"); plt.ylabel("closed-loop SR")
plt.tight_layout(); plt.savefig(f"{OUT}/SR_vs_gain.png", dpi=140); plt.close()
print("DONE", flush=True)
