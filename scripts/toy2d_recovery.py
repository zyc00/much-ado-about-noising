"""TOY-2D: does MIP/Flow learn the data-sourced recovery slope K_data?
Variants: A (PD tube k=0.25), B (no recovery until p>0.85), C (zigzag k(p)).
Models: MSE, MIP-step1 (t=0.9 aux, w=100, deploy main), rectified Flow (10-step Euler),
plus PD oracle baselines for closed-loop reference.
Outputs: TOY2D-prefixed tables + figures in analysis/toy2d/.
"""
import numpy as np, torch, torch.nn as nn, os, json
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

torch.manual_seed(0); np.random.seed(0)
dev = "cuda" if torch.cuda.is_available() else "cpu"
OUT = "analysis/toy2d"; os.makedirs(OUT, exist_ok=True)
SIG = 0.03  # sigma_train

# ---------- environment ----------
def cline(p):
    return np.stack([p, 0.35 * np.sin(2 * np.pi * p) + 0.12 * np.sin(6 * np.pi * p)], -1)
def cdot(p):
    return np.stack([np.ones_like(p),
                     0.35 * 2 * np.pi * np.cos(2 * np.pi * p) + 0.12 * 6 * np.pi * np.cos(6 * np.pi * p)], -1)
def TN(p):
    d = cdot(p); T = d / np.linalg.norm(d, axis=-1, keepdims=True)
    N = np.stack([-T[..., 1], T[..., 0]], -1)
    return T, N
PG = np.linspace(0, 1, 4001)
CG = cline(PG)
def project(s):
    d = np.linalg.norm(CG[None] - s[:, None], axis=-1)
    i = d.argmin(1)
    p = PG[i]
    _, N = TN(p)
    n = np.einsum('ij,ij->i', s - CG[i], N)
    return p, n
def vt(p):
    return 0.03 * (1 + 0.2 * np.cos(2 * np.pi * p))
def kA(p): return 0.25 * np.ones_like(p)
def kC(p):
    k = 0.25 * np.tanh(5 * np.sin(6 * np.pi * p))
    return np.where(p > 0.85, 0.35, k)
def gB(p): return 1 / (1 + np.exp(-(p - 0.85) / 0.03))

def vn(variant, p, n, rng):
    if variant == "A": return -kA(p) * n
    if variant == "C": return -kC(p) * n
    g = gB(p)
    eta = rng.randn(*p.shape) * 0.35 * SIG
    return -g * 0.35 * n + (1 - g) * eta
def kdata(variant, p):
    if variant == "A": return kA(p)
    if variant == "C": return kC(p)
    return gB(p) * 0.35

def gen(variant, n_samples, rng):
    p = rng.rand(n_samples)
    n = np.clip(rng.randn(n_samples) * SIG, -3 * SIG, 3 * SIG)
    T, N = TN(p)
    s = cline(p) + n[:, None] * N
    a = vt(p)[:, None] * T + vn(variant, p, n, rng)[:, None] * N
    return s.astype(np.float32), a.astype(np.float32), p, n

# ---------- models ----------
class MLP(nn.Module):
    def __init__(self, din, dout=2, w=256, depth=3):
        super().__init__()
        layers = []
        d = din
        for _ in range(depth):
            layers += [nn.Linear(d, w), nn.SiLU()]; d = w
        layers += [nn.Linear(d, dout)]
        self.net = nn.Sequential(*layers)
    def forward(self, x): return self.net(x)

def train_model(kind, S, A, steps=20000, bs=512):
    a_std = float(A.std())
    An = A / a_std
    S_t = torch.tensor(S, device=dev); A_t = torch.tensor(An, device=dev)
    din = {"mse": 2, "mip": 5, "flow": 5}[kind]
    net = MLP(din).to(dev)
    opt = torch.optim.Adam(net.parameters(), lr=1e-3)
    sch = torch.optim.lr_scheduler.CosineAnnealingLR(opt, steps)
    for it in range(steps):
        idx = torch.randint(0, len(S_t), (bs,), device=dev)
        s, a = S_t[idx], A_t[idx]
        if kind == "mse":
            loss = ((net(s) - a) ** 2).mean()
        elif kind == "mip":
            z0 = torch.zeros(bs, 2, device=dev)
            main = ((net(torch.cat([s, z0, z0[:, :1]], 1)) - a) ** 2).mean()
            u = a + 0.1 * torch.randn_like(a)
            tt = torch.full((bs, 1), 0.9, device=dev)
            aux = ((net(torch.cat([s, u, tt], 1)) - a) ** 2).mean()
            loss = main + 100 * aux
        else:  # flow
            z = torch.randn(bs, 2, device=dev)
            tt = torch.rand(bs, 1, device=dev)
            yt = (1 - tt) * z + tt * a
            v = net(torch.cat([s, yt, tt], 1))
            loss = ((v - (a - z)) ** 2).mean()
        opt.zero_grad(); loss.backward(); opt.step(); sch.step()
    return net, a_std

def deploy(kind, net, a_std, S, z_samples=8):
    S_t = torch.tensor(np.asarray(S, np.float32), device=dev)
    B = len(S_t)
    with torch.no_grad():
        if kind == "mse":
            out = net(S_t)
        elif kind == "mip":
            z0 = torch.zeros(B, 2, device=dev)
            out = net(torch.cat([S_t, z0, z0[:, :1]], 1))
        else:
            outs = []
            for k in range(z_samples):
                g = torch.Generator(device="cpu").manual_seed(k)
                y = torch.randn(B, 2, generator=g).to(dev)
                for i in range(10):
                    t0 = i / 10
                    tt = torch.full((B, 1), t0, device=dev)
                    v = net(torch.cat([S_t, y, tt], 1))
                    y = y + 0.1 * v
                outs.append(y)
            out = torch.stack(outs).mean(0)
    return out.cpu().numpy() * a_std

# ---------- measurements ----------
def fit_Kp(kind, net, a_std, variant, nbins=24):
    ps = np.linspace(0.02, 0.98, nbins)
    Ks = []
    for p0 in ps:
        pv = np.full(21, p0)
        nv = np.linspace(-0.2, 0.2, 21)
        T, N = TN(pv)
        s = cline(pv) + nv[:, None] * N
        a = deploy(kind, net, a_std, s)
        an = np.einsum('ij,ij->i', a, N)
        A_ = np.stack([nv, np.ones_like(nv)], 1)
        slope = np.linalg.lstsq(A_, an, rcond=None)[0][0]
        Ks.append(-slope)
    return ps, np.array(Ks)

def report_K(tag, variant, ps, Km):
    Kd = kdata(variant, ps)
    cos = float(Km @ Kd) / (np.linalg.norm(Km) * np.linalg.norm(Kd) + 1e-12)
    gainr = float(Km @ Kd) / (np.linalg.norm(Kd) ** 2 + 1e-12)
    mask = np.abs(Kd) > 0.02
    sgn = float(np.mean(np.sign(Km[mask]) == np.sign(Kd[mask]))) if mask.any() else np.nan
    ss = 1 - np.sum((Km - Kd) ** 2) / (np.sum((Kd - Kd.mean()) ** 2) + 1e-12)
    mae = float(np.abs(Km - Kd).mean())
    # zero-zone spontaneous recovery (variant B pre-final; C outward zones)
    if variant == "B":
        zz = ps < 0.7
        extra = f" K_zero-zone_mean={Km[zz].mean():+.3f} K_final_mean={Km[~zz & (ps>0.9)].mean() if (~zz & (ps>0.9)).any() else np.nan:+.3f}"
    elif variant == "C":
        outz = kdata('C', ps) < -0.02
        extra = f" K_outward-zone_mean={Km[outz].mean():+.3f} (data {kdata('C',ps)[outz].mean():+.3f})"
    else:
        extra = ""
    print(f"TOY2D-K {variant} {tag} cos={cos:+.3f} gain={gainr:.3f} sign_agree={sgn:.2f} "
          f"R2={ss:.3f} MAE={mae:.3f}{extra}", flush=True)
    return Kd

def offtube(tag, variant, kind, net, a_std, rng):
    for m in [1, 2, 3, 4, 6]:
        pv = rng.rand(400) * 0.96 + 0.02
        sgnv = np.where(rng.rand(400) > 0.5, 1.0, -1.0)
        nv = m * SIG * sgnv
        T, N = TN(pv)
        s = cline(pv) + nv[:, None] * N
        a = deploy(kind, net, a_std, s)
        an = np.einsum('ij,ij->i', a, N)
        r = -an * sgnv
        rgt = kdata(variant, pv) * np.abs(nv)
        gm = np.abs(rgt) > 1e-4
        ratio = np.median(r[gm] / rgt[gm]) if gm.any() else np.nan
        print(f"TOY2D-OFF {variant} {tag} m={m} r_p50={np.median(r):+.5f} frac_r>0={(r>0).mean():.2f} "
              f"r/GT_p50={ratio:.2f}", flush=True)

def closedloop(tag, variant, actfn, rng, n_ep=300):
    succ = fail = 0; fin = []; steps_used = []
    for ep in range(n_ep):
        p, n = 0.0, rng.uniform(-0.10, 0.10)
        T0, N0 = TN(np.array([p]))
        s = (cline(np.array([p])) + n * N0)[0]
        ok = False
        for t in range(300):
            a = actfn(s[None])[0]
            s = s + a + rng.randn(2) * 0.002
            pp, nn_ = project(s[None])
            if pp[0] > 0.98 and abs(nn_[0]) < 0.03:
                ok = True; break
            if abs(nn_[0]) > 0.25:
                break
        pp, nn_ = project(s[None])
        fin.append(abs(nn_[0])); steps_used.append(t)
        if ok: succ += 1
        elif abs(nn_[0]) > 0.25: fail += 1
    print(f"TOY2D-CL {variant} {tag} SR={succ/n_ep:.2f} blowup={fail/n_ep:.2f} "
          f"final|n|_p50={np.median(fin):.3f} steps_p50={np.median(steps_used):.0f}", flush=True)
    return succ / n_ep

def jacobian(tag, variant, kind, net, a_std, rng):
    for region, noff in [("on", 0.0), ("off2", 2 * SIG), ("off4", 4 * SIG)]:
        pv = rng.rand(200) * 0.9 + 0.05
        sg = np.where(rng.rand(200) > 0.5, 1., -1.)
        T, N = TN(pv)
        s0 = cline(pv) + (noff * sg)[:, None] * N
        eps = 0.004
        Jn, Jt, RJ = [], [], []
        a0 = deploy(kind, net, a_std, s0)
        aN = deploy(kind, net, a_std, s0 + eps * N)
        aT = deploy(kind, net, a_std, s0 + eps * T)
        dN = (aN - a0) / eps; dT = (aT - a0) / eps
        rj = -np.einsum('ij,ij->i', dN, N)
        print(f"TOY2D-J {variant} {tag} region={region} |Jn|_p50={np.median(np.linalg.norm(dN,axis=1)):.3f} "
              f"|Jt|_p50={np.median(np.linalg.norm(dT,axis=1)):.3f} rJ_p50={np.median(rj):+.3f} "
              f"frac_rJ>0={(rj>0).mean():.2f}", flush=True)

def flow_slope(variant, net, a_std, rng):
    for t0 in [0.1, 0.25, 0.5, 0.75, 0.9]:
        ps = np.linspace(0.05, 0.95, 19)
        Kv, Kr = [], []
        for p0 in ps:
            nv = np.linspace(-0.15, 0.15, 15)
            pv = np.full_like(nv, p0)
            T, N = TN(pv)
            s = cline(pv) + nv[:, None] * N
            a = (vt(pv)[:, None] * T + vn(variant, pv, nv, rng)[:, None] * N) / a_std
            vs, ys = [], []
            for k in range(8):
                g = torch.Generator(device="cpu").manual_seed(k)
                z = torch.randn(len(s), 2, generator=g).numpy()
                yt = (1 - t0) * z + t0 * a
                inp = torch.tensor(np.concatenate([s, yt, np.full((len(s), 1), t0)], 1), dtype=torch.float32, device=dev)
                with torch.no_grad():
                    v = net(inp).cpu().numpy()
                vs.append(v); ys.append(yt + (1 - t0) * v)
            v = np.mean(vs, 0); y1 = np.mean(ys, 0)
            vn_c = np.einsum('ij,ij->i', v * a_std, N)
            y1n = np.einsum('ij,ij->i', y1 * a_std, N)
            A_ = np.stack([nv, np.ones_like(nv)], 1)
            Kv.append(-np.linalg.lstsq(A_, vn_c, rcond=None)[0][0])
            Kr.append(-np.linalg.lstsq(A_, y1n, rcond=None)[0][0])
        Kd = kdata(variant, ps)
        cv = float(np.array(Kv) @ Kd) / (np.linalg.norm(Kv) * np.linalg.norm(Kd) + 1e-12)
        cr = float(np.array(Kr) @ Kd) / (np.linalg.norm(Kr) * np.linalg.norm(Kd) + 1e-12)
        gr = float(np.array(Kr) @ Kd) / (np.linalg.norm(Kd) ** 2 + 1e-12)
        print(f"TOY2D-FLOWT {variant} t={t0} cos(Kv,Kd)={cv:+.2f} cos(Krecon,Kd)={cr:+.2f} "
              f"gain_recon={gr:.2f}", flush=True)

# ---------- run ----------
results = {}
for variant in ["A", "B", "C"]:
    rng = np.random.RandomState(1)
    S, A, P, Nn = gen(variant, 50000, rng)
    Sh, Ah, _, _ = gen(variant, 10000, np.random.RandomState(2))
    curves = {}
    for kind in ["mse", "mip", "flow"]:
        net, a_std = train_model(kind, S, A)
        pred = deploy(kind, net, a_std, Sh[:4000])
        he = float(np.sqrt(((pred - Ah[:4000]) ** 2).sum(1)).mean())
        predt = deploy(kind, net, a_std, S[:4000])
        te = float(np.sqrt(((predt - A[:4000]) ** 2).sum(1)).mean())
        print(f"TOY2D-FIT {variant} {kind} trainErr={te:.5f} heldErr={he:.5f} (|a|~{np.linalg.norm(Ah,axis=1).mean():.3f})", flush=True)
        ps, Km = fit_Kp(kind, net, a_std, variant)
        report_K(kind, variant, ps, Km)
        curves[kind] = (ps, Km)
        offtube(kind, variant, kind, net, a_std, np.random.RandomState(3))
        jacobian(kind, variant, kind, net, a_std, np.random.RandomState(4))
        actfn = lambda s, k=kind, n_=net, a_=a_std: deploy(k, n_, a_, s, z_samples=1)
        closedloop(kind, variant, actfn, np.random.RandomState(5))
        if kind == "flow":
            flow_slope(variant, net, a_std, np.random.RandomState(6))
        results[(variant, kind)] = (ps.tolist(), Km.tolist())
    # PD oracle closed-loop reference
    def pd_act(s, damp=1.0):
        p, n = project(s)
        T, N = TN(p)
        return vt(p)[:, None] * T + (-damp * kdata(variant, p) * n)[:, None] * N
    closedloop("PDoracle", variant, lambda s: pd_act(s), np.random.RandomState(5))
    closedloop("PDdamped0.6", variant, lambda s: pd_act(s, 0.6), np.random.RandomState(5))

    # Figure 1: K curves
    plt.figure(figsize=(7, 4))
    ps = np.array(curves["mse"][0])
    plt.plot(ps, kdata(variant, ps), 'k-', lw=2, label='K_data')
    for kind, c in [("mse", "tab:red"), ("mip", "tab:blue"), ("flow", "tab:green")]:
        plt.plot(*curves[kind], c=c, label=f'K_{kind}')
    plt.axhline(0, color='gray', lw=0.5); plt.xlabel('p'); plt.ylabel('K(p)')
    plt.title(f'Variant {variant}: learned normal slope vs data'); plt.legend()
    plt.tight_layout(); plt.savefig(f"{OUT}/K_curves_{variant}.png", dpi=140); plt.close()

json.dump({f"{v}_{k}": r for (v, k), r in results.items()}, open(f"{OUT}/curves.json", "w"))
print("DONE", flush=True)
