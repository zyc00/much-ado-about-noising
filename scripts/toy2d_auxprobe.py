"""Toy-2D EXP-5: MIP aux-view slope across checkpoints (1k/5k/10k/20k).
K_main(p): f(t=0,u=0,s). K_aux(p): f(t=0.9, u=a_data(s)+0.1eps, s) (training-time input;
u carries the label). K_aux_ufix(p): f(t=0.9, u=a_nom(p)+0.1eps, s) with u NOT a function
of n -- isolates the s-pathway of the aux branch.
Same env/net/seeds as toy2d_recovery.py."""
import numpy as np, torch, torch.nn as nn, os

torch.manual_seed(0); np.random.seed(0)
dev = "cuda" if torch.cuda.is_available() else "cpu"
SIG = 0.03

def cline(p):
    return np.stack([p, 0.35 * np.sin(2 * np.pi * p) + 0.12 * np.sin(6 * np.pi * p)], -1)
def cdot(p):
    return np.stack([np.ones_like(p),
                     0.35 * 2 * np.pi * np.cos(2 * np.pi * p) + 0.12 * 6 * np.pi * np.cos(6 * np.pi * p)], -1)
def TN(p):
    d = cdot(p); T = d / np.linalg.norm(d, axis=-1, keepdims=True)
    return T, np.stack([-T[..., 1], T[..., 0]], -1)
def vt(p): return 0.03 * (1 + 0.2 * np.cos(2 * np.pi * p))
def kA(p): return 0.25 * np.ones_like(p)
def kC(p):
    k = 0.25 * np.tanh(5 * np.sin(6 * np.pi * p))
    return np.where(p > 0.85, 0.35, k)
def gB(p): return 1 / (1 + np.exp(-(p - 0.85) / 0.03))
def vn(variant, p, n, rng):
    if variant == "A": return -kA(p) * n
    if variant == "C": return -kC(p) * n
    return -gB(p) * 0.35 * n + (1 - gB(p)) * rng.randn(*p.shape) * 0.35 * SIG
def kdata(variant, p):
    return {"A": kA, "C": kC}.get(variant, lambda q: gB(q) * 0.35)(p)
def act_data(variant, p, n, rng):
    T, N = TN(p)
    return vt(p)[:, None] * T + vn(variant, p, n, rng)[:, None] * N
def gen(variant, ns, rng):
    p = rng.rand(ns); n = np.clip(rng.randn(ns) * SIG, -3 * SIG, 3 * SIG)
    T, N = TN(p)
    s = cline(p) + n[:, None] * N
    return s.astype(np.float32), act_data(variant, p, n, rng).astype(np.float32)

class MLP(nn.Module):
    def __init__(self, din=5, w=256, depth=3):
        super().__init__()
        L = []; d = din
        for _ in range(depth): L += [nn.Linear(d, w), nn.SiLU()]; d = w
        L += [nn.Linear(d, 2)]
        self.net = nn.Sequential(*L)
    def forward(self, x): return self.net(x)

CKPTS = [1000, 5000, 10000, 20000]
for variant in ["A", "B", "C"]:
    rng = np.random.RandomState(1)
    S, A = gen(variant, 50000, rng)
    a_std = float(A.std())
    S_t = torch.tensor(S, device=dev); A_t = torch.tensor(A / a_std, device=dev)
    torch.manual_seed(0)
    net = MLP().to(dev)
    opt = torch.optim.Adam(net.parameters(), lr=1e-3)
    sch = torch.optim.lr_scheduler.CosineAnnealingLR(opt, 20000)
    snaps = {}
    for it in range(1, 20001):
        idx = torch.randint(0, len(S_t), (512,), device=dev)
        s, a = S_t[idx], A_t[idx]
        z0 = torch.zeros(512, 2, device=dev)
        main = ((net(torch.cat([s, z0, z0[:, :1]], 1)) - a) ** 2).mean()
        u = a + 0.1 * torch.randn_like(a)
        tt = torch.full((512, 1), 0.9, device=dev)
        aux = ((net(torch.cat([s, u, tt], 1)) - a) ** 2).mean()
        loss = main + 100 * aux
        opt.zero_grad(); loss.backward(); opt.step(); sch.step()
        if it in CKPTS:
            snaps[it] = {k: v.detach().clone() for k, v in net.state_dict().items()}

    def slope_curve(net, view, variant, nbins=24, reps=16):
        ps = np.linspace(0.02, 0.98, nbins)
        Ks = []
        rloc = np.random.RandomState(7)
        for p0 in ps:
            nv = np.linspace(-0.2, 0.2, 21)
            pv = np.full_like(nv, p0)
            T, N = TN(pv)
            s = torch.tensor((cline(pv) + nv[:, None] * N).astype(np.float32), device=dev)
            outs = []
            for rep in range(reps):
                if view == "main":
                    z0 = torch.zeros(len(s), 2, device=dev)
                    inp = torch.cat([s, z0, z0[:, :1]], 1)
                elif view == "aux":
                    a_u = act_data(variant, pv, nv, rloc) / a_std
                    u = torch.tensor(a_u.astype(np.float32), device=dev) + 0.1 * torch.randn(len(s), 2, device=dev)
                    inp = torch.cat([s, u, torch.full((len(s), 1), 0.9, device=dev)], 1)
                else:  # aux_ufix: u from nominal (n=0) action, not n-dependent
                    a_u = act_data(variant, pv, np.zeros_like(nv), rloc) / a_std
                    u = torch.tensor(a_u.astype(np.float32), device=dev) + 0.1 * torch.randn(len(s), 2, device=dev)
                    inp = torch.cat([s, u, torch.full((len(s), 1), 0.9, device=dev)], 1)
                with torch.no_grad():
                    outs.append(net(inp).cpu().numpy() * a_std)
                if view == "main": break
            a = np.mean(outs, 0)
            an = np.einsum('ij,ij->i', a, N)
            X = np.stack([nv, np.ones_like(nv)], 1)
            Ks.append(-np.linalg.lstsq(X, an, rcond=None)[0][0])
        return ps, np.array(Ks)

    for ck in CKPTS:
        net.load_state_dict(snaps[ck])
        row = {}
        for view in ["main", "aux", "aux_ufix"]:
            ps, Km = slope_curve(net, view, variant)
            Kd = kdata(variant, ps)
            cos = float(Km @ Kd) / (np.linalg.norm(Km) * np.linalg.norm(Kd) + 1e-12)
            gn = float(Km @ Kd) / (np.linalg.norm(Kd) ** 2 + 1e-12)
            msk = np.abs(Kd) > 0.02
            sg = float(np.mean(np.sign(Km[msk]) == np.sign(Kd[msk]))) if msk.any() else np.nan
            r2 = 1 - np.sum((Km - Kd) ** 2) / (np.sum((Kd - Kd.mean()) ** 2) + 1e-12)
            zz = f" Kzero={Km[ps<0.7].mean():+.3f}" if variant == "B" else (
                 f" Kout={Km[kdata('C',ps)<-0.02].mean():+.3f}" if variant == "C" else "")
            print(f"AUXPROBE {variant} ck={ck} view={view} cos={cos:+.3f} gain={gn:.3f} "
                  f"sign={sg:.2f} R2={r2:.3f}{zz}", flush=True)
print("DONE", flush=True)
