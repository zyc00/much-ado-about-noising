"""Natural-error version of the 3D MP toy: no kicks. Failure initiation =
interpolation error on FRESH anchors with a LIMITED training set, matching
the real natural-episode experiment (fresh placements, 200 demos).

Geometry, controller, and models identical to toy_mp3d.py (h=256, 12k
steps, the known-good MIP budget). Trained on K demos at fixed anchors,
rolled out closed-loop on fresh g ~ U(-15,15). d = distance to the
training bundle (mm)."""
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn

dev = "cuda" if torch.cuda.is_available() else "cpu"
KP, CAP = 0.3, 4.0
A_X, S_PT, I_PT = 60.0, np.array([120.0, 0.0, 0.0]), np.array(
    [120.0, 0.0, -40.0])
EPS = 4.0


def servo(p, t):
    return np.clip(KP * (t - p), -CAP, CAP)


def collect(g):
    p = np.zeros(3)
    phase = 1
    rows = []
    for _ in range(400):
        t = (np.array([A_X, g, 0.0]) if phase == 1 else
             S_PT if phase == 2 else I_PT)
        a = servo(p, t)
        rows.append([*p, g, *a, phase])
        p = p + a
        if phase == 1 and np.linalg.norm(p - np.array([A_X, g, 0.0])) < 3:
            phase = 2
        elif phase == 2 and np.linalg.norm(p - S_PT) < 3:
            phase = 3
        if p[2] <= -38:
            break
    return np.array(rows)


def mlp(inp, out, h=256):
    return nn.Sequential(nn.Linear(inp, h), nn.SiLU(),
                         nn.Linear(h, h), nn.SiLU(), nn.Linear(h, out))


def make_models(K, seed, steps=12000):
    demos = [collect(g) for g in np.linspace(-15, 15, K)]
    data = np.concatenate(demos)
    S = torch.tensor(data[:, :4], dtype=torch.float32, device=dev)
    A = torch.tensor(data[:, 4:7], dtype=torch.float32, device=dev)
    S_MU, S_SD = S.mean(0), S.std(0) + 1e-6
    A_SD = A.std()

    def norm_s(s):
        return (s - S_MU) / S_SD

    out = {}
    for lam, name in [(0.0, "L2"), (3e-3, "L2+condreg")]:
        torch.manual_seed(seed)
        net = mlp(4, 3).to(dev)
        opt = torch.optim.AdamW(net.parameters(), lr=1e-3, weight_decay=1e-5)
        for _ in range(steps):
            idx = torch.randint(0, len(S), (256,), device=dev)
            s, a = S[idx], A[idx]
            pred = net(norm_s(s))
            loss = ((pred - a) ** 2).mean()
            if lam > 0:
                d2s = []
                for _ in range(6):
                    v = torch.randn_like(s)
                    v = v / (v.norm(dim=1, keepdim=True) + 1e-9) * 2.0
                    d2s.append(((net(norm_s(s + v)) - pred) ** 2).mean(dim=1))
                D = torch.stack(d2s, 1)
                cv2 = D.var(dim=1) / (D.mean(dim=1) ** 2 + 1e-12)
                loss = loss + lam * torch.relu(cv2 - 1.0).mean()
            opt.zero_grad()
            loss.backward()
            opt.step()

        def fwd(s, net=net, norm_s=norm_s):
            return net(norm_s(s))
        out[name] = fwd

    torch.manual_seed(seed)
    net = mlp(7, 3).to(dev)
    opt = torch.optim.AdamW(net.parameters(), lr=1e-3, weight_decay=1e-5)
    for _ in range(steps):
        idx = torch.randint(0, len(S), (256,), device=dev)
        s, a = S[idx], A[idx]
        z = torch.randn_like(a) * A_SD
        y0 = net(torch.cat([z / A_SD, norm_s(s)], 1))
        y1 = net(torch.cat([y0.detach() / A_SD, norm_s(s)], 1))
        loss = ((y0 - a) ** 2).mean() + ((y1 - a) ** 2).mean()
        opt.zero_grad()
        loss.backward()
        opt.step()

    def fwd_mip(s, net=net, norm_s=norm_s, A_SD=A_SD):
        z = torch.randn(s.shape[0], 3, device=dev)
        y0 = net(torch.cat([z, norm_s(s)], 1))
        return net(torch.cat([y0 / A_SD, norm_s(s)], 1))
    out["MIP"] = fwd_mip

    bundle = np.concatenate([d[:, :3] for d in demos])
    return out, bundle


def rollout_nat(f, g, nmax=420):
    p = np.zeros(3)
    tr = [p.copy()]
    while (len(tr) < nmax and p[2] > -38 and abs(p[1]) < 200
           and -80 < p[0] < 300 and p[2] < 80):
        s = torch.tensor([[*p, g]], dtype=torch.float32, device=dev)
        with torch.no_grad():
            a = f(s)[0].cpu().numpy()
        p = p + a
        tr.append(p.copy())
    succ = (p[2] <= -38 and np.linalg.norm(p[:2] - S_PT[:2]) < EPS)
    return succ, np.array(tr)


import os
FIGONLY = bool(os.environ.get("FIGONLY"))
rng = np.random.RandomState(7)
G_TEST = rng.uniform(-15, 15, 100)
KS = [8] if FIGONLY else [8, 15, 30]
SEEDS = [1] if FIGONLY else [1, 2, 3]
results = {}
tr_store = {}
for K in KS:
    for sd in SEEDS:
        models, bundle = make_models(K, sd)
        for name, f in models.items():
            srs, maxds, trs = [], [], []
            for g in G_TEST:
                succ, tr = rollout_nat(f, g)
                d = np.sqrt(((bundle[None] - tr[:, None]) ** 2).sum(2)
                            ).min(1)
                srs.append(succ)
                maxds.append(d.max())
                trs.append((succ, tr, d))
            key = (name, K, sd)
            results[key] = (np.mean(srs), np.mean(np.array(maxds) > 4),
                            np.percentile(maxds, 90))
            if sd == SEEDS[0]:
                tr_store[(name, K)] = (trs, bundle)
            print(f"K={K:2d} seed={sd} {name:11s} SR={np.mean(srs):.2f} "
                  f"frac(maxd>4mm)={np.mean(np.array(maxds) > 4):.2f} "
                  f"maxd_p90={np.percentile(maxds, 90):6.1f}mm", flush=True)

if not FIGONLY:
    print("\n=== mean over 3 seeds (100 fresh anchors each) ===")
    print(f"{'model':12s}" + "".join(f"  K={K}: SR/deep/p90" for K in KS))
    for name in ["L2", "L2+condreg", "MIP"]:
        row = f"{name:12s}"
        for K in KS:
            v = np.mean([results[(name, K, sd)] for sd in SEEDS], axis=0)
            row += f"  {v[0]:.2f}/{v[1]:.2f}/{v[2]:5.1f}"
        print(row)

# figure: 2D side view (x,z), the readable format — 5 rollouts/panel
Kf = 8
SHOW = 5


def trunc(tr, d):
    off = np.where(d > 25.0)[0]
    cut = min(int(off[0]) + 5, len(tr)) if len(off) else len(tr)
    return tr[:cut], d[:cut]


fig, axes = plt.subplots(1, 3, figsize=(16.5, 4.6), sharey=True)
for j, name in enumerate(["L2", "L2+condreg", "MIP"]):
    trs, bundle = tr_store[(name, Kf)]
    idx = np.argsort(G_TEST)[::len(G_TEST) // SHOW][:SHOW]
    ax = axes[j]
    for gd in np.linspace(-15, 15, Kf):
        d0 = collect(gd)
        ax.plot(d0[:, 0], d0[:, 2], color="gray", lw=1.6, alpha=0.35,
                zorder=1)
    nS = sum(s for s, _, _ in trs)
    for i in idx:
        succ, tr, d = trs[i]
        tr, d = trunc(tr, d)
        if succ:
            ax.plot(tr[:, 0], tr[:, 2], color="tab:green", lw=1.7,
                    alpha=0.85, zorder=2)
            ax.plot(tr[-1, 0], tr[-1, 2], marker="*", color="tab:green",
                    ms=13, zorder=4, ls="none")
        else:
            ax.plot(tr[:, 0], tr[:, 2], color="tab:red", lw=1.9,
                    alpha=0.95, zorder=3)
            ax.plot(tr[-1, 0], tr[-1, 2], marker="x", color="red", ms=11,
                    mew=2.5, zorder=4, ls="none")
    ax.set_xlim(-8, 185)
    ax.set_ylim(-55, 30)
    ax.set_xlabel("x (mm)")
    if j == 0:
        ax.set_ylabel("z (mm)")
    ax.set_title(f"{name} — SR {nS}/100 on fresh anchors", fontsize=12)
    ax.grid(alpha=0.25)
fig.suptitle(
    f"Toy, natural error only, side view (x, z): K={Kf} demos (gray), "
    "fresh test anchors, no kick. 5 of 100 rollouts shown; failures cut "
    "shortly after leaving. * insertion, x failure.", fontsize=12.5,
    y=1.0)
fig.tight_layout()
fig.savefig("analysis/paper/toy_natural_fig.png", dpi=150,
            bbox_inches="tight")
print("saved analysis/paper/toy_natural_fig.png")
