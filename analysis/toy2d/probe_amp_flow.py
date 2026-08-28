"""Two follow-ups on the rebalancing exhibit (user directives):
A. SPEED AMPLIFICATION: does the precision-band loss dividend emerge when
   the pace divergence grows? T1S3 1.2 (ref, already measured) vs 2.0 with
   PAUSE_P 0.25. l2/ht, seeds 0-1, grad share + insert train/val.
B. FLOW/MIP REBALANCING: the anchor-absorption prediction — flow's
   velocity-residual share on the noise band must be far below l2's
   endpoint-residual share, and its insert endpoint loss must land at/below
   ht's. flow field at tuned config (w512/150k), dir + speed sessions,
   seeds 0-1; endpoint loss reported at ns=32 (flow) and ns=2 (mip).
"""
import json
import os

os.environ.setdefault("HW_DOCK", "0.004")
os.environ.setdefault("NEP", "160")
os.environ.setdefault("AS", "1")

import numpy as np
import torch

import toytube as T

CKPTS_REG = [1000, 6000, 25000, 50000]


def bands(X, AL):
    ins = (X[:, 1] > T.Y_DOCK) & (X[:, 1] <= T.Y_FUN_LO)
    return {"noise": AL, "insert": ins}


def make_data(seed):
    X, Y, Yc, AL = T.build_dataset(np.random.RandomState(1000 + seed))
    norm = T.Norm(X, Y)
    old = T.NEP
    T.NEP = 40
    Xv, Yv, _, ALv = T.build_dataset(np.random.RandomState(5000 + seed))
    T.NEP = old
    return X, Y, AL, Xv, Yv, ALv, norm


def part_a():
    print("== A: speed amplification (T1S3=2.0, PAUSE_P=0.25) ==", flush=True)
    T.NOISE_TYPE = "speed"
    T.WIDTH = 256
    T.STEPS = 50000
    T.T1S3 = 2.0
    T.PAUSE_P = 0.25
    out = {}
    for seed in (0, 1):
        X, Y, AL, Xv, Yv, ALv, norm = make_data(seed)
        tb, vb = bands(X, AL), bands(Xv, ALv)
        Xn, Yn = norm.nx(X), norm.ny(Y)
        Xvn, Yvn = norm.nx(Xv), norm.ny(Yv)
        Xt = torch.tensor(Xn, device=T.DEV)
        Yt = torch.tensor(Yn, device=T.DEV)
        Xvt = torch.tensor(Xvn, device=T.DEV)
        for arm in ("l2", "ht"):
            torch.manual_seed(seed)
            net = T.Reg(hetero=(arm == "ht")).to(T.DEV)
            opt = torch.optim.Adam(net.parameters(), lr=1e-3)
            rng = np.random.RandomState(seed)
            curves = []
            for it in range(T.STEPS):
                idx = rng.randint(0, len(Xt), 256)
                loss = T.loss_fn(arm, net, Xt[idx], Yt[idx])
                opt.zero_grad()
                loss.backward()
                opt.step()
                if it + 1 in CKPTS_REG:
                    with torch.no_grad():
                        pr = net(Xt).cpu().numpy()
                        pv = net(Xvt).cpu().numpy()
                        w = T.weight_of(arm, net, Xt, Yt).cpu().numpy()
                    curves.append({
                        "it": it + 1,
                        "gsN": float(w[tb["noise"]].sum() / (w.sum() + 1e-12)),
                        "trI": float(((pr[tb["insert"]] - Yn[tb["insert"]]) ** 2).mean()),
                        "vaI": float(((pv[vb["insert"]] - Yvn[vb["insert"]]) ** 2).mean())})
            out[f"{arm}_s{seed}"] = curves
            f = curves[-1]
            print(f"ampspeed {arm} s{seed}: gsN {f['gsN']:.2f} trI {f['trI']:.5f} "
                  f"vaI {f['vaI']:.5f}", flush=True)
    json.dump(out, open("probe_ampspeed.json", "w"), indent=1)


def flow_grad_share(net, Xt, Yt, mask, nsamp=8):
    """true flow-matching per-sample gradient ledger: mean |velocity residual|
    over sampled (eps, t) draws."""
    with torch.no_grad():
        acc = torch.zeros(len(Xt), device=T.DEV)
        for _ in range(nsamp):
            eps = torch.randn_like(Yt)
            t = torch.rand(len(Yt), 1, device=T.DEV)
            yt = (1 - t) * eps + t * Yt
            acc += (net.vel(Xt, yt, t) - (Yt - eps)).abs().mean(1)
        w = (acc / nsamp).cpu().numpy()
    return float(w[mask].sum() / (w.sum() + 1e-12))


def part_b():
    print("== B: flow/mip training-side anatomy (w512, 150k) ==", flush=True)
    T.T1S3 = 1.2
    T.PAUSE_P = 0.12
    T.WIDTH = 512
    T.STEPS = 150000
    out = {}
    for session in ("dir", "speed"):
        T.NOISE_TYPE = session
        for seed in (0, 1):
            X, Y, AL, Xv, Yv, ALv, norm = make_data(seed)
            tb, vb = bands(X, AL), bands(Xv, ALv)
            Xn, Yn = norm.nx(X), norm.ny(Y)
            Xvn, Yvn = norm.nx(Xv), norm.ny(Yv)
            Xt = torch.tensor(Xn, device=T.DEV)
            Yt = torch.tensor(Yn, device=T.DEV)
            Xvt = torch.tensor(Xvn, device=T.DEV)
            torch.manual_seed(seed)
            net = T.FlowNet(32).to(T.DEV)
            opt = torch.optim.Adam(net.parameters(), lr=1e-3)
            rng = np.random.RandomState(seed)
            curves = []
            for it in range(T.STEPS):
                idx = rng.randint(0, len(Xt), 256)
                loss = T.loss_fn("flow8", net, Xt[idx], Yt[idx])
                opt.zero_grad()
                loss.backward()
                opt.step()
                if it + 1 in (3000, 25000, 75000, 150000):
                    gs = flow_grad_share(net, Xt, Yt, tb["noise"])
                    with torch.no_grad():
                        net.nsteps = 32
                        pv32 = net(Xvt).cpu().numpy()
                        net.nsteps = 2
                        pv2 = net(Xvt).cpu().numpy()
                        net.nsteps = 32
                    curves.append({
                        "it": it + 1, "gsN_vel": gs,
                        "vaI_32": float(((pv32[vb["insert"]] - Yvn[vb["insert"]]) ** 2).mean()),
                        "vaI_2": float(((pv2[vb["insert"]] - Yvn[vb["insert"]]) ** 2).mean())})
            out[f"{session}_flow_s{seed}"] = curves
            f = curves[-1]
            print(f"{session} flow s{seed}: gsN(vel) {f['gsN_vel']:.2f} | "
                  f"vaI ns32 {f['vaI_32']:.5f} ns2 {f['vaI_2']:.5f}", flush=True)
    json.dump(out, open("probe_flowanat.json", "w"), indent=1)


if __name__ == "__main__":
    part_a()
    part_b()
