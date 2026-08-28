"""v2.3 gradient + train/VAL loss anatomy per band (the human-data signature):
suppressing the noise band's loss should give the OTHER bands larger gradient
share and smaller train AND validation loss under ht than under l2.

Sessions zturn/dir/speed, arms l2/ht, seeds 0-1. Held-out val set = 40 fresh
episodes (disjoint rng). Bands: noise (session's own), insert (clean precision),
rest (remaining clean). Logged at checkpoints: per-band gradient-weight share,
per-band train MSE, per-band val MSE (mean prediction vs labels).
"""
import json
import os

for k, v in [("HW_DOCK", "0.004"), ("WIDTH", "256"), ("STEPS", "50000"),
             ("NEP", "160"), ("AS", "1")]:
    os.environ.setdefault(k, v)

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch

import toytube as T

SEEDS = [0, 1]
CKPTS = [1000, 3000, 6000, 12000, 25000, 50000]
SESSIONS = ("zturn", "dir", "speed")


def bands(X, AL):
    ins = (X[:, 1] > T.Y_DOCK) & (X[:, 1] <= T.Y_FUN_LO)
    rest = ~AL & ~ins
    return {"noise": AL, "insert": ins, "rest": rest}


def run(session):
    T.NOISE_TYPE = session
    out = {}
    for seed in SEEDS:
        X, Y, Yc, AL = T.build_dataset(np.random.RandomState(1000 + seed))
        norm = T.Norm(X, Y)
        Xn, Yn = norm.nx(X), norm.ny(Y)
        # held-out validation episodes (disjoint rng stream)
        old_nep = T.NEP
        T.NEP = 40
        Xv, Yv, Ycv, ALv = T.build_dataset(np.random.RandomState(5000 + seed))
        T.NEP = old_nep
        Xvn, Yvn = norm.nx(Xv), norm.ny(Yv)
        tb, vb = bands(X, AL), bands(Xv, ALv)
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
                if it + 1 in CKPTS:
                    with torch.no_grad():
                        pr = net(Xt).cpu().numpy()
                        pv = net(Xvt).cpu().numpy()
                        w = T.weight_of(arm, net, Xt, Yt).cpu().numpy()
                    c = {"it": it + 1}
                    for b, m in tb.items():
                        c[f"tr_{b}"] = float(((pr[m] - Yn[m]) ** 2).mean())
                        c[f"gs_{b}"] = float(w[m].sum() / (w.sum() + 1e-12))
                    for b, m in vb.items():
                        c[f"va_{b}"] = float(((pv[m] - Yvn[m]) ** 2).mean())
                    curves.append(c)
            out[f"{arm}_s{seed}"] = curves
            f = curves[-1]
            print(f"{session} {arm} s{seed}: gs(noise/ins/rest) "
                  f"{f['gs_noise']:.2f}/{f['gs_insert']:.2f}/{f['gs_rest']:.2f} | "
                  f"tr_ins {f['tr_insert']:.5f} va_ins {f['va_insert']:.5f} | "
                  f"va_rest {f['va_rest']:.5f}", flush=True)
        out[f"share_s{seed}"] = {b: float(m.mean()) for b, m in tb.items()}
    return out


def main():
    res = {s: run(s) for s in SESSIONS}
    json.dump(res, open("tube_anatomy2.json", "w"), indent=1)

    fig, axes = plt.subplots(3, 3, figsize=(14, 12))
    for row, session in enumerate(SESSIONS):
        r = res[session]
        its = [c["it"] for c in r["l2_s0"]]

        ax = axes[row, 0]
        for arm, c in (("l2", "tab:red"), ("ht", "tab:purple")):
            for b, ls in (("noise", "-"), ("insert", "--")):
                m = np.mean([[x[f"gs_{b}"] for x in r[f"{arm}_s{s}"]]
                             for s in SEEDS], 0)
                ax.plot(its, m, ls, color=c, marker="o", ms=2.5,
                        label=f"{arm} {b}")
        ax.set_xscale("log")
        ax.set_title(f"{session}: gradient share\n(solid=noise band, dashed=insert)",
                     fontsize=9)
        ax.legend(fontsize=6)

        ax = axes[row, 1]
        for arm, c in (("l2", "tab:red"), ("ht", "tab:purple")):
            m = np.mean([[x["tr_insert"] for x in r[f"{arm}_s{s}"]]
                         for s in SEEDS], 0)
            ax.plot(its, m, "-o", ms=2.5, color=c, label=f"{arm} train")
            m = np.mean([[x["va_insert"] for x in r[f"{arm}_s{s}"]]
                         for s in SEEDS], 0)
            ax.plot(its, m, "--s", ms=2.5, color=c, label=f"{arm} val")
        ax.set_xscale("log")
        ax.set_yscale("log")
        ax.set_title("insert band (clean precision):\ntrain + val loss", fontsize=9)
        ax.legend(fontsize=6)

        ax = axes[row, 2]
        for arm, c in (("l2", "tab:red"), ("ht", "tab:purple")):
            m = np.mean([[x["va_rest"] for x in r[f"{arm}_s{s}"]]
                         for s in SEEDS], 0)
            ax.plot(its, m, "--s", ms=2.5, color=c, label=f"{arm} val rest")
        ax.set_xscale("log")
        ax.set_yscale("log")
        ax.set_title("other clean areas: val loss", fontsize=9)
        ax.legend(fontsize=6)
        for a in axes[row]:
            a.set_xlabel("step")
            a.tick_params(labelsize=7)
    fig.suptitle("v2.3 anatomy: suppression of the noise band -> larger clean-band "
                 "gradient share, lower clean train+val loss (mean of 2 seeds)",
                 fontsize=11)
    fig.tight_layout(rect=[0, 0, 1, 0.95])
    fig.savefig("tube_anatomy2_fig.png", dpi=150)
    print("wrote tube_anatomy2_fig.png")


if __name__ == "__main__":
    main()
