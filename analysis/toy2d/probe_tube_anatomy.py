"""Loss & gradient anatomy for the tube-funnel toy (dir + speed sessions).

Per session (settled harness: w256, 50k, NEP160, AS=1):
  arms l2, ht; seeds 0,1. Logged at checkpoints:
   - l2 per-band train loss (noise band / insert band / rest) + kNN
     noise floor of the noise band (irreducible aleatoric level)
   - gradient-mass share on the noise band (weight_of ledger) l2 vs ht
   - insert-band fit |pred - Y| (clean precision content) l2 vs ht
   - ht sigma ratio noise/clean (sigma-map formation)
Figure: tube_anatomy_fig.png (2 sessions x 4 panels).
"""
import json
import os

os.environ.setdefault("HW_DOCK", "0.004")
os.environ.setdefault("WIDTH", "256")
os.environ.setdefault("STEPS", "50000")
os.environ.setdefault("NEP", "160")
os.environ.setdefault("AS", "1")

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch

import toytube as T

SEEDS = [0, 1]
CKPTS = [1000, 3000, 6000, 12000, 25000, 50000]


def knn_floor(Xn, Yn, mask, k=10):
    Xm, Ym = Xn[mask], Yn[mask]
    if len(Xm) > 6000:
        idx = np.random.RandomState(0).choice(len(Xm), 6000, replace=False)
        Xm, Ym = Xm[idx], Ym[idx]
    d2 = ((Xm[:, None, :] - Xm[None, :, :]) ** 2).sum(-1)
    nn = np.argsort(d2, 1)[:, 1:k + 1]  # exclude self
    mu = Ym[nn].mean(1)
    return float(((Ym - mu) ** 2).mean())


def run(session):
    T.NOISE_TYPE = session
    out = {}
    for seed in SEEDS:
        drng = np.random.RandomState(1000 + seed)
        X, Y, Yc, AL = T.build_dataset(drng)
        norm = T.Norm(X, Y)
        Xn, Yn = norm.nx(X), norm.ny(Y)
        ins = (X[:, 1] > T.Y_DOCK) & (X[:, 1] <= T.Y_FUN_LO)
        rest = ~AL & ~ins
        floor = knn_floor(Xn, Yn, AL)
        floor_ins = knn_floor(Xn, Yn, ins)
        Xt = torch.tensor(Xn, device=T.DEV)
        Yt = torch.tensor(Yn, device=T.DEV)
        for arm in ("l2", "ht"):
            torch.manual_seed(seed)
            net = T.Reg(hetero=(arm == "ht")).to(T.DEV)
            opt = torch.optim.Adam(net.parameters(), lr=1e-3)
            rng = np.random.RandomState(seed)
            curves = []
            for it in range(T.STEPS):
                bidx = rng.randint(0, len(Xt), 256)
                loss = T.loss_fn(arm, net, Xt[bidx], Yt[bidx])
                opt.zero_grad()
                loss.backward()
                opt.step()
                if it + 1 in CKPTS:
                    with torch.no_grad():
                        pred = net(Xt).cpu().numpy()
                        w = T.weight_of(arm, net, Xt, Yt).cpu().numpy()
                    c = {"it": it + 1,
                         "mseN": float(((pred[AL] - Yn[AL]) ** 2).mean()),
                         "mseI": float(((pred[ins] - Yn[ins]) ** 2).mean()),
                         "mseR": float(((pred[rest] - Yn[rest]) ** 2).mean()),
                         "gshareN": float(w[AL].sum() / (w.sum() + 1e-12)),
                         "fitI": float(np.abs(pred[ins] - Yn[ins]).mean())}
                    if arm == "ht":
                        with torch.no_grad():
                            sig = T.sigma_of(net, Xt).cpu().numpy()
                        c["sigR"] = float(sig[AL].mean() / sig[~AL].mean())
                    curves.append(c)
            out[f"{arm}_s{seed}"] = curves
            last = curves[-1]
            print(f"{session} {arm} s{seed}: mseN {last['mseN']:.4f} "
                  f"(floor {floor:.4f}) mseI {last['mseI']:.5f} "
                  f"(floor {floor_ins:.5f}) gshareN {last['gshareN']:.2f} "
                  f"fitI {last['fitI']:.4f}"
                  + (f" sigR {last['sigR']:.1f}" if arm == "ht" else ""),
                  flush=True)
        out[f"data_s{seed}"] = {"floorN": floor, "floorI": floor_ins,
                                "shareN": float(AL.mean()),
                                "shareI": float(ins.mean())}
    return out


def main():
    res = {s: run(s) for s in ("dir", "speed")}
    json.dump(res, open("tube_anatomy.json", "w"), indent=1)

    fig, axes = plt.subplots(2, 4, figsize=(17, 8))
    for row, session in enumerate(("dir", "speed")):
        r = res[session]
        its = [c["it"] for c in r["l2_s0"]]
        fl = np.mean([r[f"data_s{s}"]["floorN"] for s in SEEDS])

        ax = axes[row, 0]
        for band, lab in (("mseN", "noise band"), ("mseI", "insert band"),
                          ("mseR", "rest")):
            m = np.mean([[c[band] for c in r[f"l2_s{s}"]] for s in SEEDS], 0)
            ax.plot(its, m, "-o", ms=3, label=lab)
        ax.axhline(fl, color="k", ls="--", lw=0.8, label="kNN noise floor")
        ax.set_yscale("log")
        ax.set_xscale("log")
        ax.set_title(f"{session}: l2 per-band loss", fontsize=10)
        ax.legend(fontsize=7)

        ax = axes[row, 1]
        for arm, col in (("l2", "tab:red"), ("ht", "tab:purple")):
            m = np.mean([[c["gshareN"] for c in r[f"{arm}_s{s}"]] for s in SEEDS], 0)
            ax.plot(its, m, "-o", ms=3, color=col, label=arm)
        ax.axhline(np.mean([r[f"data_s{s}"]["shareN"] for s in SEEDS]),
                   color="k", ls=":", lw=0.8, label="sample share")
        ax.set_xscale("log")
        ax.set_ylim(0, 1)
        ax.set_title("gradient share on noise band", fontsize=10)
        ax.legend(fontsize=7)

        ax = axes[row, 2]
        for arm, col in (("l2", "tab:red"), ("ht", "tab:purple")):
            m = np.mean([[c["fitI"] for c in r[f"{arm}_s{s}"]] for s in SEEDS], 0)
            ax.plot(its, m, "-o", ms=3, color=col, label=arm)
        ax.set_xscale("log")
        ax.set_title("insert-band fit |err| (clean content)", fontsize=10)
        ax.legend(fontsize=7)

        ax = axes[row, 3]
        m = np.mean([[c["sigR"] for c in r[f"ht_s{s}"]] for s in SEEDS], 0)
        ax.plot(its, m, "-o", ms=3, color="tab:purple")
        ax.axhline(1.0, color="k", ls=":", lw=0.8)
        ax.set_xscale("log")
        ax.set_title("ht sigma ratio noise/clean (map formation)", fontsize=10)
        for ax2 in axes[row]:
            ax2.set_xlabel("train step")
            ax2.tick_params(labelsize=8)
    fig.suptitle("tube-funnel: loss & gradient anatomy (dir top, speed bottom; "
                 "mean of 2 seeds, settled harness)", fontsize=12)
    fig.tight_layout(rect=[0, 0, 1, 0.94])
    fig.savefig("tube_anatomy_fig.png", dpi=150)
    print("wrote tube_anatomy_fig.png")


if __name__ == "__main__":
    main()
