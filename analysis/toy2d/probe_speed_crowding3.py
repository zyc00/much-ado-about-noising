"""Speed-cell interference probe v2 — exact-counterfactual version.

v1 verified E1 (l2's corridor loss == kNN floor: it predicts the average
pace) but its clean-region endpoint was confounded (mu_hat estimator
error differs between noisy and clean labels).

v2 removes the confound: states and clean labels are held FIXED; the
empirical pace residuals (Y - Yc, corridor samples, per-dim centered)
are PERMUTED across corridor samples and added to Yc there. So
E[Ytil | x] = Yc exactly, everywhere, and fit-vs-Yc is unbiased.

arms   : l2b = l2 on Yc            (no noise, baseline)
         l2t = l2 on Ytil          (mean-zero heavy-tailed corridor noise)
         htt = ht on Ytil
bands  : transit y>=0.55 | prealign 0.5<=y<0.55 | align 0.25<=y<0.5
         | insert 0.05<=y<0.25 | dock y<0.05
endpoints:
  P2 (primary): fit-vs-Yc on TRANSIT (and per band): l2t > l2b means the
     corridor mixture damaged optimization elsewhere (interference).
     htt vs l2t shows whether repricing removes it.
  E2: per-band gradient-mass share vs sample share (crowding factor).
  E1 check: l2t corridor loss -> injected noise variance.
"""
import json
import os

os.environ.setdefault("NOISE_TYPE", "speed3")
os.environ.setdefault("MOM", "0.5")
os.environ.setdefault("SCRAPE", "1")
os.environ.setdefault("T1S3", "1.2")
os.environ.setdefault("PAUSE_P", "0.12")
os.environ.setdefault("WSCALE", "0.7")
os.environ.setdefault("NEP", "40")
os.environ.setdefault("STEPS", "12000")

import numpy as np
import torch

import toyhuman as T

SEEDS = [0, 1, 2, 3]
CKPTS = [250, 500, 1000, 2000, 4000, 8000, 12000]
BANDS = [("transit", 0.55, 9.9), ("prealign", 0.5, 0.55),
         ("align", 0.25, 0.5), ("insert", 0.05, 0.25), ("dock", -9.9, 0.05)]


def band_masks(X):
    return {n: (X[:, 1] >= lo) & (X[:, 1] < hi) for n, lo, hi in BANDS}


def run_seed(seed, out):
    drng = np.random.RandomState(1000 + seed)
    X, Y, Yc, AL = T.build_dataset(drng)
    norm = T.Norm(X, Y)
    Xn, Yn, Ycn = norm.nx(X), norm.ny(Y), norm.ny(Yc)
    masks = band_masks(X)
    corridor = X[:, 1] < 0.55
    # mean-zero permuted real residuals on corridor only
    prng = np.random.RandomState(7000 + seed)
    R = (Yn - Ycn)[corridor]
    R = R - R.mean(0, keepdims=True)
    Ytil = Ycn.copy()
    Ytil[corridor] = Ycn[corridor] + R[prng.permutation(len(R))]
    noise_var = float((R ** 2).mean())
    rec = {"n": int(len(X)), "noise_var_corridor": noise_var,
           "sample_share": {n: float(m.mean()) for n, m in masks.items()},
           "band_amp": {n: float(Ycn[m].var()) for n, m in masks.items()}}
    Xt = torch.tensor(Xn, device=T.DEV)
    Yct = torch.tensor(Ycn, device=T.DEV)
    Ytt = torch.tensor(Ytil.astype(np.float32), device=T.DEV)
    for arm, lab, labels_t in (("l2", "l2b", Yct), ("l2", "l2t", Ytt),
                               ("ht", "htt", Ytt)):
        torch.manual_seed(seed)
        net = T.Reg(hetero=(arm == "ht")).to(T.DEV)
        opt = torch.optim.Adam(net.parameters(), lr=1e-3)
        rng = np.random.RandomState(seed)
        curves = []
        for it in range(T.STEPS):
            idx = rng.randint(0, len(Xt), 256)
            loss = T.loss_fn(arm, net, Xt[idx], labels_t[idx])
            opt.zero_grad()
            loss.backward()
            opt.step()
            if it + 1 in CKPTS:
                with torch.no_grad():
                    pred = net(Xt).cpu().numpy()
                curves.append({"it": it + 1, "fitYc": {
                    n: float(((pred[m] - Ycn[m]) ** 2).mean())
                    for n, m in masks.items()}})
        with torch.no_grad():
            pred = net(Xt).cpu().numpy()
            w = T.weight_of(arm, net, Xt, labels_t).cpu().numpy()
        r = {"curves": curves,
             "fitYc": {n: float(((pred[m] - Ycn[m]) ** 2).mean())
                       for n, m in masks.items()},
             "loss_corridor": float(((pred[corridor]
                                      - (Ytil if lab != "l2b" else Ycn)[corridor]) ** 2).mean()),
             "gshare": {n: float(w[m].sum() / w.sum()) for n, m in masks.items()}}
        if arm == "ht":
            with torch.no_grad():
                sig = T.sigma_of(net, Xt).cpu().numpy()
            r["sig_band"] = {n: float(sig[m].mean()) for n, m in masks.items()}
        out[f"{lab}_s{seed}"] = r
        print(f"s{seed} {lab}: fitYc " +
              " ".join(f"{n} {r['fitYc'][n]:.5f}" for n, *_ in BANDS) +
              f" | lossCorr {r['loss_corridor']:.4f}", flush=True)
    out[f"data_s{seed}"] = rec
    print(f"s{seed} data: noiseVar {noise_var:.4f} shares " +
          " ".join(f"{n} {rec['sample_share'][n]:.3f}" for n, *_ in BANDS),
          flush=True)


def main():
    out = {}
    for s in SEEDS:
        run_seed(s, out)
    json.dump(out, open(f"probe_speed_crowding3_w{T.WIDTH}.json", "w"), indent=1)
    print("\n=== SUMMARY (mean±sd over seeds) ===")
    nv = np.mean([out[f"data_s{s}"]["noise_var_corridor"] for s in SEEDS])
    print(f"injected corridor noise var {nv:.4f}")
    print("sample share: " + " ".join(
        f"{n} {np.mean([out[f'data_s{s}']['sample_share'][n] for s in SEEDS]):.3f}"
        for n, *_ in BANDS))
    for lab in ("l2b", "l2t", "htt"):
        for n, *_ in BANDS:
            v = np.array([out[f"{lab}_s{s}"]["fitYc"][n] for s in SEEDS])
            print(f"{lab} fitYc {n}: {v.mean():.5f}±{v.std():.5f}")
        g = {n: np.mean([out[f"{lab}_s{s}"]["gshare"][n] for s in SEEDS])
             for n, *_ in BANDS}
        lc = np.mean([out[f"{lab}_s{s}"]["loss_corridor"] for s in SEEDS])
        print(f"{lab} gshare: " + " ".join(f"{n} {g[n]:.3f}" for n, *_ in BANDS)
              + f" | lossCorr {lc:.4f}")


if __name__ == "__main__":
    main()
