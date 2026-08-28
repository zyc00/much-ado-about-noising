"""Causal-closure controls for the tube-funnel toy (dir + speed).

Arms per session, seeds 0-3, settled harness:
  ht    : trains first; its frozen sigma(x) supplies the transplant map
  l2sw  : plain MSE with FIXED per-sample weights 1/sigma(x)^2 from the
          seed-matched ht (clipped [0.02, 20], mean-normalized) — the
          sigma-transplant: repricing without the t tail
  l2ow  : plain MSE with oracle band weights (noise band x 0.05) —
          suppression-alone control
Interpretation gates (same as the toyhuman canonical closure):
  l2sw recovers ht band  => mechanism = sigma-map repricing (causal)
  l2ow does NOT recover  => suppression alone insufficient; the map's
                            relative amplification carries the effect
"""
import json
import os

os.environ.setdefault("HW_DOCK", "0.004")
os.environ.setdefault("WIDTH", "256")
os.environ.setdefault("STEPS", "50000")
os.environ.setdefault("NEP", "160")
os.environ.setdefault("AS", "1")

import numpy as np
import torch

import toytube as T

SEEDS = [0, 1, 2, 3]
OW = 0.05


def train_weighted(seed, Xn, Yn, w_all):
    torch.manual_seed(seed)
    net = T.Reg(hetero=False).to(T.DEV)
    opt = torch.optim.Adam(net.parameters(), lr=1e-3)
    Xt = torch.tensor(Xn, device=T.DEV)
    Yt = torch.tensor(Yn, device=T.DEV)
    Wt = torch.tensor(w_all.astype(np.float32), device=T.DEV)
    rng = np.random.RandomState(seed)
    for it in range(T.STEPS):
        idx = rng.randint(0, len(Xt), 256)
        per = ((net(Xt[idx]) - Yt[idx]) ** 2).mean(1)
        loss = (per * Wt[idx]).mean()
        opt.zero_grad()
        loss.backward()
        opt.step()
    return net


def main():
    results = {}
    for session in ("dir", "speed"):
        T.NOISE_TYPE = session
        for seed in SEEDS:
            drng = np.random.RandomState(1000 + seed)
            X, Y, Yc, AL = T.build_dataset(drng)
            norm = T.Norm(X, Y)
            Xn, Yn = norm.nx(X), norm.ny(Y)
            Xt = torch.tensor(Xn, device=T.DEV)

            ht = T.train("ht", seed, Xn, Yn)
            ev = T.evaluate(ht, norm, n=100, seed0=seed)
            results[f"{session}_ht_s{seed}"] = ev["success"]
            with torch.no_grad():
                sig = T.sigma_of(ht, Xt).cpu().numpy()
            w = 1.0 / sig ** 2
            w = np.clip(w / w.mean(), 0.02, 20.0)
            w = w / w.mean()

            sw = train_weighted(seed, Xn, Yn, w)
            ev_sw = T.evaluate(sw, norm, n=100, seed0=seed)
            results[f"{session}_l2sw_s{seed}"] = ev_sw["success"]

            wow = np.where(AL, OW, 1.0)
            wow = wow / wow.mean()
            ow = train_weighted(seed, Xn, Yn, wow)
            ev_ow = T.evaluate(ow, norm, n=100, seed0=seed)
            results[f"{session}_l2ow_s{seed}"] = ev_ow["success"]

            print(f"{session} s{seed}: ht {ev['success']:.2f} | "
                  f"l2sw {ev_sw['success']:.2f} | l2ow {ev_ow['success']:.2f}",
                  flush=True)
    json.dump(results, open("tube_controls.json", "w"), indent=1)
    print("\n=== TABLE (mean±sd) ===")
    for session in ("dir", "speed"):
        for arm in ("ht", "l2sw", "l2ow"):
            v = [results[f"{session}_{arm}_s{s}"] for s in SEEDS]
            print(f"{session} {arm:>5}: {np.mean(v):.2f}±{np.std(v):.2f}")


if __name__ == "__main__":
    main()
