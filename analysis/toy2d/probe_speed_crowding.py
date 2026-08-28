"""Speed-cell loss-interference probe (SR-free).

Hypothesis under test (user, 2026-07): pace mixtures at similar states
force l2 to fit the average pace; the persistent large loss/gradient on
those states makes optimization of OTHER states harder (crowding).

Design (pre-registered):
  dataset : speed3 locked cell (MOM=.5 SCRAPE=1 T1S3=1.2 PAUSE_P=.12
            WSCALE=.7 NEP=40), seeds 0-3, 12k steps (declared budget).
  arms    : l2n = l2 on noisy labels Y
            l2c = l2 on clean reference Yc at the SAME states
                  (counterfactual: only the noise removed)
            htn = ht on noisy labels Y
  regions : NOISY  = corridor, y < Y_TOP+0.05 (pace eta + pauses there)
            CLEAN  = transit,  y >= Y_TOP+0.05 (no eta; only MOM comp)
  floor   : kNN (k=10, state space) conditional variance of the labels,
            per region, normalized space  == irreducible loss.
  mu_hat  : kNN conditional mean of the training labels == best
            achievable prediction; fit(pred, mu_hat) is optimization
            quality with the aleatoric part removed.
  endpoints (the interference prediction, all vs l2c):
    E1 l2n corridor loss converges to ~floor (mean-fitting).
    E2 grad-mass share of NOISY region for l2n >> its sample share;
       htn share ~ sample share or below (repricing).
    E3 PRIMARY: fit-vs-mu_hat on the CLEAN region: l2n > l2c means the
       noisy corridor damaged clean-region optimization (interference
       VERIFIED); l2n ~ l2c means no measurable crowding damage there.
    E4 corridor NORMAL-component error (unnormalized, per-step residual
       vs Yc projected off the clean step direction): l2n vs l2c vs htn.
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
CKPTS = [500, 1000, 2000, 3000, 6000, 9000, 12000]
K = 10
YSPLIT = T.Y_TOP + 0.05


def knn_stats(X, Y, k=K):
    """kNN conditional mean and per-sample residual variance (normalized)."""
    d2 = ((X[:, None, :] - X[None, :, :]) ** 2).sum(-1)
    idx = np.argsort(d2, 1)[:, :k]
    mu = Y[idx].mean(1)
    var = ((Y[idx] - mu[:, None, :]) ** 2).mean(1).mean(1)  # per-dim avg
    return mu.astype(np.float32), var.astype(np.float32)


def region_masks(X):
    noisy = X[:, 1] < YSPLIT
    return noisy, ~noisy


def fit_err(pred, target, mask):
    return float(((pred[mask] - target[mask]) ** 2).mean())


def tang_norm_err(pred_u, Yc, mask):
    """unnormalized per-step residual split along/off the clean direction."""
    P = pred_u[mask].reshape(-1, T.H, 2)
    C = Yc[mask].reshape(-1, T.H, 2)
    u = C / (np.linalg.norm(C, axis=-1, keepdims=True) + 1e-9)
    r = P - C
    tang = (r * u).sum(-1)
    norm = r[..., 0] * (-u[..., 1]) + r[..., 1] * u[..., 0]
    return float(np.abs(tang).mean()), float(np.abs(norm).mean())


def grad_share(arm, net, Xt, Yt, noisy_mask):
    w = T.weight_of(arm, net, Xt, Yt).cpu().numpy()
    return float(w[noisy_mask].sum() / (w.sum() + 1e-12))


def run_seed(seed, out):
    drng = np.random.RandomState(1000 + seed)
    X, Y, Yc, AL = T.build_dataset(drng)
    norm = T.Norm(X, Y)
    Xn, Yn, Ycn = norm.nx(X), norm.ny(Y), norm.ny(Yc)
    noisy, clean = region_masks(X)
    mu_n, var_n = knn_stats(Xn, Yn)
    mu_c, _ = knn_stats(Xn, Ycn)
    rec = {"n": int(len(X)), "sample_share_noisy": float(noisy.mean()),
           "floor_noisy": float(var_n[noisy].mean()),
           "floor_clean": float(var_n[clean].mean())}
    Xt = torch.tensor(Xn, device=T.DEV)
    Yt = torch.tensor(Yn, device=T.DEV)
    Yct = torch.tensor(Ycn, device=T.DEV)
    for arm, lab, labels_t, labels_n, mu in (
            ("l2", "l2n", Yt, Yn, mu_n),
            ("l2", "l2c", Yct, Ycn, mu_c),
            ("ht", "htn", Yt, Yn, mu_n)):
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
                curves.append({
                    "it": it + 1,
                    "loss_noisy": fit_err(pred, labels_n, noisy),
                    "loss_clean": fit_err(pred, labels_n, clean),
                    "muhat_noisy": fit_err(pred, mu, noisy),
                    "muhat_clean": fit_err(pred, mu, clean),
                    "gshare_noisy": grad_share(arm, net, Xt, labels_t, noisy),
                })
        with torch.no_grad():
            pred = net(Xt).cpu().numpy()
        pred_u = norm.uy(pred)
        tN, nN = tang_norm_err(pred_u, Yc, noisy)
        tC, nC = tang_norm_err(pred_u, Yc, clean)
        r = {"curves": curves,
             "final": {"tang_noisy": tN, "norm_noisy": nN,
                       "tang_clean": tC, "norm_clean": nC}}
        if arm == "ht":
            with torch.no_grad():
                sig = T.sigma_of(net, Xt).cpu().numpy()
            r["sig_ratio"] = float(sig[noisy].mean() / sig[clean].mean())
        out[f"{lab}_s{seed}"] = r
        rec[f"{lab}_done"] = True
        c = curves[-1]
        print(f"s{seed} {lab}: loss(noisy/clean) {c['loss_noisy']:.4f}/"
              f"{c['loss_clean']:.4f} | muhat(noisy/clean) "
              f"{c['muhat_noisy']:.5f}/{c['muhat_clean']:.5f} | "
              f"gshareN {c['gshare_noisy']:.2f} | normN {nN:.5f}", flush=True)
    out[f"data_s{seed}"] = rec
    print(f"s{seed} data: n {rec['n']} shareN {rec['sample_share_noisy']:.2f} "
          f"floor(noisy/clean) {rec['floor_noisy']:.4f}/{rec['floor_clean']:.4f}",
          flush=True)


def main():
    out = {}
    for s in SEEDS:
        run_seed(s, out)
    json.dump(out, open("probe_speed_crowding.json", "w"), indent=1)

    def agg(lab, fn):
        return np.array([fn(out[f"{lab}_s{s}"]) for s in SEEDS])

    print("\n=== SUMMARY (mean over seeds) ===")
    shareS = np.mean([out[f"data_s{s}"]["sample_share_noisy"] for s in SEEDS])
    floorN = np.mean([out[f"data_s{s}"]["floor_noisy"] for s in SEEDS])
    floorC = np.mean([out[f"data_s{s}"]["floor_clean"] for s in SEEDS])
    print(f"sample share noisy {shareS:.2f} | floor noisy {floorN:.4f} "
          f"clean {floorC:.4f}")
    for lab in ("l2n", "l2c", "htn"):
        lossN = agg(lab, lambda r: r["curves"][-1]["loss_noisy"]).mean()
        lossC = agg(lab, lambda r: r["curves"][-1]["loss_clean"]).mean()
        mN = agg(lab, lambda r: r["curves"][-1]["muhat_noisy"])
        mC = agg(lab, lambda r: r["curves"][-1]["muhat_clean"])
        gs = agg(lab, lambda r: r["curves"][-1]["gshare_noisy"]).mean()
        nN = agg(lab, lambda r: r["final"]["norm_noisy"])
        nC = agg(lab, lambda r: r["final"]["norm_clean"])
        print(f"{lab}: loss N/C {lossN:.4f}/{lossC:.4f} | "
              f"muhatN {mN.mean():.5f}±{mN.std():.5f} "
              f"muhatC {mC.mean():.5f}±{mC.std():.5f} | gshareN {gs:.2f} | "
              f"normErr N {nN.mean():.5f}±{nN.std():.5f} "
              f"C {nC.mean():.5f}±{nC.std():.5f}")


if __name__ == "__main__":
    main()
