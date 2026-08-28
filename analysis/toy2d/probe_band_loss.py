"""Per-band converged loss table (canonical env): for each arm, mean |err|
on the NOISE band and the PRECISION (insert) band, measured against both the
noisy training labels Y and the clean law Yc. Env: SESSIONS, BL_ARMS, SEEDS.
Prints: BANDLOSS <session> <arm> <seed> noisy[noise_band precision_band]
        clean[noise_band precision_band]
"""
import os

import numpy as np
import torch

import toytube as T


def run(session, arm, seed):
    # re-init module globals for the session
    os.environ["NOISE_TYPE"] = session
    import importlib
    importlib.reload(T)
    drng = np.random.RandomState(1000 + seed)
    X, Y, Yc, AL = T.build_dataset(drng)
    norm = T.Norm(X, Y)
    Xn, Yn, Ycn = norm.nx(X), norm.ny(Y), norm.ny(Yc)
    ins = (X[:, 1] > T.Y_DOCK) & (X[:, 1] <= T.Y_FUN_LO)
    net = T.train(arm, seed, Xn, Yn)
    net.eval()
    with torch.no_grad():
        pred = net(torch.tensor(Xn, device=T.DEV)).cpu().numpy()
    r = {}
    for bname, mask in (("noise", AL), ("prec", ins)):
        r[f"noisy_{bname}"] = float(np.abs(pred[mask] - Yn[mask]).mean())
        r[f"clean_{bname}"] = float(np.abs(pred[mask] - Ycn[mask]).mean())
    ev = T.evaluate(net, norm, n=50, seed0=seed)
    print(f"BANDLOSS {session} {arm} s{seed} "
          f"noisy[{r['noisy_noise']:.4f} {r['noisy_prec']:.4f}] "
          f"clean[{r['clean_noise']:.4f} {r['clean_prec']:.4f}] "
          f"SR {ev['success']:.2f}", flush=True)


SESSIONS = os.environ.get("SESSIONS", "dir,zturn,speed").split(",")
ARMS_ = os.environ.get("BL_ARMS", "l2,ht,hg").split(",")
SEEDS_ = [int(x) for x in os.environ.get("BL_SEEDS", "0,1").split(",")]
for ss in SESSIONS:
    for arm in ARMS_:
        for sd in SEEDS_:
            run(ss, arm, sd)
print(f"BANDLOSS done DOSE dir:S_HI={os.environ.get('S_HI','0.07')} "
      f"speed:T1S3={os.environ.get('T1S3','1.2')} zturn:X_OFF={os.environ.get('X_OFF','0.06')}",
      flush=True)
