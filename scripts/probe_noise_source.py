"""Where does the state-conditional noise (the loss pockets) come from —
partial observability (aleatoric in obs space) or sparse-coverage estimation
error (epistemic)?

Model-free measurement on the raw datasets. For state pairs (i, j) from
DIFFERENT trajectories with obs-window distance ||x_i - x_j|| <= eps, the
paired target dispersion D(eps) = E||a_i - a_j||^2 satisfies
  D(eps) -> 2 Var[a | x]  as eps -> 0   (aleatoric-in-obs intercept)
while its growth with eps is the state-resolvable (epistemic-at-finite-
coverage) part. Comparing MP-200 vs MP-2k at MATCHED eps separates a data
property from a coverage artifact; the typical nearest-neighbor radius each
dataset actually offers ("r_nn") shows what dispersion a learner at that
coverage cannot avoid seeing.

Strata: pocket deciles (3-4 mid-contact, 8-9 settle/release) vs control
(0-2 approach). Dims split: pose (0:6 of raw 7-dim action) vs gripper (6).
Prints NSRC lines. Env: NS_N (queries/stratum), NS_CHUNK (target steps).
"""
import os
import sys

sys.path.insert(0, "scripts")
sys.path.insert(0, ".")
import h5py
import numpy as np
from scipy.spatial import cKDTree

NQ = int(os.environ.get("NS_N", "3000"))
CH = int(os.environ.get("NS_CHUNK", "8"))
OK = ["object", "robot0_eef_pos", "robot0_eef_quat", "robot0_gripper_qpos"]
SETS = {"MP-200": "data/tool_hang_full2ins_mp_200.hdf5",
        "MP-2k": "data/tool_hang_full2ins_mp_2000.hdf5"}
if os.environ.get("NS_SETS"):
    SETS = dict(p.split("=") for p in os.environ["NS_SETS"].split(","))
STRATA = {"pocket34": (0.3, 0.5), "pocket89": (0.8, 1.0), "ctrl02": (0.0, 0.3)}
EPS = [0.05, 0.1, 0.2, 0.4, 0.8, 1.6]

# common z-scoring from MP-2k so distances are comparable across datasets
def load(path):
    h = h5py.File(path, "r")
    names = sorted(h["data"].keys(), key=lambda d: int(d.split("_")[1]))
    X, Y, P, TID = [], [], [], []
    for ti, dn in enumerate(names):
        o = h[f"data/{dn}/obs"]
        S = np.concatenate([np.asarray(o[k]) for k in OK], 1).astype(np.float32)
        A = np.asarray(h[f"data/{dn}/actions"], dtype=np.float32)
        L = len(S)
        if L < CH + 3:
            continue
        for i in range(1, L - CH, 2):
            X.append(np.stack([S[i - 1], S[i]]).reshape(-1))
            Y.append(A[i:i + CH])
            P.append(i / L)
            TID.append(ti)
    h.close()
    return (np.stack(X), np.stack(Y), np.asarray(P),
            np.asarray(TID, dtype=np.int64))

_refname = list(SETS)[-1]
X2k, Y2k, P2k, T2k = load(SETS[_refname])
mu, sd = X2k.mean(0), X2k.std(0) + 1e-6
amu = Y2k.reshape(-1, 7).mean(0)
asd = Y2k.reshape(-1, 7).std(0) + 1e-6
print(f"NSRC pool MP-2k {len(X2k)}", flush=True)

rng = np.random.RandomState(0)
for dname, path in SETS.items():
    if dname == _refname:
        X, Y, P, TID = X2k, Y2k, P2k, T2k
    else:
        X, Y, P, TID = load(path)
        print(f"NSRC pool {dname} {len(X)}", flush=True)
    Xn = (X - mu) / sd
    Yn = (Y - amu) / asd                       # (N, CH, 7)
    tree = cKDTree(Xn)
    for sname, (lo, hi) in STRATA.items():
        qidx = np.where((P >= lo) & (P < hi))[0]
        qidx = rng.choice(qidx, min(NQ, len(qidx)), replace=False)
        # nearest cross-trajectory neighbor radius the dataset offers
        rnn = []
        pairs = {e: [] for e in EPS}
        for q in qidx:
            d, nb = tree.query(Xn[q], k=48)
            cross = nb[(TID[nb] != TID[q])]
            dc = d[(TID[nb] != TID[q])]
            if len(cross) == 0:
                continue
            rnn.append(dc[0])
            dy2 = ((Yn[cross] - Yn[q]) ** 2)   # (k, CH, 7)
            pose = dy2[..., 0:6].mean(axis=(1, 2))
            grip = dy2[..., 6].mean(axis=1)
            for e in EPS:
                m = dc <= e
                if m.any():
                    pairs[e].append((pose[m].mean(), grip[m].mean(),
                                     m.sum()))
        rnn = np.asarray(rnn)
        line = " ".join(
            f"e{e}:" + (f"{np.mean([p[0] for p in pairs[e]]):.4f}/"
                        f"{np.mean([p[1] for p in pairs[e]]):.4f}/"
                        f"n{len(pairs[e])}"
                        if pairs[e] else "--")
            for e in EPS)
        print(f"NSRC {dname} {sname} rnn_p50 {np.median(rnn):.3f} "
              f"rnn_p90 {np.percentile(rnn, 90):.3f} | D(eps) pose/grip: "
              f"{line}", flush=True)
print("NSRC done", flush=True)
