"""Reconcile 'TH is speed-multimodal' vs 'pockets are direction-dispersed':
pocket (insert/hang quiet) chunk direction dispersion under (a) unit-normalized
(as before), (b) magnitude-weighted 1-||mean v||/mean||v|| (tremor-robust),
(c) orientation-conditioned matching (kNN metric upweights eef_quat + grip x5;
if the branch is orientation-keyed, dispersion collapses), and (d) a tremor
baseline: synthetic isotropic noise chunks at the pocket speed/SNR. Env: DSP."""
import os
import h5py
import numpy as np
import torch

DSP = os.environ["DSP"]
OK = ["object", "robot0_eef_pos", "robot0_eef_quat", "robot0_gripper_qpos"]
def sustained(sig, k=5):
    on = sig >= 0
    for t in range(len(on) - k):
        if on[t:t + k].all():
            return t
    return None
h = h5py.File(DSP, "r")
keys = sorted(h["data"].keys(), key=lambda k: int(k.split("_")[-1]))
S, OBS, ACT, QIDX = [], {}, {}, {}
for di, k in enumerate(keys):
    a = np.clip(np.asarray(h[f"data/{k}/actions"]), -1, 1).astype(np.float32)
    o = h[f"data/{k}/obs"]
    ov_ = np.concatenate([np.asarray(o[q]) for q in OK], axis=1).astype(np.float32)
    nq = 0
    for q in OK[:-2]:
        nq += np.asarray(o[q]).shape[1]
    QIDX[di] = (nq - 4, nq + 2)   # eef_quat(4) start, through gripper(2)
    g = a[:, -1]; T = len(a)
    c1 = sustained(g)
    if c1 is None: continue
    o1 = sustained(-g[c1:]); o1 = None if o1 is None else o1 + c1
    c2 = None if o1 is None else sustained(g[o1:]); c2 = None if c2 is None else c2 + o1
    OBS[di] = ov_; ACT[di] = a
    for t in range(1, T - 34):
        inpocket = (o1 is not None and c1 <= t < o1) or (c2 is not None and o1 is not None and t >= c2)
        if inpocket:
            S.append((di, t))
    if di >= 199: break
h.close()
rng = np.random.RandomState(0)
sel = rng.choice(len(S), min(4000, len(S)), replace=False)
S = [S[i] for i in sel]
Dm = np.array([di for di, t in S])
apos = np.array([np.linalg.norm(ACT[di][t, :3]) for di, t in S])
quiet = apos < np.quantile(apos, 0.5)
X = np.stack([OBS[di][t] for di, t in S])
mu, sd = X.mean(0), X.std(0) + 1e-6
for variant in ("plain", "orient5x"):
    W = np.ones(X.shape[1], dtype=np.float32)
    if variant == "orient5x":
        q0, q1 = QIDX[S[0][0]]
        W[q0:q1] = 5.0
    Xs = torch.tensor(((X - mu) / sd) * W)
    DIST = torch.cdist(Xs, Xs)
    DIST[torch.tensor(Dm[:, None] == Dm[None, :])] = 1e9
    NN = DIST.topk(8, largest=False).indices.numpy()
    for H in (1, 8):
        V = np.stack([ACT[di][t:t + H, :3].sum(0) for di, t in S])
        d_unit, d_wt = [], []
        for i in np.where(quiet)[0]:
            vs = V[NN[i]]
            s = np.linalg.norm(vs, axis=1)
            if s.mean() < 1e-6: continue
            u = vs / (s[:, None] + 1e-8)
            d_unit.append(1 - np.linalg.norm(u.mean(0)))
            d_wt.append(1 - np.linalg.norm(vs.mean(0)) / (s.mean() + 1e-8))
        print(f"POCKET {variant} H={H}: unit-disp p50={np.median(d_unit):.2f} "
              f"mag-weighted p50={np.median(d_wt):.2f} n={len(d_unit)}", flush=True)
# tremor baseline: mean drift vector at pocket SNR
for H in (1, 8):
    drift = np.array([0.1, 0.0, -0.2]) * H   # arbitrary coherent direction, |v|~pocket scale
    sims = []
    for _ in range(2000):
        vs = drift + 0.2 * np.sqrt(H) * rng.randn(8, 3)
        s = np.linalg.norm(vs, axis=1); u = vs / s[:, None]
        sims.append(1 - np.linalg.norm(u.mean(0)))
    print(f"POCKET tremor-baseline H={H} (coherent dir + iso noise, SNR~pocket): unit-disp p50={np.median(sims):.2f}", flush=True)
print("POCKET-DONE")
