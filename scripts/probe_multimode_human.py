"""Level-attribution of human-data multimodality: at matched states (state-kNN
sets), decompose neighbor CHUNK labels into net-displacement SPEED and
DIRECTION; report per-H and per-phase the within-state speed CV and direction
dispersion (1 - ||mean unit vector||). Distinguishes single-action direction
bimodality (branch states) from chunk-level pacing spread (grows with H).
Env: DSP."""
import os

import h5py
import numpy as np
import torch

DSP = os.environ.get("DSP", "data/tool_hang_human_lowdim_up.hdf5")
OK = ["object", "robot0_eef_pos", "robot0_eef_quat", "robot0_gripper_qpos"]
PHN = ["reach1", "insert_frame", "reach2", "hang_tool"]
def sustained(sig, k=5):
    on = sig >= 0
    for t in range(len(on) - k):
        if on[t:t + k].all():
            return t
    return None

h = h5py.File(DSP, "r")
keys = sorted(h["data"].keys(), key=lambda k: int(k.split("_")[-1]))
S, PH = [], []
OBS, ACT = {}, {}
for di, k in enumerate(keys):
    a = np.clip(np.asarray(h[f"data/{k}/actions"]), -1, 1).astype(np.float32)
    o = h[f"data/{k}/obs"]
    ov_ = np.concatenate([np.asarray(o[q]) for q in OK], axis=1).astype(np.float32)
    g = a[:, -1]; T = len(a)
    c1 = sustained(g)
    if c1 is None: continue
    o1 = sustained(-g[c1:]); o1 = None if o1 is None else o1 + c1
    c2 = None if o1 is None else sustained(g[o1:]); c2 = None if c2 is None else c2 + o1
    o2 = None if c2 is None else sustained(-g[c2:]); o2 = None if o2 is None else o2 + c2
    OBS[di] = ov_; ACT[di] = a
    for t in range(1, T - 33):
        if t < c1: p = 0
        elif o1 is not None and t < o1: p = 1
        elif c2 is not None and t < c2: p = 2
        elif o2 is not None and t < o2: p = 3
        else: continue
        S.append((di, t)); PH.append(p)
    if di >= 199: break
h.close()
PH = np.array(PH); Dm = np.array([di for di, t in S])
rng = np.random.RandomState(0)
sel = rng.choice(len(S), min(6000, len(S)), replace=False)
S = [S[i] for i in sel]; PH = PH[sel]; Dm = Dm[sel]

X = np.stack([OBS[di][t] for di, t in S])
Xs = torch.tensor((X - X.mean(0)) / (X.std(0) + 1e-6))
DIST = torch.cdist(Xs, Xs)
DIST[torch.tensor(Dm[:, None] == Dm[None, :])] = 1e9
NN = DIST.topk(8, largest=False).indices.numpy()

print("H | phase | speed p50 | speed-CV p50 | dir-dispersion p50 | frac dispersion>0.5 (branchy)", flush=True)
for H in [1, 4, 8, 16, 32]:
    V = np.stack([ACT[di][t:t + H, :3].sum(0) for di, t in S])   # net displacement of the chunk
    for p in range(4):
        idx = np.where(PH == p)[0]
        cvs, disps, spds = [], [], []
        for i in idx:
            vs = V[NN[i]]
            s = np.linalg.norm(vs, axis=1)
            if s.mean() < 1e-6: continue
            u = vs / (s[:, None] + 1e-8)
            cvs.append(s.std() / (s.mean() + 1e-8))
            disps.append(1 - np.linalg.norm(u.mean(0)))
            spds.append(s.mean())
        cvs = np.array(cvs); disps = np.array(disps); spds = np.array(spds)
        print(f"H={H:2d} | {PHN[p]:12s} | {np.median(spds):.3f} | {np.median(cvs):.2f} | "
              f"{np.median(disps):.2f} | {np.mean(disps > 0.5):.2f}", flush=True)
print("MULTIMODE-DONE")
