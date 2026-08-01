"""Chunking-vs-ambiguity on HUMAN data: tremor separates RAW labels (noise
de-collides coincidences) but a regression policy fits CONDITIONAL MEANS —
so measure both: (a) raw chunk collision P(d<eps | different phase), (b)
collision of state-kNN-SMOOTHED chunk labels (local estimate of E[chunk|s]).
Registered prediction: raw ~0 at every H (noise-separated); smoothed high at
H=1 on quiet cross-phase pairs, falling with H — the mean-level form in which
the ambiguity (and its chunking cure) exists on human data.
Phases via gripper cycles (reach1/insert_frame/reach2/hang_tool). Env: DSP."""
import os

import h5py
import numpy as np
import torch

DSP = os.environ.get("DSP", "data/tool_hang_human_lowdim_up.hdf5")
OK = ["object", "robot0_eef_pos", "robot0_eef_quat", "robot0_gripper_qpos"]
def sustained(sig, k=5):
    on = sig >= 0
    for t in range(len(on) - k):
        if on[t:t + k].all():
            return t
    return None

h = h5py.File(DSP, "r")
keys = sorted(h["data"].keys(), key=lambda k: int(k.split("_")[-1]))
A, PH, D, S = [], [], [], []
PHN = ["reach1", "insert_frame", "reach2", "hang_tool"]
for di, k in enumerate(keys):
    a = np.clip(np.asarray(h[f"data/{k}/actions"]), -1, 1).astype(np.float32)
    o = h[f"data/{k}/obs"]
    ov_ = np.concatenate([np.asarray(o[q]) for q in OK], axis=1).astype(np.float32)
    g = a[:, -1]; T = len(a)
    c1 = sustained(g)
    if c1 is None: continue
    o1 = sustained(-g[c1:]);  o1 = None if o1 is None else o1 + c1
    c2 = None if o1 is None else sustained(g[o1:]); c2 = None if c2 is None else c2 + o1
    o2 = None if c2 is None else sustained(-g[c2:]); o2 = None if o2 is None else o2 + c2
    for t in range(1, T - 33):
        if t < c1: p = 0
        elif o1 is not None and t < o1: p = 1
        elif c2 is not None and t < c2: p = 2
        elif o2 is not None and t < o2: p = 3
        else: continue
        A.append(a); PH.append(p); D.append(di); S.append((di, t))
    if di >= 199: break
h.close()
PH = np.array(PH); Dm_ = np.array(D)
demos = {di: a for (di, t), a in zip(S, [A[i] for i in range(len(A))])}
rng = np.random.RandomState(0)
sel = rng.choice(len(S), min(6000, len(S)), replace=False)
S = [S[i] for i in sel]; PH = PH[sel]; Dm_ = Dm_[sel]

# state vectors for kNN smoothing (standardized raw obs at t)
h = h5py.File(DSP, "r")
OBS, ACT = {}, {}
for di in sorted(set(Dm_)):
    k = keys[di]
    o = h[f"data/{k}/obs"]
    OBS[di] = np.concatenate([np.asarray(o[q]) for q in OK], axis=1).astype(np.float32)
    ACT[di] = np.clip(np.asarray(h[f"data/{k}/actions"]), -1, 1).astype(np.float32)
h.close()
X = np.stack([OBS[di][t] for di, t in S])
mu, sd = X.mean(0), X.std(0) + 1e-6
Xs = torch.tensor((X - mu) / sd)
DIST = torch.cdist(Xs, Xs)
DIST[torch.tensor(Dm_[:, None] == Dm_[None, :])] = 1e9
NN = DIST.topk(8, largest=False).indices.numpy()

apos = np.array([np.linalg.norm(ACT[di][t, :3]) for di, t in S])
qthr = np.quantile(apos, 0.4)
quiet = apos < qthr
dev = "cuda" if torch.cuda.is_available() else "cpu"
print(f"n={len(S)} demos={len(set(Dm_))} quiet-thr(|a_pos| 40pct)={qthr:.4f} | phase counts: " +
      " ".join(f"{PHN[p]}:{int((PH == p).sum())}" for p in range(4)), flush=True)
print("H | RAW cross-phase P(d<.1)/P(<.3) | SMOOTH cross P(d<.1)/P(<.3) | SMOOTH quiet-cross P(<.1)/P(<.3) | SMOOTH same-phase P(<.3) | amb(.3)", flush=True)
for H in [1, 2, 4, 8, 16, 32]:
    def chunk(di, t):
        c = ACT[di][t:t + H]
        if len(c) < H:
            c = np.pad(c, ((0, H - len(c)), (0, 0)), "edge")
        return c.reshape(-1)
    C = np.stack([chunk(di, t) for di, t in S]) / np.sqrt(H)
    CS = np.stack([C[NN[i]].mean(0) for i in range(len(S))])   # kNN-smoothed = local conditional mean
    res = {}
    for nm, V in [("raw", C), ("sm", CS)]:
        Vt = torch.tensor(V, device=dev)
        tot_c = tot_s = 0; hit = {(e, "c"): 0 for e in (0.1, 0.3)}; hit.update({(e, "s"): 0 for e in (0.1, 0.3)})
        qtot = 0; qhit = {0.1: 0, 0.3: 0}
        B = 1024
        pt = torch.tensor(PH, device=dev); dt = torch.tensor(Dm_, device=dev); qt = torch.tensor(quiet, device=dev)
        for i0 in range(0, len(S), B):
            Dd = torch.cdist(Vt[i0:i0 + B], Vt)
            sd_ = dt[i0:i0 + B, None] == dt[None, :]
            cross = (pt[i0:i0 + B, None] != pt[None, :]) & ~sd_
            same = (pt[i0:i0 + B, None] == pt[None, :]) & ~sd_
            tot_c += int(cross.sum()); tot_s += int(same.sum())
            for e in (0.1, 0.3):
                hit[(e, "c")] += int((cross & (Dd < e)).sum()); hit[(e, "s")] += int((same & (Dd < e)).sum())
            qq = qt[i0:i0 + B, None] & qt[None, :] & cross
            qtot += int(qq.sum())
            for e in (0.1, 0.3):
                qhit[e] += int((qq & (Dd < e)).sum())
        res[nm] = (hit[(0.1, "c")] / max(1, tot_c), hit[(0.3, "c")] / max(1, tot_c),
                   qhit[0.1] / max(1, qtot), qhit[0.3] / max(1, qtot),
                   hit[(0.3, "s")] / max(1, tot_s),
                   hit[(0.3, "c")] / max(1, hit[(0.3, "c")] + hit[(0.3, "s")]))
    r, s = res["raw"], res["sm"]
    print(f"H={H:2d} | {r[0]:.4f}/{r[1]:.4f} | {s[0]:.4f}/{s[1]:.4f} | {s[2]:.4f}/{s[3]:.4f} | {s[4]:.4f} | {s[5]:.2f}", flush=True)
print("CHUNKHUMAN-DONE")
