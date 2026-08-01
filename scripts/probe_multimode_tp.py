"""Transport (two-arm) chunk-level multimodality with HISTORY-CONDITIONED
matching: is the direction dispersion at matched instantaneous states
RESOLVED when the kNN matching includes obs history? (User hypothesis: the
transport branch variable is temporal context -> obs history eliminates it,
explaining history's +25 on transport vs -15 on TH.)
For hist in {1, 2, 8} (frames concatenated for the kNN metric) x chunk H in
{1, 8, 16}: per episode-time-quartile and quiet subset, report direction
dispersion (1-||mean u||, concatenated 6-dim r0+r1 net displacement) and
speed CV over 8 state-NNs. Envs: DSP (hdf5 path), MAXD (demos cap)."""
import os

import h5py
import numpy as np
import torch

DSP = os.environ["DSP"]
MAXD = int(os.environ.get("MAXD", "150"))
h = h5py.File(DSP, "r")
keys = sorted(h["data"].keys(), key=lambda k: int(k.split("_")[-1]))[:MAXD]
k0 = keys[0]
if os.environ.get("OBSKEYS"):
    obs_keys = os.environ["OBSKEYS"].split(",")
else:
    obs_keys = sorted([k for k in h[f"data/{k0}/obs"].keys() if not k.endswith("image")])
OBS, ACT, TLEN = {}, {}, {}
for di, k in enumerate(keys):
    o = h[f"data/{k}/obs"]
    OBS[di] = np.concatenate([np.asarray(o[q]).reshape(len(h[f"data/{k}/actions"]), -1) for q in obs_keys], axis=1).astype(np.float32)
    ACT[di] = np.clip(np.asarray(h[f"data/{k}/actions"]), -1, 1).astype(np.float32)
    TLEN[di] = len(ACT[di])
h.close()
AD = ACT[0].shape[1]
ARM2 = AD >= 14
print(f"dataset={os.path.basename(DSP)} demos={len(OBS)} obs_dim={OBS[0].shape[1]} act_dim={AD} two-arm={ARM2}", flush=True)

S = []
for di in OBS:
    for t in range(8, TLEN[di] - 34):
        S.append((di, t))
rng = np.random.RandomState(0)
sel = rng.choice(len(S), min(6000, len(S)), replace=False)
S = [S[i] for i in sel]
Dm = np.array([di for di, t in S])
TQ = np.array([min(3, int(4 * t / TLEN[di])) for di, t in S])
def netdisp(di, t, H):
    c = ACT[di][t:t + H]
    if ARM2:
        return np.concatenate([c[:, :3].sum(0), c[:, 7:10].sum(0)])
    return c[:, :3].sum(0)
apos1 = np.array([np.linalg.norm(netdisp(di, t, 1)) for di, t in S])
quiet = apos1 < np.quantile(apos1, 0.4)

print("hist | H | quartile | n | speed-CV p50 | dir-disp p50 | frac>0.5 | quiet-subset disp p50", flush=True)
HISTMODES = [("h1", 1, False), ("h2cat", 2, False), ("h2vel", 2, True), ("h8vel", 8, True)]
for hname, hist, veldiff in HISTMODES:
    if veldiff:
        Xf = np.stack([OBS[di][t] for di, t in S])
        Xd = np.stack([(OBS[di][t] - OBS[di][t - hist + 1]) for di, t in S])
        Xd = Xd / (Xd.std(0) + 1e-6)
        X = np.concatenate([Xf / (Xf.std(0) + 1e-6), Xd], axis=1)
    else:
        X = np.stack([OBS[di][t - hist + 1:t + 1].reshape(-1) for di, t in S])
    Xs = torch.tensor((X - X.mean(0)) / (X.std(0) + 1e-6))
    DIST = torch.cdist(Xs, Xs)
    DIST[torch.tensor(Dm[:, None] == Dm[None, :])] = 1e9
    NN = DIST.topk(8, largest=False).indices.numpy()
    for H in (1, 8, 16):
        V = np.stack([netdisp(di, t, H) for di, t in S])
        for q in range(4):
            idx = np.where(TQ == q)[0]
            cvs, dsp = [], []
            for i in idx:
                vs = V[NN[i]]
                s = np.linalg.norm(vs, axis=1)
                if s.mean() < 1e-6: continue
                u = vs / (s[:, None] + 1e-8)
                cvs.append(s.std() / (s.mean() + 1e-8))
                dsp.append(1 - np.linalg.norm(u.mean(0)))
            dsp = np.array(dsp); cvs = np.array(cvs)
            qidx = np.array([i for i in idx if quiet[i]])
            qd = []
            for i in qidx:
                vs = V[NN[i]]
                s = np.linalg.norm(vs, axis=1)
                if s.mean() < 1e-6: continue
                u = vs / (s[:, None] + 1e-8)
                qd.append(1 - np.linalg.norm(u.mean(0)))
            qd = np.array(qd) if qd else np.array([np.nan])
            print(f"{hname} | H={H:2d} | Q{q+1} | {len(dsp):4d} | {np.median(cvs):.2f} | "
                  f"{np.median(dsp):.2f} | {np.mean(dsp > 0.5):.2f} | {np.nanmedian(qd):.2f}", flush=True)
print("MMTP-DONE")
