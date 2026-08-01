"""Does motion-conditioning fix the SPEED BIAS of the conditional mean on
transport? For MOVING query states (current |a| above median), compare the
kNN-mean chunk speed (the regression fit's local estimate) to the query's own
future speed, under pose-only (h1) vs pose+velocity (h2vel) matching.
Reports mean ratio fitted/true and the fraction of neighbors that are pauses.
Env: DSP, OBSKEYS."""
import os
import h5py
import numpy as np
import torch

DSP = os.environ["DSP"]
obs_keys = os.environ["OBSKEYS"].split(",")
h = h5py.File(DSP, "r")
keys = sorted(h["data"].keys(), key=lambda k: int(k.split("_")[-1]))[:150]
OBS, ACT = {}, {}
for di, k in enumerate(keys):
    o = h[f"data/{k}/obs"]
    n = len(h[f"data/{k}/actions"])
    OBS[di] = np.concatenate([np.asarray(o[q]).reshape(n, -1) for q in obs_keys], axis=1).astype(np.float32)
    ACT[di] = np.clip(np.asarray(h[f"data/{k}/actions"]), -1, 1).astype(np.float32)
h.close()
AD = ACT[0].shape[1]
def spd(di, t, H=8):
    c = ACT[di][t:t + H]
    if AD >= 14:
        return np.linalg.norm(np.concatenate([c[:, :3], c[:, 7:10]], 1), axis=1).mean()
    return np.linalg.norm(c[:, :3], axis=1).mean()
S = [(di, t) for di in OBS for t in range(8, len(ACT[di]) - 10)]
rng = np.random.RandomState(0)
S = [S[i] for i in rng.choice(len(S), min(6000, len(S)), replace=False)]
Dm = np.array([di for di, t in S])
cur = np.array([spd(di, t, 1) for di, t in S])
fut = np.array([spd(di, t, 8) for di, t in S])
moving = cur > np.median(cur)
paused = cur < np.quantile(cur, 0.25)
X = np.stack([OBS[di][t] for di, t in S])
r_pers = np.corrcoef(cur[moving], fut[moving])[0, 1]
print(f"SPEEDBIAS pacing persistence: corr(current speed, own 8-step future) at moving states = {r_pers:.2f}", flush=True)
# locate eef_pos dims inside the concatenated obs
eef_slices = []
off = 0
for q in obs_keys:
    w = OBS[0].shape[1]  # placeholder
for q, dim in [(q, None) for q in obs_keys]:
    pass
offs = {}
off = 0
h2 = h5py.File(DSP, "r")
o0 = h2[f"data/{keys[0]}/obs"]
for q in obs_keys:
    d = np.asarray(o0[q]).reshape(len(ACT[0]), -1).shape[1]
    offs[q] = (off, off + d); off += d
h2.close()
EEF = np.concatenate([np.arange(*offs[q]) for q in obs_keys if q.endswith("eef_pos")])
for name, vk, eefonly in (("pose-only", 0, False), ("pose+vel8-all", 8, False), ("pose+vel8-EEF", 8, True)):
    Xf = X / (X.std(0) + 1e-6)
    if vk:
        Xd = np.stack([(OBS[di][t] - OBS[di][t - vk]) / vk for di, t in S])
        if eefonly:
            Xd = Xd[:, EEF]
            Xd = Xd / (Xd.std() + 1e-8) * 3.0   # joint scaling preserves magnitude; upweight
        else:
            Xd = Xd / (Xd.std(0) + 1e-6)
        Xm = np.concatenate([Xf, Xd], 1)
    else:
        Xm = Xf
    D = torch.cdist(torch.tensor(Xm), torch.tensor(Xm))
    D[torch.tensor(Dm[:, None] == Dm[None, :])] = 1e9
    NN = D.topk(8, largest=False).indices.numpy()
    fit = np.array([fut[NN[i]].mean() for i in range(len(S))])
    pausefrac = np.array([paused[NN[i]].mean() for i in range(len(S))])
    m = moving
    print(f"SPEEDBIAS {name}: fitted/true speed at MOVING states p50={np.median(fit[m]/ (fut[m]+1e-8)):.2f} "
          f"mean={np.mean(fit[m]/(fut[m]+1e-8)):.2f} | pause-neighbor frac={pausefrac[m].mean():.2f}", flush=True)
print("SPEEDBIAS-DONE")
