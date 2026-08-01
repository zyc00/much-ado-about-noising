"""Channel decomposition of the label-deviation tail: which action channels
carry the top-10% cross-demo disagreement, and are tail samples adjacent to
gripper flips / concentrated at the handover? Runs on any lowdim h5.
Envs: DSP, OBSKEYS, TAG."""
import os
import h5py
import numpy as np
import torch

DSP = os.environ["DSP"]; TAG = os.environ["TAG"]
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
if AD >= 14:
    GROUPS = [("a0_pos", slice(0, 3)), ("a0_rot", slice(3, 6)), ("a0_grip", slice(6, 7)),
              ("a1_pos", slice(7, 10)), ("a1_rot", slice(10, 13)), ("a1_grip", slice(13, 14))]
    GRIPS = [6, 13]
else:
    GROUPS = [("pos", slice(0, 3)), ("rot", slice(3, 6)), ("grip", slice(6, 7))]
    GRIPS = [6]
S = [(di, t) for di in OBS for t in range(2, len(ACT[di]) - 10)]
rng = np.random.RandomState(0)
S = [S[i] for i in rng.choice(len(S), min(6000, len(S)), replace=False)]
Dm = np.array([di for di, t in S])
X = np.stack([OBS[di][t] for di, t in S])
MODE = os.environ.get("MODE", "pose")
Xf = X / (X.std(0) + 1e-6)
if MODE == "vel":
    Xd = np.stack([OBS[di][t] - OBS[di][t - 2] for di, t in S])
    Xd = Xd / (Xd.std(0) + 1e-6)
    Xf = np.concatenate([Xf, Xd], 1)
Xs = torch.tensor(Xf)
D = torch.cdist(Xs, Xs)
D[torch.tensor(Dm[:, None] == Dm[None, :])] = 1e9
NN = D.topk(8, largest=False).indices.numpy()
A = np.stack([ACT[di][t] for di, t in S])
dev = A - A[NN].mean(1)
tot = (dev ** 2).sum(1)
tail = tot >= np.quantile(tot, 0.9)
print(f"TAILCHAN {TAG} mode={MODE} total-dev p50/p90={np.median(tot):.4f}/{np.quantile(tot,0.9):.4f}")
print(f"TAILCHAN {TAG} act_dim={AD} | tail kurt={float(((tot-tot.mean())**4).mean()/tot.var()**2):.1f}", flush=True)
te = (dev[tail] ** 2)
shares = " ".join(f"{nm}:{100*te[:, sl].sum()/te.sum():.0f}%" for nm, sl in GROUPS)
print(f"TAILCHAN {TAG} tail-energy by channel: {shares}", flush=True)
# flip adjacency: tail sample within +-3 steps of own-demo gripper sign flip
def flipadj(di, t, g, w=3):
    a = ACT[di][max(0, t - w):t + w + 1, g]
    return float((np.sign(a[:-1]) != np.sign(a[1:])).any())
fa = np.array([[flipadj(di, t, g) for g in GRIPS] for (di, t) in S])
base = fa.mean(0)
tfa = fa[tail].mean(0)
print(f"TAILCHAN {TAG} flip-adjacent frac: tail={np.round(tfa,2)} vs base={np.round(base,2)} "
      f"(enrichment {np.round(tfa/np.maximum(base,1e-6),1)})", flush=True)
tq = np.array([t / len(ACT[di]) for di, t in S])
hq = np.histogram(tq[tail], bins=4, range=(0, 1))[0]
print(f"TAILCHAN {TAG} tail episode-quartile counts: {hq.tolist()}", flush=True)
print("TAILCHAN-DONE")
