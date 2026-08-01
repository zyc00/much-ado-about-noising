"""Direct sigma probe on human ToolHang: does the learned sigma(x) of hetero-t
(a) track label noise (kNN action deviation), (b) suppress the noisy tail's
gradient pressure, (c) amplify quiet learnable states? Envs: CKPT, TAG, DSP, NU.
Prints SIG lines: percentiles, per-phase sigma, Spearman correlations, and a
knn-noise-decile table of median sigma + mean-normalized gradient pressure
under L2 vs hetero-t."""
import os

os.environ["MUJOCO_GL"] = "egl"
import sys

import numpy as np
import torch

sys.path.insert(0, "scripts"); sys.path.insert(0, ".")
import h5py
from hydra import compose, initialize_config_dir
from omegaconf import OmegaConf

from mip.agent import TrainingAgent
from mip.datasets.robomimic_dataset import make_dataset

DSP = os.environ.get("DSP", "data/tool_hang_human_lowdim_up.hdf5")
NU = float(os.environ.get("NU", "2.0"))
with initialize_config_dir(version_base=None, config_dir=os.path.abspath("examples/configs")):
    cfg = compose(config_name="main", overrides=["task=tool_hang_ph_state_delta_legacy",
        "+task.dataset_path=" + DSP, "network=chiunet",
        "optimization.loss_type=regression_hetero_t", "optimization.auto_resume=false", "log.wandb_mode=disabled"])
OmegaConf.set_struct(cfg, False); cfg.task.obs_dim = 53; cfg.task.horizon = 16
ds = make_dataset(cfg.task)
ag = TrainingAgent(cfg)
dev = cfg.optimization.device

h = h5py.File(DSP, "r")
keys = sorted(h["data"].keys(), key=lambda k: int(k.split("_")[-1]))
def sustained(sig, k=5):
    on = sig >= 0
    for t in range(len(on) - k):
        if on[t:t + k].all():
            return t
    return None
demo_meta = []
for k in keys:
    a = np.asarray(h[f"data/{k}/actions"]).astype(np.float32)
    g = a[:, -1]; T = len(a)
    c1 = sustained(g)
    o1 = None if c1 is None else sustained(-g[c1:]) and sustained(-g[c1:]) + c1
    c2 = None if o1 is None else (sustained(g[o1:]) and sustained(g[o1:]) + o1)
    o2 = None if c2 is None else (sustained(-g[c2:]) and sustained(-g[c2:]) + c2)
    demo_meta.append({"T": T, "c1": c1, "o1": o1, "c2": c2, "o2": o2, "a": a})
h.close()
PH = ["reach1", "insert_frame", "reach2", "hang_tool", "end"]
def phase(d, t):
    m = demo_meta[d]
    if m["c1"] is not None and t < m["c1"]: return "reach1"
    if m["o1"] is not None and t < m["o1"]: return "insert_frame"
    if m["c2"] is not None and t < m["c2"]: return "reach2"
    if m["o2"] is not None and t < m["o2"]: return "hang_tool"
    return "end"

ee = ds.sampler.replay_buffer.episode_ends[:]
starts = np.concatenate([[0], ee[:-1]])
N = 4096
idx = np.random.RandomState(0).choice(len(ds), N, replace=False)
meta = []
for i in idx:
    b0, b1, s0, s1 = ds.sampler.indices[int(i)]
    d = int(np.searchsorted(ee, b0, side="right"))
    t = int(b0 - starts[d] + (1 - s0))
    m = demo_meta[d]
    t = min(max(t, 0), m["T"] - 1)
    aa = m["a"]
    jerk = float(np.linalg.norm(aa[t, :3] - aa[max(t - 1, 0), :3]))
    meta.append({"demo": d, "t": t, "phase": phase(d, t),
                 "amag": float(np.linalg.norm(aa[t, :3])), "jerk": jerk})

def get_batch(ids):
    xs, ys = [], []
    for i in ids:
        b = ds[int(i)]
        o = b["obs"]["state"] if isinstance(b["obs"], dict) else b["obs"]
        xs.append(o[:2] if o.shape[0] > 2 else o)
        ys.append(b["action"])
    return torch.stack(xs).to(dev), torch.stack(ys).to(dev)

Xs, Ys = get_batch(idx)
flatX = Xs.reshape(N, -1).cpu().numpy(); flatY = Ys.reshape(N, -1).cpu().numpy()
D = torch.cdist(torch.tensor(flatX), torch.tensor(flatX))
nbr = D.topk(9, largest=False).indices[:, 1:].numpy()
knn_dev = np.linalg.norm(flatY - flatY[nbr].mean(1), axis=1)

ag.load(os.environ["CKPT"], load_optimizer=False)
enc, fm = ag.encoder, ag.flow_map
enc.eval(); fm.eval()
TAG = os.environ.get("TAG", "ckpt")

sig, rnorm, wL2, wHT, wHG = [], [], [], [], []
B = 256
for b0 in range(0, N, B):
    xb, yb = get_batch(idx[b0:b0 + B])
    with torch.no_grad():
        emb = enc(xb, None)
        t0 = torch.zeros(len(xb), device=dev)
        pred, s_raw = fm.net(torch.zeros_like(yb), t0, t0, emb)
        r = (pred - yb).reshape(len(xb), -1)
        sb = torch.nn.functional.softplus(s_raw).reshape(len(xb), -1).mean(dim=1) + 1e-3
        Dn = r.shape[1]
        r2 = r.pow(2).sum(dim=1)
        z2 = r2 / (sb ** 2 * Dn)
        # |dL/dpred| per sample: L2 -> 2|r| ; hetero-t -> (nu+1)/(nu) * |r| / (sigma^2 (1+z2/nu))
        wL2.append((2 * r2.sqrt()).cpu().numpy())
        wHT.append((((NU + 1) / NU) * r2.sqrt() / (sb ** 2 * (1 + z2 / NU))).cpu().numpy())
        wHG.append((r2.sqrt() / sb ** 2).cpu().numpy())
        sig.append(sb.cpu().numpy()); rnorm.append(r2.sqrt().cpu().numpy())
sig = np.concatenate(sig); rnorm = np.concatenate(rnorm)
wL2 = np.concatenate(wL2); wHT = np.concatenate(wHT); wHG = np.concatenate(wHG)
wL2 /= wL2.mean(); wHT /= wHT.mean(); wHG /= wHG.mean()

def spear(a, b):
    ra = np.argsort(np.argsort(a)).astype(np.float64)
    rb = np.argsort(np.argsort(b)).astype(np.float64)
    return float(np.corrcoef(ra, rb)[0, 1])

pct = lambda a: "/".join(f"{np.percentile(a, q):.4f}" for q in (10, 50, 90, 99))
print(f"SIG {TAG} sigma p10/50/90/99 = {pct(sig)} | resid p10/50/90/99 = {pct(rnorm)}", flush=True)
print(f"SIG {TAG} spearman: rho(sigma,knn_dev)={spear(sig, knn_dev):.3f} rho(sigma,|r|)={spear(sig, rnorm):.3f} "
      f"rho(sigma,jerk)={spear(sig, np.array([m['jerk'] for m in meta])):.3f}", flush=True)
for p in PH:
    m = np.array([mm["phase"] == p for mm in meta])
    if m.sum() > 10:
        print(f"SIG {TAG} phase {p:12s} n={int(m.sum()):4d} | sigma p50={np.median(sig[m]):.4f} "
              f"knn_dev p50={np.median(knn_dev[m]):.3f} | w_L2={wL2[m].mean():.2f} w_HT={wHT[m].mean():.2f} w_HG={wHG[m].mean():.2f}", flush=True)
qs = np.quantile(knn_dev, np.linspace(0, 1, 11))
print(f"SIG {TAG} knn-noise-decile table: decile | sigma_p50 | w_L2 | w_HT | w_HG (mean-normalized pressure)", flush=True)
for di in range(10):
    m = (knn_dev >= qs[di]) & (knn_dev <= qs[di + 1])
    print(f"SIG {TAG} D{di} knn[{qs[di]:.3f},{qs[di+1]:.3f}] | {np.median(sig[m]):.4f} | {wL2[m].mean():.2f} | {wHT[m].mean():.2f} | {wHG[m].mean():.2f}", flush=True)
# action-magnitude deciles: the scripted starvation axis (micro vs macro labels)
amag_chunk = np.linalg.norm(flatY, axis=1)
qa = np.quantile(amag_chunk, np.linspace(0, 1, 11))
print(f"SIG {TAG} action-magnitude-decile table: decile | sigma_p50 | w_L2 | w_HT | w_HG", flush=True)
for di in range(10):
    m = (amag_chunk >= qa[di]) & (amag_chunk <= qa[di + 1])
    print(f"SIG {TAG} A{di} amag[{qa[di]:.3f},{qa[di+1]:.3f}] | {np.median(sig[m]):.4f} | {wL2[m].mean():.2f} | {wHT[m].mean():.2f} | {wHG[m].mean():.2f}", flush=True)
top = np.argsort(knn_dev)[-N // 10:]
print(f"SIG {TAG} top-10% noisiest states: share of total pressure L2={wL2[top].sum()/wL2.sum()*100:.0f}% "
      f"HT={wHT[top].sum()/wHT.sum()*100:.0f}% HG={wHG[top].sum()/wHG.sum()*100:.0f}% | sigma_p50={np.median(sig[top]):.4f} vs all {np.median(sig):.4f}", flush=True)
print("SIGMA-DONE")
