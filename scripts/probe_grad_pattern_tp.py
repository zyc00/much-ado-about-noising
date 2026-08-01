"""Transport version of the gradient-pattern probe: WHO gets the top per-sample
encoder gradient. Stages from dual-arm gripper events (raw 14-dim actions: g0=6,
g1=13). Envs: TASK (hydra name), RAWFILE (HF filename for raw h5), CKPT, TAG,
GLOSSES ("L2 HT"), OBS_STEPS (match ckpt)."""
import os

os.environ["MUJOCO_GL"] = "egl"
import sys

import numpy as np
import torch

sys.path.insert(0, "scripts"); sys.path.insert(0, ".")
import h5py
from huggingface_hub import hf_hub_download
from hydra import compose, initialize_config_dir
from omegaconf import OmegaConf

from mip.agent import TrainingAgent
from mip.datasets.robomimic_dataset import make_dataset

TASK = os.environ.get("TASK", "transport_mh_state_abs")
with initialize_config_dir(version_base=None, config_dir=os.path.abspath("examples/configs")):
    cfg = compose(config_name="main", overrides=[f"task={TASK}", "network=chiunet",
        "optimization.loss_type=regression", "optimization.auto_resume=false", "log.wandb_mode=disabled"])
OmegaConf.set_struct(cfg, False)
if os.environ.get("OBS_STEPS"):
    cfg.task.obs_steps = int(os.environ["OBS_STEPS"])
OS = int(cfg.task.obs_steps)
cfg.task.horizon = 16
ds = make_dataset(cfg.task)
_s0 = ds[0]["obs"]["state"] if isinstance(ds[0]["obs"], dict) else ds[0]["obs"]
cfg.task.obs_dim = int(_s0.shape[-1])
ag = TrainingAgent(cfg)
dev = cfg.optimization.device

raw = hf_hub_download(repo_id=cfg.task.dataset_repo, filename=cfg.task.dataset_filename, repo_type="dataset")
h = h5py.File(raw, "r")
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
    g0, g1 = a[:, 6], a[:, 13]
    c0 = sustained(g0)
    c1 = None if c0 is None else (lambda x: x + c0 if x is not None else None)(sustained(g1[c0:]))
    o0 = None if c1 is None else (lambda x: x + c1 if x is not None else None)(sustained(-g0[c1:]))
    demo_meta.append({"T": len(a), "c0": c0, "c1": c1, "o0": o0, "a": a})
h.close()
PH = ["reach0", "carry0", "handover", "place", "end"]
def phase(d, t):
    m = demo_meta[d]
    if m["c0"] is not None and t < m["c0"]: return "reach0"
    if m["c1"] is not None and t < m["c1"]: return "carry0"
    if m["o0"] is not None and t < m["o0"]: return "handover"
    if m["o0"] is not None and t < m["T"] - 30: return "place"
    return "end"

ee = ds.sampler.replay_buffer.episode_ends[:]
starts = np.concatenate([[0], ee[:-1]])
N = 4096
idx = np.random.RandomState(0).choice(len(ds), N, replace=False)
meta = []
for i in idx:
    b0, b1, s0, s1 = ds.sampler.indices[int(i)]
    d = int(np.searchsorted(ee, b0, side="right"))
    t = int(b0 - starts[d] + (OS - 1 - s0))
    m = demo_meta[d]
    t = min(max(t, 0), m["T"] - 1)
    aa = m["a"]
    amag = float(max(np.linalg.norm(aa[t, 0:3]), np.linalg.norm(aa[t, 7:10])))
    meta.append({"demo": d, "t": t, "tf": t / m["T"], "phase": phase(d, t), "amag": amag})

def get_batch(ids):
    xs, ys = [], []
    for i in ids:
        b = ds[int(i)]
        o = b["obs"]["state"] if isinstance(b["obs"], dict) else b["obs"]
        xs.append(o[:OS])
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
from collections import Counter
for loss_name in os.environ.get("GLOSSES", "L2 HT").split():
    gs = []
    for b0 in range(0, N, 256):
        xb, yb = get_batch(idx[b0:b0 + 256])
        emb = enc(xb, None)
        emb_r = emb.detach().requires_grad_(True)
        t0 = torch.zeros(len(xb), device=dev)
        if loss_name == "HT":
            pred, s_raw = fm.net(torch.zeros_like(yb), t0, t0, emb_r)
            r = pred - yb
            sb = torch.nn.functional.softplus(s_raw).reshape(len(xb), -1).mean(dim=1) + 1e-3
            Dn = r[0].numel()
            z2 = r.reshape(len(xb), -1).pow(2).sum(dim=1) / (sb ** 2 * Dn)
            li = 0.5 * 3.0 * torch.log1p(z2 / 2.0) * Dn + Dn * torch.log(sb)
        elif loss_name == "MIPv2":
            a0 = yb + 0.1 * torch.randn_like(yb)
            tt = torch.full((len(xb),), 0.9, device=dev)
            pred = fm.get_velocity(tt, a0, emb_r)
            li = ((pred - yb) ** 2).sum(dim=(1, 2))
        else:
            pred = fm.get_velocity(t0, torch.zeros_like(yb), emb_r)
            li = ((pred - yb) ** 2).sum(dim=(1, 2))
        g = torch.autograd.grad(li.sum(), emb_r)[0]
        gs.append(g.reshape(len(xb), -1).norm(dim=1).detach().cpu().numpy())
    g = np.concatenate(gs)
    order = np.argsort(g)[::-1]
    srt = np.sort(g)[::-1]
    kurt = float(((g - g.mean()) ** 4).mean() / (g.var() ** 2))
    hi = knn_dev >= np.quantile(knn_dev, 0.9)
    lo = knn_dev <= np.quantile(knn_dev, 0.5)
    print(f"PAT {TAG} {loss_name} SHARES | top1%={100*srt[:N//100].sum()/g.sum():.1f}% "
          f"top10%={100*srt[:N//10].sum()/g.sum():.1f}% kurt={kurt:.1f} "
          f"g(tail)/g(typ)={np.median(g[hi])/np.median(g[lo]):.2f}", flush=True)
    for nm, sel in [("top1%", order[:N // 100]), ("top10%", order[:N // 10]), ("ALL", np.arange(N))]:
        ph_counts = Counter(meta[j]["phase"] for j in sel)
        tot = len(sel)
        phs = " ".join(f"{p}:{100*ph_counts.get(p,0)/tot:.0f}%" for p in PH)
        demos = [meta[j]["demo"] for j in sel]
        top5share = sum(c for _, c in Counter(demos).most_common(5)) / tot
        med = lambda f: float(np.median([meta[j][f] for j in sel]))
        print(f"PAT {TAG} {loss_name} {nm:6s} | {phs} | demos: uniq={len(set(demos))} top5={100*top5share:.0f}% "
              f"| tf={med('tf'):.2f} knn_dev={float(np.median(knn_dev[list(sel)])):.3f} amag={med('amag'):.3f}", flush=True)
print("PATTERN-DONE")
