"""WHO gets the top gradient on ToolHang: characterize top-1%/top-10% per-sample
encoder-gradient samples by task phase, demo identity, time-in-demo, kNN label
deviation, and action kinematics. Envs: CKPT, TAG, GLOSSES (default "L2 HT"), DSP."""
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

DSP = os.environ.get("DSP", "/home/jigu/.cache/huggingface/hub/datasets--ChaoyiPan--mip-dataset/blobs/c5cb01b119a109a3d4842ca515906e86f1f35a8ee1a333d22b3503c1a513a8f4")
with initialize_config_dir(version_base=None, config_dir=os.path.abspath("examples/configs")):
    cfg = compose(config_name="main", overrides=["task=tool_hang_ph_state_delta_legacy",
        "+task.dataset_path=" + DSP, "network=chiunet",
        "optimization.loss_type=regression", "optimization.auto_resume=false", "log.wandb_mode=disabled"])
OmegaConf.set_struct(cfg, False); cfg.task.obs_dim = 53; cfg.task.horizon = 16
ds = make_dataset(cfg.task)
ag = TrainingAgent(cfg)
dev = cfg.optimization.device

# ---- phase labels per demo from raw gripper cycles
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
    g = a[:, 6]; T = len(a)
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
    meta.append({"demo": d, "t": t, "tf": t / m["T"], "phase": phase(d, t),
                 "amag": float(np.linalg.norm(aa[t, :3])), "rmag": float(np.linalg.norm(aa[t, 3:6])), "jerk": jerk})

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
tops = {}
for loss_name in os.environ.get("GLOSSES", "L2 HT").split():
    gs = []
    B = 256
    for b0 in range(0, N, B):
        xb, yb = get_batch(idx[b0:b0 + B])
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
        elif loss_name == "HG":
            pred, s_raw = fm.net(torch.zeros_like(yb), t0, t0, emb_r)
            r = pred - yb
            sb = torch.nn.functional.softplus(s_raw).reshape(len(xb), -1).mean(dim=1) + 1e-3
            Dn = r[0].numel()
            li = 0.5 * r.reshape(len(xb), -1).pow(2).sum(dim=1) / (sb ** 2) + Dn * torch.log(sb)
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
    top1, top10 = set(order[:N // 100]), set(order[:N // 10])
    tops[loss_name] = (top1, top10)
    for nm, sel in [("top1%", order[:N // 100]), ("top10%", order[:N // 10]), ("ALL", np.arange(N))]:
        ph_counts = {p: 0 for p in PH}
        for j in sel:
            ph_counts[meta[j]["phase"]] += 1
        tot = len(sel)
        phs = " ".join(f"{p}:{100*ph_counts[p]/tot:.0f}%" for p in PH)
        demos = [meta[j]["demo"] for j in sel]
        uniq = len(set(demos))
        from collections import Counter
        top5share = sum(c for _, c in Counter(demos).most_common(5)) / tot
        med = lambda f: float(np.median([meta[j][f] for j in sel]))
        print(f"PAT {TAG} {loss_name} {nm:6s} | {phs} | demos: uniq={uniq} top5share={100*top5share:.0f}% "
              f"| tf={med('tf'):.2f} knn_dev={float(np.median(knn_dev[list(sel)])):.3f} amag={med('amag'):.3f} jerk={med('jerk'):.3f}", flush=True)
if "L2" in tops and "HT" in tops:
    o1 = len(tops["L2"][0] & tops["HT"][0]) / max(1, len(tops["L2"][0]))
    o10 = len(tops["L2"][1] & tops["HT"][1]) / max(1, len(tops["L2"][1]))
    print(f"PAT {TAG} OVERLAP L2-vs-HT: top1% {100*o1:.0f}% | top10% {100*o10:.0f}%", flush=True)
print("PATTERN-DONE")
