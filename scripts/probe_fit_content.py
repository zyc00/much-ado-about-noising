"""Does Student-t actually FIT the important content better (in plain MSE terms)?
For a ladder of checkpoints, compute per-sample prediction MSE (deterministic
readout f(s,0,0)) on 4096 fixed samples, grouped by:
  typ    = kNN-label-dev bottom half (learnable)
  tail   = kNN-label-dev top decile (irreducible disagreement)
  servo  = insert_frame phase AND dev <= median  (the SR-critical content)
  hang   = hang_tool phase AND dev <= median
PRE-REGISTERED: HT-trained servo-MSE < L2-trained servo-MSE late; L2 servo-MSE
non-monotone (acquire-then-unlearn); tail-MSE similar or HT >= L2.
Envs: RUN (log dir tag), STEPS (space-sep), TAG, DSP."""
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

DSP = os.environ.get("DSP")
with initialize_config_dir(version_base=None, config_dir=os.path.abspath("examples/configs")):
    cfg = compose(config_name="main", overrides=["task=tool_hang_ph_state_delta_legacy",
        "+task.dataset_path=" + DSP, "network=chiunet",
        "optimization.loss_type=regression", "optimization.auto_resume=false", "log.wandb_mode=disabled"])
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
    g = a[:, 6]
    c1 = sustained(g)
    o1 = None if c1 is None else (lambda x: x + c1 if x is not None else None)(sustained(-g[c1:]))
    c2 = None if o1 is None else (lambda x: x + o1 if x is not None else None)(sustained(g[o1:]))
    o2 = None if c2 is None else (lambda x: x + c2 if x is not None else None)(sustained(-g[c2:]))
    demo_meta.append({"T": len(a), "c1": c1, "o1": o1, "c2": c2, "o2": o2})
h.close()
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
phases = []
for i in idx:
    b0, b1, s0, s1 = ds.sampler.indices[int(i)]
    d = int(np.searchsorted(ee, b0, side="right"))
    t = min(max(int(b0 - starts[d] + (1 - s0)), 0), demo_meta[d]["T"] - 1)
    phases.append(phase(d, t))
phases = np.array(phases)

def get_batch(ids):
    xs, ys = [], []
    for i in ids:
        b = ds[int(i)]
        o = b["obs"]["state"] if isinstance(b["obs"], dict) else b["obs"]
        xs.append(o[:2]); ys.append(b["action"])
    return torch.stack(xs).to(dev), torch.stack(ys).to(dev)

Xs, Ys = get_batch(idx)
flatX = Xs.reshape(N, -1).cpu().numpy(); flatY = Ys.reshape(N, -1).cpu().numpy()
D = torch.cdist(torch.tensor(flatX), torch.tensor(flatX))
nbr = D.topk(9, largest=False).indices[:, 1:].numpy()
knn_dev = np.linalg.norm(flatY - flatY[nbr].mean(1), axis=1)
typ = knn_dev <= np.median(knn_dev)
tail = knn_dev >= np.quantile(knn_dev, 0.9)
servo = (phases == "insert_frame") & typ
hangm = (phases == "hang_tool") & typ

RUN = os.environ["RUN"]; TAG = os.environ.get("TAG", RUN)
for step in os.environ["STEPS"].split():
    ck = f"logs/{RUN}/models/snap_{step}.pt"
    if not os.path.exists(ck):
        print(f"FIT {TAG} {step} MISSING"); continue
    ag.load(ck, load_optimizer=False)
    ag.encoder.eval(); ag.flow_map.eval()
    mses = []
    with torch.no_grad():
        for b0 in range(0, N, 256):
            xb, yb = get_batch(idx[b0:b0 + 256])
            emb = ag.encoder(xb, None)
            t0 = torch.zeros(len(xb), device=dev)
            pred = ag.flow_map.get_velocity(t0, torch.zeros_like(yb), emb)
            mses.append(((pred - yb) ** 2).mean(dim=(1, 2)).cpu().numpy())
    m = np.concatenate(mses)
    q = lambda sel: float(np.median(m[sel]))
    print(f"FIT {TAG} step={step} | servo={q(servo):.5f} (n={servo.sum()}) | hang={q(hangm):.5f} "
          f"| typ={q(typ):.5f} | tail={q(tail):.5f} | all={float(np.median(m)):.5f}", flush=True)
print("FIT-DONE")
