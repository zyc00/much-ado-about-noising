"""Transport content-fit: per-sample readout MSE f(s,0,0) by group, over a
checkpoint ladder. Groups: place_servo = place-stage & low kNN-dev (SR-critical:
conversion), handover & low-dev, typ = low-dev, tail = top-decile dev.
Envs: TASK, RUN, STEPS, TAG, OBS_STEPS."""
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
    demo_meta.append({"T": len(a), "c0": c0, "c1": c1, "o0": o0})
h.close()
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
phases = []
for i in idx:
    b0, b1, s0, s1 = ds.sampler.indices[int(i)]
    d = int(np.searchsorted(ee, b0, side="right"))
    t = min(max(int(b0 - starts[d] + (OS - 1 - s0)), 0), demo_meta[d]["T"] - 1)
    phases.append(phase(d, t))
phases = np.array(phases)

def get_batch(ids):
    xs, ys = [], []
    for i in ids:
        b = ds[int(i)]
        o = b["obs"]["state"] if isinstance(b["obs"], dict) else b["obs"]
        xs.append(o[:OS]); ys.append(b["action"])
    return torch.stack(xs).to(dev), torch.stack(ys).to(dev)

Xs, Ys = get_batch(idx)
flatX = Xs.reshape(N, -1).cpu().numpy(); flatY = Ys.reshape(N, -1).cpu().numpy()
D = torch.cdist(torch.tensor(flatX), torch.tensor(flatX))
nbr = D.topk(9, largest=False).indices[:, 1:].numpy()
knn_dev = np.linalg.norm(flatY - flatY[nbr].mean(1), axis=1)
typ = knn_dev <= np.median(knn_dev)
tail = knn_dev >= np.quantile(knn_dev, 0.9)
place_servo = (phases == "place") & typ
hand = (phases == "handover") & typ

RUN = os.environ["RUN"]; TAG = os.environ.get("TAG", RUN)
for step in os.environ["STEPS"].split():
    ck = f"logs/{RUN}/models/snap_{step}.pt"
    if not os.path.exists(ck):
        print(f"FITTP {TAG} {step} MISSING"); continue
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
    print(f"FITTP {TAG} step={step} | place_servo={q(place_servo):.5f} (n={place_servo.sum()}) "
          f"| handover={q(hand):.5f} | typ={q(typ):.5f} | tail={q(tail):.5f}", flush=True)
print("FITTP-DONE")
