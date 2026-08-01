"""Chunk-tail robustness at slightly-off states: perturb the (normalized) obs
window by Gaussian noise of norm delta in {0, 0.5, 1, 2} (tube edge ~2) and
measure per-step-index chunk error vs the GT continuation (pos channels,
normalized units). If L2's error grows with index j faster than HG's, open-loop
execution (AS=8) integrates the tail error while AS=1 only pays j=1 —
explaining SR 75(AS8) -> 99(AS1). Anchors from the ds sampler (r in [-8,30)).
Envs: CKPT, LOSS, TAG, ACT_DIM."""
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

with initialize_config_dir(version_base=None, config_dir=os.path.abspath("examples/configs")):
    cfg = compose(config_name="main", overrides=["task=tool_hang_ph_state_delta_legacy",
        "+task.dataset_path=data/tool_hang_full2ins_2000.hdf5", "network=chiunet",
        f"optimization.loss_type={os.environ['LOSS']}", "optimization.auto_resume=false", "log.wandb_mode=disabled"])
OmegaConf.set_struct(cfg, False); cfg.task.obs_dim = 53; cfg.task.horizon = 16
if int(os.environ.get("ACT_DIM", "10")) == 11:
    cfg.task.act_dim = 11; cfg.task.progress_indicator = True
ds = make_dataset(cfg.task)
ag = TrainingAgent(cfg)
dev = cfg.optimization.device

h = h5py.File("data/tool_hang_full2ins_2000.hdf5", "r")
keys = sorted(h["data"].keys(), key=lambda k: int(k.split("_")[-1]))
NDEMO = 150
c1s = []
for k in keys[:NDEMO]:
    a = np.asarray(h[f"data/{k}/actions"])
    g = a[:, 6]
    cl = [t for t in range(1, len(a)) if g[t - 1] < 0 and g[t] >= 0]
    c1s.append(cl[0] if cl else -10**6)
h.close()

ee = ds.sampler.replay_buffer.episode_ends[:]
starts = np.concatenate([[0], ee[:-1]])
idxs = ds.sampler.indices
anchors = []
for i in range(len(idxs)):
    b0, b1, s0, s1 = idxs[i]
    d = int(np.searchsorted(ee, b0, side="right"))
    if d >= NDEMO: break
    t = int(b0 - starts[d] + (1 - s0))
    r = t - c1s[d]
    if -8 <= r < 30 and t % 4 == 0:
        anchors.append(i)
rng = np.random.RandomState(0)
if len(anchors) > 400:
    anchors = [anchors[j] for j in rng.choice(len(anchors), 400, replace=False)]

X, Y = [], []
for i in anchors:
    b = ds[int(i)]
    o = b["obs"]["state"] if isinstance(b["obs"], dict) else b["obs"]
    X.append(o[:2] if o.shape[0] > 2 else o)
    Y.append(b["action"])
X = torch.stack(X); Y = torch.stack(Y)
EPS = torch.tensor(rng.randn(*X.shape).astype(np.float32))
EPS = EPS / EPS.reshape(len(X), -1).norm(dim=1).reshape(-1, 1, 1)

ag.load(os.environ["CKPT"], load_optimizer=False)
TAG = os.environ.get("TAG", "ckpt")

def run(delta):
    outs = []
    with torch.no_grad():
        for b0 in range(0, len(X), 256):
            wb = (X[b0:b0 + 256] + delta * EPS[b0:b0 + 256]).to(dev)
            an = ag.sample(act_0=torch.randn((len(wb), 16, cfg.task.act_dim), device=dev),
                           obs={"state": wb}, use_ema=True)
            outs.append(an[:, :, :3].cpu())
    P = torch.cat(outs)
    E = (P - Y[:, :, :3]).norm(dim=2).numpy()   # (N,16) per-index pos error
    return E

for delta in (0.0, 0.5, 1.0, 2.0):
    E = run(delta)
    j1 = float(np.median(E[:, 1])); j18 = float(np.median(E[:, 1:9].mean(axis=1)))
    j15 = float(np.median(E[:, 15])); grow = j18 / max(j1, 1e-9)
    print(f"TAILROB {TAG} delta={delta}: err j=1 p50={j1:.4f} | mean j=1..8 p50={j18:.4f} "
          f"| j=15 p50={j15:.4f} | exec-window/head ratio={grow:.2f} | "
          f"profile: " + " ".join(f"{np.median(E[:, j]):.3f}" for j in range(0, 16, 2)), flush=True)
print("TAILROB-DONE")
