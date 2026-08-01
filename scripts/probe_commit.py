"""Generic feature-commitment audit: local encoder-Jacobian participation ratio
(PR of squared singular values), kappa10 = s1/s10, and top-column Frobenius
share, at QUIET (|a_pos|<q40) vs LOUD (>q60) states. Answers whether the
trained encoder commits to ~one feature in the regions where labels are quiet,
on any task/dataset. Envs: TASK, CKPT, LOSS, DSP, TAG, OBSDIM (optional)."""
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

TASK = os.environ["TASK"]; DSP = os.environ["DSP"]; TAG = os.environ["TAG"]
with initialize_config_dir(version_base=None, config_dir=os.path.abspath("examples/configs")):
    cfg = compose(config_name="main", overrides=[f"task={TASK}", "network=chiunet",
        f"+task.dataset_path={DSP}",
        f"optimization.loss_type={os.environ.get('LOSS','regression')}",
        "optimization.auto_resume=false", "log.wandb_mode=disabled"])
OmegaConf.set_struct(cfg, False)
h = h5py.File(DSP, "r")
k0 = sorted(h["data"].keys())[0]
two = "robot1_eef_pos" in h[f"data/{k0}/obs"]
OK = ["object", "robot0_eef_pos", "robot0_eef_quat", "robot0_gripper_qpos"] + \
     (["robot1_eef_pos", "robot1_eef_quat", "robot1_gripper_qpos"] if two else [])
n0 = len(h[f"data/{k0}/actions"])
obs_dim = sum(np.asarray(h[f"data/{k0}/obs/{q}"]).reshape(n0, -1).shape[1] for q in OK)
cfg.task.obs_dim = int(os.environ.get("OBSDIM", obs_dim))
ds = make_dataset(cfg.task)
ag = TrainingAgent(cfg)
ag.load(os.environ["CKPT"], load_optimizer=False)
ag.encoder.eval()
no = ds.normalizer["obs"]["state"]
dev = cfg.optimization.device

keys = sorted(h["data"].keys(), key=lambda k: int(k.split("_")[-1]))[:120]
W, AM = [], []
for k in keys:
    o = h[f"data/{k}/obs"]
    n = len(h[f"data/{k}/actions"])
    ov_ = np.concatenate([np.asarray(o[q]).reshape(n, -1) for q in OK], axis=1).astype(np.float32)
    a = np.clip(np.asarray(h[f"data/{k}/actions"]), -1, 1).astype(np.float32)
    pos = np.concatenate([a[:, :3], a[:, 7:10]], 1) if two else a[:, :3]
    m = np.linalg.norm(pos, axis=1)
    for t in range(1, n - 2, 3):
        W.append(np.stack([ov_[t - 1], ov_[t]])); AM.append(m[t])
h.close()
W = np.stack(W); AM = np.array(AM)
rng = np.random.RandomState(0)
qthr, lthr = np.quantile(AM, 0.4), np.quantile(AM, 0.6)
qi = rng.permutation(np.where(AM < qthr)[0])[:120]
li = rng.permutation(np.where(AM > lthr)[0])[:120]

def stats(idx):
    prs, k10s, colsh = [], [], []
    for i in idx:
        w = torch.tensor(no.normalize(W[i:i + 1]), device=dev, dtype=torch.float32, requires_grad=True)
        e = ag.encoder(w, None).reshape(-1)
        J = torch.autograd.functional.jacobian(lambda x: ag.encoder(x, None).reshape(-1), w, vectorize=True)
        J = J.reshape(e.numel(), -1).detach()
        s = torch.linalg.svdvals(J)
        s2 = (s ** 2)
        prs.append(float(s2.sum() ** 2 / (s2 ** 2).sum()))
        k10s.append(float(s[0] / (s[min(9, len(s) - 1)] + 1e-12)))
        cn = (J ** 2).sum(0)
        colsh.append(float(cn.max() / cn.sum()))
    return np.median(prs), np.median(k10s), np.median(colsh)

pq = stats(qi); pl = stats(li)
print(f"COMMIT {TAG} (obs_dim={cfg.task.obs_dim}) QUIET: PR={pq[0]:.1f} kappa10={pq[1]:.1f} topcol={pq[2]:.2f} | "
      f"LOUD: PR={pl[0]:.1f} kappa10={pl[1]:.1f} topcol={pl[2]:.2f}", flush=True)
print("COMMIT-DONE")
