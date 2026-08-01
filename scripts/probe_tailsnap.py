"""Tail-snap probe: at off-support states between settle and mid-stroke, does
the CHUNK TAIL snap to the stroke before the head does? Reports per-step-index
|a_pos| p50 of the sampled chunk at interpolated states (alpha in {0.3,0.5})
and at the SET endpoint. If L2's tail snaps while its head stays quiet, AS=8
executes the snapped tail (75) while AS=1 only executes the quiet head (99).
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

OK = ["object", "robot0_eef_pos", "robot0_eef_quat", "robot0_gripper_qpos"]
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
no = ds.normalizer["obs"]["state"]

h = h5py.File("data/tool_hang_full2ins_2000.hdf5", "r")
keys = sorted(h["data"].keys(), key=lambda k: int(k.split("_")[-1]))[:150]
W1, W3 = [], []
for k in keys:
    o = h[f"data/{k}/obs"]
    ov_ = np.concatenate([np.asarray(o[q]) for q in OK], axis=1).astype(np.float32)
    a = np.clip(np.asarray(h[f"data/{k}/actions"]), -1, 1).astype(np.float32)
    g = a[:, 6]; T = len(a)
    cl = [t for t in range(1, T) if g[t - 1] < 0 and g[t] >= 0]
    if not cl: continue
    c1 = cl[0]
    if c1 - 7 >= 1 and c1 + 25 < T:
        W1.append(np.stack([ov_[c1 - 7], ov_[c1 - 6]]))
        W3.append(np.stack([ov_[c1 + 23], ov_[c1 + 24]]))
h.close()
W1 = np.stack(W1); W3 = np.stack(W3)

ag.load(os.environ["CKPT"], load_optimizer=False)
ag.eval() if hasattr(ag, "eval") else None
TAG = os.environ.get("TAG", "ckpt")

def chunks(Wq):
    outs = []
    with torch.no_grad():
        for b0 in range(0, len(Wq), 256):
            wb = torch.tensor(no.normalize(Wq[b0:b0 + 256]), device=dev, dtype=torch.float32)
            an = ag.sample(act_0=torch.randn((len(wb), 16, cfg.task.act_dim), device=dev),
                           obs={"state": wb}, use_ema=True)
            outs.append(an[:, :, :3].cpu().numpy())
    return np.concatenate(outs)

for nm, Wq in [("SET(a=0)", W1), ("a=0.3", 0.7 * W1 + 0.3 * W3), ("a=0.5", 0.5 * W1 + 0.5 * W3),
               ("MID(a=1)", W3)]:
    A = chunks(Wq)
    prof = np.median(np.linalg.norm(A, axis=2), axis=0)
    head = float(np.median(np.linalg.norm(A[:, 1], axis=1)))
    tail = float(np.median(np.linalg.norm(A[:, 1:9], axis=2).max(axis=1)))
    print(f"TAILSNAP {TAG} {nm}: head(j=1)={head:.3f} maxexec(j=1..8)={tail:.3f} | profile p50 by j: " +
          " ".join(f"{p:.2f}" for p in prof), flush=True)
print("TAILSNAP-DONE")
