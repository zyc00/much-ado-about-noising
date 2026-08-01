"""Does the learned sigma rise off-support? sigma(x) of hetero losses at
settle->midstroke interpolated states (alpha 0 / .3 / .5 / 1). If sigma grows
with alpha, the NLL's precision-weighted fit implements adaptive shrinkage off-
support — the mechanism behind the attenuated/contractive tail.
Envs: CKPT, LOSS, TAG."""
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
ag.encoder.eval(); ag.flow_map.eval()
TAG = os.environ.get("TAG", "ckpt")
AD = cfg.task.act_dim

def sig(Wq):
    out = []
    with torch.no_grad():
        for b0 in range(0, len(Wq), 256):
            wb = torch.tensor(no.normalize(Wq[b0:b0 + 256]), device=dev, dtype=torch.float32)
            emb = ag.encoder(wb, None)
            t0 = torch.zeros(len(wb), device=dev)
            y0 = torch.zeros((len(wb), 16, AD), device=dev)
            _, s_raw = ag.flow_map.net(y0, t0, t0, emb)
            sb = torch.nn.functional.softplus(s_raw).reshape(len(wb), -1).mean(dim=1) + 1e-3
            out.append(sb.cpu().numpy())
    return np.concatenate(out)

# also random-direction offsets at matched norms for comparison
rng = np.random.RandomState(0)
base = sig(W1)
print(f"SIGOFF {TAG} alpha=0 (SET on-tube): sigma p50={np.median(base):.4f} p90={np.percentile(base,90):.4f}", flush=True)
for al in (0.3, 0.5, 1.0):
    s = sig((1 - al) * W1 + al * W3)
    print(f"SIGOFF {TAG} alpha={al}: sigma p50={np.median(s):.4f} ratio-to-onsupport={np.median(s)/np.median(base):.2f}", flush=True)
nrm = np.linalg.norm(no.normalize(0.5 * W1 + 0.5 * W3) - no.normalize(W1), axis=(1, 2))
eps = rng.randn(*W1.shape).astype(np.float32)
eps = eps / np.linalg.norm(eps, axis=(1, 2), keepdims=True)
Wr = no.unnormalize(no.normalize(W1) + nrm[:, None, None] * eps) if hasattr(no, "unnormalize") else None
if Wr is not None:
    s = sig(Wr)
    print(f"SIGOFF {TAG} random-dir matched-norm: sigma p50={np.median(s):.4f} ratio={np.median(s)/np.median(base):.2f}", flush=True)
print("SIGOFF-DONE")
