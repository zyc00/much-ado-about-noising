"""Self-purity fold probe over DATASET states (no rollout queries): phase-labeled
bank from full2ins (same conventions as foldsweep: phase by time-to-closure:
0=approach(r<-12), 1=settle(-12..5), 2=lift/ins(5..70), 3=post), embed with a
checkpoint, report settle-state 10-NN phase composition + within-phase metric
alignment. Envs: CKPT, LOSS, TAG."""
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
keys = sorted(h["data"].keys(), key=lambda k: int(k.split("_")[-1]))
W, phase = [], []
for k in keys[:150]:
    o = h[f"data/{k}/obs"]
    ov = np.concatenate([np.asarray(o[q]) for q in OK], axis=1).astype(np.float32)
    a = np.clip(np.asarray(h[f"data/{k}/actions"]), -1, 1); g = a[:, 6]
    cl = [t for t in range(1, len(g)) if g[t-1] < 0 and g[t] >= 0]
    if not cl: continue
    c1 = cl[0]
    for t in range(1, len(g), 2):
        r = t - c1
        ph = 0 if r < -12 else (1 if r < 5 else (2 if r < 70 else 3))
        W.append(np.stack([ov[t-1], ov[t]])); phase.append(ph)
h.close()
W = np.stack(W); phase = np.array(phase)
ag.load(os.environ["CKPT"], load_optimizer=False)
ag.encoder.eval()
embs = []
with torch.no_grad():
    for b0 in range(0, len(W), 512):
        wb = torch.tensor(no.normalize(W[b0:b0+512]), device=dev, dtype=torch.float32)
        embs.append(ag.encoder(wb, None).reshape(len(wb), -1).cpu())
E = torch.cat(embs)
E = E / (E.norm(dim=1, keepdim=True) + 1e-8)
TAG = os.environ.get("TAG", "ckpt")
S = torch.tensor(np.where(phase == 1)[0])
sims = E[S] @ E.T
for i, s in enumerate(S): sims[i, s] = -2
nn = sims.topk(10, dim=1).indices.numpy()
comp = np.array([(phase[nn] == p).mean() for p in range(4)])
print(f"PURITY {TAG}: settle 10-NN composition appr/settle/lift-ins/post = "
      f"{comp[0]:.0%}/{comp[1]:.0%}/{comp[2]:.0%}/{comp[3]:.0%} (n_settle={len(S)}, bank={len(W)})", flush=True)
print("PURITY-DONE")
