"""NN phase composition of off-tube states (d>=2) from twofactor DUMPTRAJ dumps.
Envs: CKPT, LOSS, QD (dump dir), TAG."""
import glob
import os
import sys

os.environ["MUJOCO_GL"] = "egl"
import h5py
import numpy as np
import torch

sys.path.insert(0, "scripts"); sys.path.insert(0, ".")
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
keys = sorted(h["data"].keys(), key=lambda k: int(k.split("_")[-1]))
W, phase = [], []
for k in keys[:150]:
    o = h[f"data/{k}/obs"]
    ov_ = np.concatenate([np.asarray(o[q]) for q in OK], axis=1).astype(np.float32)
    a = np.clip(np.asarray(h[f"data/{k}/actions"]), -1, 1); g = a[:, 6]
    cl = [t for t in range(1, len(g)) if g[t-1] < 0 and g[t] >= 0]
    if not cl: continue
    c1 = cl[0]
    for t in range(1, len(g), 2):
        r = t - c1
        W.append(np.stack([ov_[t-1], ov_[t]]))
        phase.append(0 if r < -12 else (1 if r < 5 else (2 if r < 70 else 3)))
h.close()
W = np.stack(W); phase = np.array(phase)
ag.load(os.environ["CKPT"], load_optimizer=False)
ag.encoder.eval()

def embed(batch):
    out = []
    with torch.no_grad():
        for b0 in range(0, len(batch), 512):
            wb = torch.tensor(no.normalize(batch[b0:b0 + 512]), device=dev, dtype=torch.float32)
            out.append(ag.encoder(wb, None).reshape(len(wb), -1).cpu())
    e = torch.cat(out)
    return e / (e.norm(dim=1, keepdim=True) + 1e-8)

E_bank = embed(W)
TAG = os.environ.get("TAG", "ckpt")
for label, want_asm in [("failEP", 0), ("succEP", 1)]:
    Q, dd = [], []
    for f in glob.glob(os.environ["QD"] + "/ep_*.npz"):
        z = np.load(f)
        if int(z["asm"]) != want_asm: continue
        dser = z["dser"]; obsw = z["obsw"]
        off = np.where(dser >= 2.0)[0]
        for t in off[::5][:20]:
            Q.append(obsw[t]); dd.append(float(dser[t]))
    if not Q:
        print(f"FAILQ {TAG} {label}: no off-tube states")
        continue
    Q = np.stack(Q)
    Eq = embed(Q)
    sims = Eq @ E_bank.T
    nn = sims.topk(10, dim=1).indices.numpy()
    comp = np.array([(phase[nn] == p).mean() for p in range(4)])
    print(f"FAILQ {TAG} {label} n={len(Q)} dmed={np.median(dd):.1f} | NN appr/settle/lift-ins/post = "
          f"{comp[0]:.0%}/{comp[1]:.0%}/{comp[2]:.0%}/{comp[3]:.0%}", flush=True)
print("FAILQ-DONE")
