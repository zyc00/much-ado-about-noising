"""Conflict of the TEACHER signal vs the DATA signal at the same near-twin
clusters: spread of ATK-teacher predictions across twin states vs spread of
ground-truth chunks. Prints TC lines."""
import os
import sys

os.environ["MUJOCO_GL"] = "egl"
sys.path.insert(0, "scripts")
sys.path.insert(0, ".")
import h5py
import numpy as np
import torch
from hydra import compose, initialize_config_dir
from omegaconf import OmegaConf

from mip.agent import TrainingAgent
from mip.datasets.robomimic_dataset import make_dataset
from mip.samplers import get_sampler

D2 = "data/tool_hang_full2ins_mp_200.hdf5"
OK = ["object", "robot0_eef_pos", "robot0_eef_quat", "robot0_gripper_qpos"]
with initialize_config_dir(version_base=None,
                           config_dir=os.path.abspath("examples/configs")):
    cfg = compose(config_name="main", overrides=[
        "task=tool_hang_ph_state_delta_legacy",
        "+task.dataset_path=" + os.path.abspath(D2), "network=chiunet",
        "optimization.loss_type=mip_nonoise_atk",
        "optimization.auto_resume=false", "log.wandb_mode=disabled"])
OmegaConf.set_struct(cfg, False)
cfg.task.obs_dim = 53
cfg.task.horizon = int(2 ** np.ceil(np.log2(cfg.task.horizon)))
Hn = int(cfg.task.horizon)
ds = make_dataset(cfg.task)
ag = TrainingAgent(cfg)
ag.load("logs/mp200_nonoise_atk/models/snap_300000.pt", load_optimizer=False)
ag.eval()
sampler = get_sampler("mip_nonoise_atk")
no, na = ds.normalizer["obs"]["state"], ds.normalizer["action"]
dev = cfg.optimization.device

h = h5py.File(D2, "r")
names = sorted(h["data"].keys(), key=lambda d: int(d.split("_")[1]))
PW, PA, PD = [], [], []
for di, dn in enumerate(names):
    o = h[f"data/{dn}/obs"]
    S = np.concatenate([np.asarray(o[k]) for k in OK], 1).astype(np.float32)
    A = np.asarray(h[f"data/{dn}/actions"], dtype=np.float32)
    for i in range(1, len(S) - Hn):
        PW.append(np.stack([S[i - 1], S[i]]))
        PA.append(A[i:i + 8, 0:3])
        PD.append(di)
h.close()
PWn = np.stack([no.normalize(w).reshape(-1) for w in PW])
PA, PD = np.stack(PA), np.asarray(PD)
tP = torch.tensor(PWn)
rng = np.random.default_rng(0)
qidx = rng.choice(np.where(PD < 8)[0], 300, replace=False)


def teach(ws):
    xb = torch.tensor(np.stack([no.normalize(w) for w in ws]), device=dev,
                      dtype=torch.float32)
    with torch.no_grad():
        a0 = torch.zeros((len(xb), Hn, 10), device=dev)
        an = sampler(cfg.optimization, ag.flow_map_ema, ag.encoder_ema, a0,
                     {"state": xb})
    return np.asarray(ds.undo_transform_action(
        na.unnormalize(an.cpu().numpy())[:, 1:9]))[:, :, 0:3]


dconf, tconf = [], []
for qi in qidx:
    d = torch.norm(tP - torch.tensor(PWn[qi]), dim=1).numpy()
    d[PD == PD[qi]] = 1e9
    nb = np.where(d < 0.5)[0]
    if len(nb) < 4:
        continue
    T = PA[nb].reshape(len(nb), -1)
    dconf.append(float(np.sqrt(((T - T.mean(0)) ** 2).mean())))
    P = teach([PW[j] for j in nb]).reshape(len(nb), -1)
    tconf.append(float(np.sqrt(((P - P.mean(0)) ** 2).mean())))
dconf, tconf = np.asarray(dconf), np.asarray(tconf)
print(f"TC clusters {len(dconf)} | DATA conflict p50 {np.median(dconf):.4f} "
      f"p90 {np.percentile(dconf, 90):.4f} | TEACHER conflict p50 "
      f"{np.median(tconf):.4f} p90 {np.percentile(tconf, 90):.4f} | ratio "
      f"p50 {np.median(tconf / (dconf + 1e-9)):.2f}", flush=True)
print("TC done", flush=True)
