"""In-traj vs cross-traj injury decomposition: per-state (a) in-trajectory
steepness ||dA||/||dS||, (b) cross-demo twin conflict, vs stored L2@60k
per-state residuals (loss_traj.npz, demos 0-3). Prints IJ lines."""
import os
import sys

sys.path.insert(0, ".")
import h5py
import numpy as np
import torch
from hydra import compose, initialize_config_dir
from omegaconf import OmegaConf
from scipy.stats import spearmanr

os.environ["MUJOCO_GL"] = "egl"
from mip.datasets.robomimic_dataset import make_dataset

D2 = "data/tool_hang_full2ins_mp_200.hdf5"
OK = ["object", "robot0_eef_pos", "robot0_eef_quat", "robot0_gripper_qpos"]
with initialize_config_dir(version_base=None,
                           config_dir=os.path.abspath("examples/configs")):
    cfg = compose(config_name="main", overrides=[
        "task=tool_hang_ph_state_delta_legacy",
        "+task.dataset_path=" + os.path.abspath(D2), "network=chiunet",
        "optimization.auto_resume=false", "log.wandb_mode=disabled"])
OmegaConf.set_struct(cfg, False)
cfg.task.obs_dim = 53
ds = make_dataset(cfg.task)
no = ds.normalizer["obs"]["state"]

z = np.load("analysis/manifold/loss_traj.npz")
h = h5py.File(D2, "r")
names = sorted(h["data"].keys(), key=lambda d: int(d.split("_")[1]))
PW, PD = [], []
for di, dn in enumerate(names):
    o = h[f"data/{dn}/obs"]
    S = np.concatenate([np.asarray(o[k]) for k in OK], 1).astype(np.float32)
    for i in range(1, len(S) - 16):
        PW.append(no.normalize(np.stack([S[i - 1], S[i]])).reshape(-1))
        PD.append(di)
PWa = np.stack(PW)
PDa = np.asarray(PD)
tP = torch.tensor(PWa)
PA = []
for dn in names:
    PA.append(np.asarray(h[f"data/{dn}/actions"], dtype=np.float32))

R_all, ST_all, CF_all, GR_all, RT_all = [], [], [], [], []
for di in range(4):
    dn = names[di]
    key = f"L260000_{dn}_r"
    if key not in z.files:
        continue
    r = z[key]
    o = h[f"data/{dn}/obs"]
    S = np.concatenate([np.asarray(o[k]) for k in OK], 1).astype(np.float32)
    A = PA[di]
    Sn = no.normalize(S)
    astd = A.std(0) + 1e-6
    L = len(r)
    for i in range(1, L):
        ds_ = np.linalg.norm(Sn[i + 1] - Sn[i - 1]) + 1e-6
        da_ = np.linalg.norm((A[i + 1] - A[i - 1]) / astd)
        ST_all.append(da_ / ds_)
        GR_all.append(float(abs(A[i + 1, 6] - A[i - 1, 6]) > 0.5))
        RT_all.append(float(np.abs(A[i, 3:6]).sum() > 0.5))
        R_all.append(r[i])
h.close()
R = np.asarray(R_all)
ST = np.asarray(ST_all)
GR = np.asarray(GR_all)
RT = np.asarray(RT_all)
rho_st = spearmanr(R, ST).statistic
print(f"IJ n={len(R)} | spearman(resid, in-traj steep) {rho_st:+.3f}",
      flush=True)
q90 = ST >= np.percentile(ST, 90)
print(f"IJ loss share of top-10% STEEP states: {R[q90].sum() / R.sum():.2f} "
      f"(share of states .10)", flush=True)
print(f"IJ steep-state composition: gripper-event {GR[q90].mean():.2f} "
      f"rot-active {RT[q90].mean():.2f} (base rates {GR.mean():.2f} "
      f"{RT.mean():.2f})", flush=True)
print(f"IJ resid p50 steep-top10 {np.median(R[q90]):.2e} vs rest "
      f"{np.median(R[~q90]):.2e}", flush=True)
print("IJ done", flush=True)
