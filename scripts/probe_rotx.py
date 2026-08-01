"""Cross-arm rotation-command comparison at IDENTICAL states: HT's
pre-onset rollout states, evaluated by HT/HG/MIP/L2. Prints RX lines."""
import glob
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
ARMS = [
    ("L2", "regression", "logs/mp200_l2_s1000/models/snap_300000.pt"),
    ("HT", "regression_hetero_t", "logs/mp200_ht_s1000/models/snap_300000.pt"),
    ("HG", "regression_hetero_gauss",
     "logs/mp200_hg_s1000/models/snap_300000.pt"),
    ("MIP", "mip", "logs/mp200_mip_s1000/models/snap_300000.pt"),
]
ONSETS = {21001: 143, 21002: 167, 21007: 201, 21010: 57, 21012: 9,
          21013: 142, 21015: 177, 21022: 7}

with initialize_config_dir(version_base=None,
                           config_dir=os.path.abspath("examples/configs")):
    cfg0 = compose(config_name="main", overrides=[
        "task=tool_hang_ph_state_delta_legacy",
        "+task.dataset_path=" + os.path.abspath(D2), "network=chiunet",
        "optimization.auto_resume=false", "log.wandb_mode=disabled"])
OmegaConf.set_struct(cfg0, False)
cfg0.task.obs_dim = 53
cfg0.task.horizon = int(2 ** np.ceil(np.log2(cfg0.task.horizon)))
Hn = int(cfg0.task.horizon)
ds0 = make_dataset(cfg0.task)
no, na0 = ds0.normalizer["obs"]["state"], ds0.normalizer["action"]
AS = int(cfg0.task.act_steps)
start = int(cfg0.task.obs_steps) - 1

h = h5py.File(D2, "r")
names = sorted(h["data"].keys(), key=lambda d: int(d.split("_")[1]))
PW, PA = [], []
for dn in names:
    o = h[f"data/{dn}/obs"]
    S = np.concatenate([np.asarray(o[k]) for k in OK], 1).astype(np.float32)
    A = np.asarray(h[f"data/{dn}/actions"], dtype=np.float32)
    for i in range(1, len(S) - Hn):
        PW.append(no.normalize(np.stack([S[i - 1], S[i]])).reshape(-1))
        PA.append(A[i - 1 + start:i - 1 + start + AS])
h.close()
PW, PA = np.stack(PW), np.stack(PA)
tP = torch.tensor(PW)

W, REF = [], []
for f in sorted(glob.glob("logs/rd_ht/ep_*.npz")):
    z = np.load(f)
    if int(z["asm"]):
        continue
    obs, seed = z["obs"], int(z["seed"])
    onset = ONSETS[seed]
    for i in range(max(1, onset - 30), onset + 2, 2):
        w = np.stack([obs[i - 1], obs[i]])
        q = torch.tensor(no.normalize(w).reshape(-1))
        j = int(torch.norm(tP - q, dim=1).argmin())
        W.append(w)
        REF.append(PA[j])
W, REF = np.stack(W), np.stack(REF)
print(f"RX states {len(W)}", flush=True)

for arm, loss, ck in ARMS:
    cfg = OmegaConf.create(OmegaConf.to_container(cfg0))
    cfg.optimization.loss_type = loss
    ag = TrainingAgent(cfg)
    ag.load(ck, load_optimizer=False)
    ag.eval()
    sampler = get_sampler(loss)
    dev = cfg.optimization.device
    E = []
    for i in range(0, len(W), 128):
        xb = torch.tensor(np.stack([no.normalize(w) for w in W[i:i + 128]]),
                          device=dev, dtype=torch.float32)
        with torch.no_grad():
            a0 = torch.zeros((len(xb), Hn, 10), device=dev)
            an = sampler(cfg.optimization, ag.flow_map_ema, ag.encoder_ema,
                         a0, {"state": xb})
        pe = np.asarray(ds0.undo_transform_action(
            na0.unnormalize(an.cpu().numpy())[:, start:start + AS]))
        for k in range(len(pe)):
            r = REF[i + k]
            E.append((np.abs(pe[k, :, 0:3] - r[:, 0:3]).mean(),
                      np.abs(pe[k, :, 3:6] - r[:, 3:6]).mean(),
                      np.abs(pe[k, :, 3:6]).sum() /
                      (np.abs(r[:, 3:6]).sum() + 1e-6)))
    E = np.asarray(E)
    print(f"RX {arm} pos p50 {np.median(E[:, 0]):.5f} rot p50 "
          f"{np.median(E[:, 1]):.5f} p90 {np.percentile(E[:, 1], 90):.5f} "
          f"atten {np.median(E[:, 2]):.2f}", flush=True)
print("RX done", flush=True)
