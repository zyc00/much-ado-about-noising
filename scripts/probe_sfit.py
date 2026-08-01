"""Scripted (MP-200) fit-allocation anatomy: per-arm per-state residual along
training demos, stratified by progress decile; pockets = deciles {3,4,8,9}
(contact + settle). Quantitative test of the unbalanced-fitting account:
predicts pocket-residual ordering L2 > HT > HG > MIP (tracking SR 63/75/87/94)
with HT/HG pocket/nonpocket ratio >> MIP's. Prints SFIT lines."""
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
    ("NONOISE", "mip_nonoise", "logs/mp200_nonoise_s1000/models/snap_300000.pt"),
    ("MIP", "mip", "logs/mp200_mip_s1000/models/snap_300000.pt"),
]
NDEMO = int(os.environ.get("SF_NDEMO", "8"))
AS_LO, AS_HI = 1, 9
POCKET = {3, 4, 8, 9}

h = h5py.File(D2, "r")
names = sorted(h["data"].keys(), key=lambda d: int(d.split("_")[1]))
demos = []
for dn in names[:NDEMO]:
    o = h[f"data/{dn}/obs"]
    S = np.concatenate([np.asarray(o[k]) for k in OK], 1).astype(np.float32)
    A = np.asarray(h[f"data/{dn}/actions"], dtype=np.float32)
    demos.append((dn, S, A))
h.close()
print(f"SFIT demos {NDEMO} lens {[len(s) for _, s, _ in demos]}", flush=True)

for arm, loss, ck in ARMS:
    with initialize_config_dir(version_base=None,
                               config_dir=os.path.abspath("examples/configs")):
        cfg = compose(config_name="main", overrides=[
            "task=tool_hang_ph_state_delta_legacy",
            "+task.dataset_path=" + os.path.abspath(D2), "network=chiunet",
            f"optimization.loss_type={loss}", "optimization.auto_resume=false",
            "log.wandb_mode=disabled"])
    OmegaConf.set_struct(cfg, False)
    cfg.task.obs_dim = 53
    cfg.task.horizon = int(2 ** np.ceil(np.log2(cfg.task.horizon)))
    Hn = int(cfg.task.horizon)
    ds = make_dataset(cfg.task)
    ag = TrainingAgent(cfg)
    ag.load(ck, load_optimizer=False)
    ag.eval()
    sampler = get_sampler(loss)
    no, na = ds.normalizer["obs"]["state"], ds.normalizer["action"]
    dev = cfg.optimization.device

    R, DEC = [], []
    for dn, S, A in demos:
        L = len(S)
        hi = L - Hn
        W = np.stack([np.stack([S[i - 1], S[i]]) for i in range(1, hi)])
        GA = np.stack([A[i - 1 + AS_LO:i - 1 + AS_HI, 0:3]
                       for i in range(1, hi)])
        for i in range(0, len(W), 256):
            xb = torch.tensor(np.stack([no.normalize(w) for w in W[i:i + 256]]),
                              device=dev, dtype=torch.float32)
            with torch.no_grad():
                a0 = torch.zeros((len(xb), Hn, 10), device=dev)
                an = sampler(cfg.optimization, ag.flow_map_ema,
                             ag.encoder_ema, a0, {"state": xb})
            pe = np.asarray(ds.undo_transform_action(
                na.unnormalize(an.cpu().numpy())[:, AS_LO:AS_HI]))
            R.append(np.abs(pe[:, :, 0:3] - GA[i:i + 256]).mean(axis=(1, 2)))
        DEC.append(np.minimum((10 * np.arange(1, hi) / L).astype(int), 9))
    R, DEC = np.concatenate(R), np.concatenate(DEC)
    pk = np.isin(DEC, list(POCKET))
    dl = " ".join(f"d{d} {np.median(R[DEC == d]):.5f}" for d in range(10))
    print(f"SFIT {arm} perdecile {dl}", flush=True)
    print(f"SFIT {arm} pocket p50 {np.median(R[pk]):.5f} nonpocket "
          f"{np.median(R[~pk]):.5f} ratio "
          f"{np.median(R[pk]) / np.median(R[~pk]):.2f} | pocket loss-share "
          f"{(R[pk] ** 2).sum() / (R ** 2).sum():.2f}", flush=True)
print("SFIT done", flush=True)
