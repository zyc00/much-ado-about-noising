"""HG failure-onset anatomy: at pre-escape states of failed episodes vs
phase-matched states of successful episodes, measure (1) learned sigma /
effective weight (downweighting hypothesis), (2) neighbor one-sidedness +
error alignment with the one-sided direction (boundary-pull hypothesis).
Prints HE lines."""
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

D2 = "data/tool_hang_full2ins_mp_200.hdf5"
OK = ["object", "robot0_eef_pos", "robot0_eef_quat", "robot0_gripper_qpos"]
K = 8

with initialize_config_dir(version_base=None,
                           config_dir=os.path.abspath("examples/configs")):
    cfg = compose(config_name="main", overrides=[
        "task=tool_hang_ph_state_delta_legacy",
        "+task.dataset_path=" + os.path.abspath(D2), "network=chiunet",
        "optimization.loss_type=regression_hetero_gauss",
        "optimization.auto_resume=false", "log.wandb_mode=disabled"])
OmegaConf.set_struct(cfg, False)
cfg.task.obs_dim = 53
cfg.task.horizon = int(2 ** np.ceil(np.log2(cfg.task.horizon)))
Hn = int(cfg.task.horizon)
ds = make_dataset(cfg.task)
ag = TrainingAgent(cfg)
ag.load("logs/mp200_hg_s1000/models/snap_300000.pt", load_optimizer=False)
ag.eval()
no, na = ds.normalizer["obs"]["state"], ds.normalizer["action"]
dev = cfg.optimization.device

h = h5py.File(D2, "r")
names = sorted(h["data"].keys(), key=lambda d: int(d.split("_")[1]))
PW, PP = [], []
for dn in names:
    o = h[f"data/{dn}/obs"]
    S = np.concatenate([np.asarray(o[k]) for k in OK], 1).astype(np.float32)
    L = len(S)
    for i in range(1, L - Hn):
        PW.append(no.normalize(np.stack([S[i - 1], S[i]])).reshape(-1))
        PP.append(i / L)
h.close()
PW, PP = np.stack(PW), np.asarray(PP)
tP = torch.tensor(PW)


def stats_at(w2):
    q = torch.tensor(no.normalize(w2).reshape(-1))
    d = torch.norm(tP - q, dim=1)
    dk, nk = torch.topk(-d, K)
    nk = nk.numpy()
    NB = PW[nk] - q.numpy()
    nbn = np.linalg.norm(NB, axis=1)
    onesided = float(np.linalg.norm(NB.mean(0)) / (nbn.mean() + 1e-9))
    xb = torch.tensor(no.normalize(w2)[None], device=dev,
                      dtype=torch.float32)
    with torch.no_grad():
        emb = ag.encoder_ema({"state": xb}, None)
        t0 = torch.zeros((1,), device=dev)
        a0 = torch.zeros((1, Hn, 10), device=dev)
        pr, sraw = ag.flow_map_ema.net(a0, t0, t0, emb)
        sig = float(torch.nn.functional.softplus(sraw).mean() + 1e-3)
    return float(d.min()), onesided, sig, nk[0]


rows = {"fail-onset": [], "fail-pre": [], "succ-match": []}
for f in sorted(glob.glob("logs/rd_hg/ep_*.npz")):
    z = np.load(f)
    obs, asm = z["obs"], int(z["asm"])
    T = len(obs)
    if not asm:
        dser = []
        for i in range(1, T):
            q = torch.tensor(no.normalize(np.stack(
                [obs[i - 1], obs[i]])).reshape(-1))
            dser.append(float(torch.norm(tP - q, dim=1).min()))
        dser = np.asarray(dser)
        esc = np.where(dser > 2.0)[0]
        onset = int(esc[0]) if len(esc) else int(dser.argmax())
        for tag, lo, hi in [("fail-onset", max(1, onset - 8), onset + 1),
                            ("fail-pre", max(1, onset - 60),
                             max(1, onset - 30))]:
            for i in range(lo, hi, 2):
                rows[tag].append(stats_at(np.stack([obs[i - 1], obs[i]])))
    else:
        for i in range(int(0.88 * T), min(int(0.95 * T), T - 1), 4):
            rows["succ-match"].append(stats_at(np.stack([obs[i - 1],
                                                         obs[i]])))
for tag, R in rows.items():
    A = np.array([r[:3] for r in R])
    print(f"HE {tag} n={len(A)} tube-d p50 {np.median(A[:, 0]):.2f} p90 "
          f"{np.percentile(A[:, 0], 90):.2f} | onesided p50 "
          f"{np.median(A[:, 1]):.2f} | sigma p50 {np.median(A[:, 2]):.4f} "
          f"p90 {np.percentile(A[:, 2], 90):.4f}", flush=True)
print("HE done", flush=True)
