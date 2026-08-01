"""SFIT2: scripted fit anatomy stratified by PER-STATE difficulty, not
progress deciles. For each probe state: conflict = RMS spread of executed-chunk
pos targets among cross-demo neighbors within EPS (normalized obs space);
density = distance to nearest cross-demo neighbor. Residual reported per
conflict quartile and for sparse (no-neighbor) states, p50 AND p90.
Prints SFIT2 lines."""
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
NDEMO = int(os.environ.get("SF_NDEMO", "8"))
EPS = float(os.environ.get("SF_EPS", "0.5"))
AS_LO, AS_HI = 1, 9

with initialize_config_dir(version_base=None,
                           config_dir=os.path.abspath("examples/configs")):
    cfg0 = compose(config_name="main", overrides=[
        "task=tool_hang_ph_state_delta_legacy",
        "+task.dataset_path=" + os.path.abspath(D2), "network=chiunet",
        "optimization.auto_resume=false", "log.wandb_mode=disabled"])
OmegaConf.set_struct(cfg0, False)
cfg0.task.obs_dim = 53
cfg0.task.horizon = int(2 ** np.ceil(np.log2(cfg0.task.horizon)))
ds0 = make_dataset(cfg0.task)
no = ds0.normalizer["obs"]["state"]

h = h5py.File(D2, "r")
names = sorted(h["data"].keys(), key=lambda d: int(d.split("_")[1]))
PW, PA, PD = [], [], []
for di, dn in enumerate(names):
    o = h[f"data/{dn}/obs"]
    S = np.concatenate([np.asarray(o[k]) for k in OK], 1).astype(np.float32)
    A = np.asarray(h[f"data/{dn}/actions"], dtype=np.float32)
    for i in range(1, len(S) - 10):
        PW.append(no.normalize(np.stack([S[i - 1], S[i]])).reshape(-1))
        PA.append(A[i - 1 + AS_LO:i - 1 + AS_HI, 0:3])
        PD.append(di)
PW, PA, PD = np.stack(PW), np.stack(PA), np.asarray(PD)
print(f"SFIT2 pool {len(PW)} eps {EPS}", flush=True)

demos = []
for dn in names[:NDEMO]:
    o = h[f"data/{dn}/obs"]
    S = np.concatenate([np.asarray(o[k]) for k in OK], 1).astype(np.float32)
    A = np.asarray(h[f"data/{dn}/actions"], dtype=np.float32)
    demos.append((dn, S, A))
h.close()

Wq, GAq, CONF, RNN, DISP = [], [], [], [], []
tP = torch.tensor(PW)
for di, (dn, S, A) in enumerate(demos):
    hi = len(S) - int(cfg0.task.horizon)
    for i in range(1, hi):
        w = np.stack([S[i - 1], S[i]])
        Wq.append(w)
        g = A[i - 1 + AS_LO:i - 1 + AS_HI, 0:3]
        GAq.append(g)
        q = torch.tensor(no.normalize(w).reshape(-1))
        d = torch.norm(tP - q, dim=1).numpy()
        d[PD == di] = 1e9
        RNN.append(d.min())
        nb = np.where(d < EPS)[0]
        if len(nb) >= 3:
            T = PA[nb]
            mu = T.mean(0)
            CONF.append(float(np.sqrt(((T - mu) ** 2).mean())))
            DISP.append(float(np.sqrt(((g - mu) ** 2).mean())))
        else:
            CONF.append(np.nan)
            DISP.append(np.nan)
Wq, GAq = np.stack(Wq), np.stack(GAq)
CONF, RNN, DISP = map(np.asarray, (CONF, RNN, DISP))
ok = ~np.isnan(CONF)
qs = np.nanpercentile(CONF, [25, 50, 75])
print(f"SFIT2 queries {len(Wq)} with-nb {ok.sum()} conf quartiles "
      f"{qs[0]:.5f} {qs[1]:.5f} {qs[2]:.5f} | disp p50 "
      f"{np.nanmedian(DISP):.5f} rnn p50 {np.median(RNN):.3f}", flush=True)

STRATA = {"c1": ok & (CONF < qs[0]), "c2": ok & (CONF >= qs[0]) & (CONF < qs[1]),
          "c3": ok & (CONF >= qs[1]) & (CONF < qs[2]),
          "c4": ok & (CONF >= qs[2]), "sparse": ~ok}
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
    na = ds.normalizer["action"]
    dev = cfg.optimization.device
    R = []
    for i in range(0, len(Wq), 256):
        xb = torch.tensor(np.stack([no.normalize(w) for w in Wq[i:i + 256]]),
                          device=dev, dtype=torch.float32)
        with torch.no_grad():
            a0 = torch.zeros((len(xb), Hn, 10), device=dev)
            an = sampler(cfg.optimization, ag.flow_map_ema, ag.encoder_ema,
                         a0, {"state": xb})
        pe = np.asarray(ds.undo_transform_action(
            na.unnormalize(an.cpu().numpy())[:, AS_LO:AS_HI]))
        R.append(np.abs(pe[:, :, 0:3] - GAq[i:i + 256]).mean(axis=(1, 2)))
    R = np.concatenate(R)
    row = " ".join(f"{k} {np.median(R[m]):.5f}/{np.percentile(R[m], 90):.5f}"
                   for k, m in STRATA.items() if m.sum() > 5)
    print(f"SFIT2 {arm} p50/p90: {row}", flush=True)
    top = ok & (CONF >= np.nanpercentile(CONF, 95))
    print(f"SFIT2 {arm} top5%conflict p50 {np.median(R[top]):.5f} p90 "
          f"{np.percentile(R[top], 90):.5f} n={top.sum()}", flush=True)
print("SFIT2 done", flush=True)
