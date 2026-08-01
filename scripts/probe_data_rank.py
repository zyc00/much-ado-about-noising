"""Data-side predictors of the learned Jacobian rank, phase by phase.

For each decile of normalized episode time (the same binning as
probe_jac_phases) we measure, from the DATA ALONE (k=32 neighbors in
normalized window space):
  dens      : local density  (1 / mean kNN radius)  -> the rho(x) factor
  PR_dX     : participation ratio of the local INPUT covariance  -> how many
              input directions the data varies in locally (lambda_Sigma_rho)
  PR_dY     : participation ratio of the local LABEL covariance  -> how many
              action directions vary locally
  mag_dY    : mean ||dY|| among neighbors  -> action diversity magnitude
  PR_W      : participation ratio of the local linear map's singular values
              (dY ~ dX W, ridge) -> the Jacobian rank the task DEMANDS here
Theory (jacobian_supervision_note Eq.3): MSE's retained rank should track
dens * PR_dX * (label variation); a flow/repriced objective should track it
LESS (its stiffness is objective-supplied).
Prints DRANK lines; pair with JACPHASE per-decile network PR.
Env: DR_DATASET, DR_K, DR_NPD.
"""
import os
import sys

os.environ["MUJOCO_GL"] = "egl"
sys.path.insert(0, "scripts")
sys.path.insert(0, ".")
import numpy as np
from hydra import compose, initialize_config_dir
from omegaconf import OmegaConf
from scipy.spatial import cKDTree

from mip.datasets.robomimic_dataset import make_dataset

DSET = os.environ["DR_DATASET"]
K = int(os.environ.get("DR_K", "32"))
NPD = int(os.environ.get("DR_NPD", "40"))

with initialize_config_dir(version_base=None,
                           config_dir=os.path.abspath("examples/configs")):
    cfg = compose(config_name="main", overrides=[
        "task=tool_hang_ph_state_delta_legacy",
        "+task.dataset_path=" + os.path.abspath(DSET),
        "network=chiunet", "optimization.loss_type=regression",
        "optimization.auto_resume=false", "log.wandb_mode=disabled"])
OmegaConf.set_struct(cfg, False)
cfg.task.obs_dim = 53
cfg.task.horizon = int(2 ** np.ceil(np.log2(cfg.task.horizon)))
ds = make_dataset(cfg.task)
rb = ds.replay_buffer
ends = rb.episode_ends[:]
obs_entry = rb["obs"]
try:
    S_all = obs_entry["state"][:]
except (TypeError, IndexError, KeyError):
    S_all = obs_entry[:]
A_all = rb["action"][:]
H = int(cfg.task.horizon)
no = ds.normalizer["obs"]["state"]
na = ds.normalizer["action"]

starts = np.concatenate([[0], ends[:-1]])
W_list, Y_list, FRAC = [], [], []
for e in range(len(ends)):
    s0, e0 = int(starts[e]), int(ends[e])
    T = e0 - s0
    if T < H + 3:
        continue
    S, A = S_all[s0:e0], A_all[s0:e0]
    for i in range(1, T - H):
        W_list.append(no.normalize(np.stack([S[i - 1], S[i]])).reshape(-1))
        Y_list.append(na.normalize(A[i:i + H]).reshape(-1))
        FRAC.append(i / (T - H))
WN = np.stack(W_list).astype(np.float64)
YN = np.stack(Y_list).astype(np.float64)
FRAC = np.array(FRAC)
tree = cKDTree(WN)
print(f"windows {len(WN)} dataset {DSET}", flush=True)


def pr_of(M):
    """participation ratio of the spectrum of M^T M (M rows = samples)"""
    s = np.linalg.svd(M, compute_uv=False)
    s2 = s ** 2
    return float(s2.sum() ** 2 / ((s2 ** 2).sum() + 1e-30))


rng = np.random.RandomState(0)
for d in range(10):
    m = (FRAC >= d / 10) & (FRAC < (d + 1) / 10)
    idx = np.where(m)[0]
    if len(idx) < NPD:
        continue
    sel = rng.choice(idx, NPD, replace=False)
    dens, prx, pry, mag, prw = [], [], [], [], []
    for i in sel:
        dist, nb = tree.query(WN[i], k=K + 1)
        nb, dist = nb[1:], dist[1:]
        dX = WN[nb] - WN[i]
        dY = YN[nb] - YN[i]
        dens.append(1.0 / (dist.mean() + 1e-12))
        prx.append(pr_of(dX))
        pry.append(pr_of(dY))
        mag.append(float(np.linalg.norm(dY, axis=1).mean()))
        G = dX.T @ dX + 1e-6 * np.eye(dX.shape[1])
        Wl = np.linalg.solve(G, dX.T @ dY)
        prw.append(pr_of(Wl.T))
    print(f"DRANK {d} n={NPD} dens {np.mean(dens):.3f} PR_dX {np.mean(prx):.2f} "
          f"PR_dY {np.mean(pry):.2f} mag_dY {np.mean(mag):.4f} "
          f"PR_W {np.mean(prw):.2f}", flush=True)
print("DRANK done", flush=True)
