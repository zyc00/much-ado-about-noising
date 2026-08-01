"""Interpolation-path test of the branch-commitment / composed-Lipschitz
account. For conflict pairs (cross-episode states A,B with large output
disagreement), interpolate the obs window x(lam) linearly and trace each
arm's output:
  smooth averager      : gradual morph, midpoint output BETWEEN branches
  branch-committed map : plateau -> thin jump -> plateau

Per pair: sharp = max_lam ||f(x_{l+dl})-f(x_l)|| / (dl * ||f(B)-f(A)||)
(linear morph -> 1; step -> 1/dl), and mid-branchness
bm = min(||f(x_.5)-f(A)||, ||f(x_.5)-f(B)||) / ||f(B)-f(A)|| (averaging ->
~0.5; committed -> ~0). Prints BSHARP lines. Env: BR_N pairs.
"""
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
from scipy.spatial import cKDTree

from mip.agent import TrainingAgent
from mip.datasets.robomimic_dataset import make_dataset
from mip.samplers import get_sampler, mip_step1_only_sampler

D2 = "data/tool_hang_full2ins_mp_200.hdf5"
OK = ["object", "robot0_eef_pos", "robot0_eef_quat", "robot0_gripper_qpos"]
ARMS = [
    ("L2", "regression", "logs/mp200_l2_s1000/models/snap_300000.pt", False),
    ("HT", "regression_hetero_t", "logs/mp200_ht_s1000/models/snap_300000.pt", False),
    ("HG", "regression_hetero_gauss", "logs/mp200_hg_s1000/models/snap_300000.pt", False),
    ("NONOISE", "mip_nonoise", "logs/mp200_nonoise_s1000/models/snap_300000.pt", False),
    ("MIP-s1", "mip", "logs/mp200_mip_s1000/models/snap_300000.pt", True),
    ("MIP", "mip", "logs/mp200_mip_s1000/models/snap_300000.pt", False),
]
NP_ = int(os.environ.get("BR_N", "300"))
LAMS = np.linspace(0, 1, 21)          # dl = 0.05 -> step map sharp ~ 20

# ---- conflict pairs: near-ish cross-episode training states with
#      different executed actions --------------------------------------
h = h5py.File(D2, "r")
names = sorted(h["data"].keys(), key=lambda d: int(d.split("_")[1]))
PX, PA, PT = [], [], []
for ti, dn in enumerate(names):
    o = h[f"data/{dn}/obs"]
    S = np.concatenate([np.asarray(o[k]) for k in OK], 1).astype(np.float32)
    A = np.asarray(h[f"data/{dn}/actions"], dtype=np.float32)
    L = len(S)
    if L < 12:
        continue
    for i in range(1, L - 9, 2):
        PX.append(np.stack([S[i - 1], S[i]]))
        PA.append(A[i:i + 8])
        PT.append(ti)
h.close()
PX, PA, PT = np.stack(PX), np.stack(PA), np.asarray(PT)
F = PX.reshape(len(PX), -1)
mu, sd = F.mean(0), F.std(0) + 1e-6
Fn = (F - mu) / sd
tree = cKDTree(Fn)
rng = np.random.RandomState(0)
cand = rng.choice(len(F), 6000, replace=False)
pairs = []
for q in cand:
    d, nb = tree.query(Fn[q], k=24)
    m = (PT[nb] != PT[q]) & (d < 1.5) & (d > 0.05)
    nb2, d2 = nb[m], d[m]
    if len(nb2) == 0:
        continue
    # action disagreement of the pair (raw pos dims)
    dis = np.abs(PA[nb2][:, :, 0:3] - PA[q][None, :, 0:3]).mean((1, 2))
    j = int(np.argmax(dis))
    if dis[j] > 0.02:                  # conflicting branch pair
        pairs.append((q, int(nb2[j])))
    if len(pairs) >= NP_:
        break
pairs = np.asarray(pairs)
print(f"BSHARP pairs {len(pairs)}", flush=True)
A_ = PX[pairs[:, 0]]
B_ = PX[pairs[:, 1]]

for name, loss, ck, s1 in ARMS:
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
    H = int(cfg.task.horizon)
    ds = make_dataset(cfg.task)
    ag = TrainingAgent(cfg)
    ag.load(ck, load_optimizer=False)
    ag.eval()
    sampler = mip_step1_only_sampler if s1 else get_sampler(loss)
    no = ds.normalizer["obs"]["state"]
    dev = cfg.optimization.device

    def act(WS):
        out = []
        for i in range(0, len(WS), 256):
            xb = torch.tensor(np.stack([no.normalize(w) for w in WS[i:i + 256]]),
                              device=dev, dtype=torch.float32)
            with torch.no_grad():
                a0 = torch.zeros((len(xb), H, 10), device=dev)
                an = sampler(cfg.optimization, ag.flow_map_ema,
                             ag.encoder_ema, a0, {"state": xb})
            out.append(an.reshape(len(xb), -1).cpu().numpy())
        return np.concatenate(out)

    outs = []                                  # per lam: (Npairs, D)
    for lam in LAMS:
        outs.append(act((1 - lam) * A_ + lam * B_))
    outs = np.stack(outs)                      # (L, N, D)
    fa, fb = outs[0], outs[-1]
    span = np.linalg.norm(fb - fa, axis=1) + 1e-9
    step = np.linalg.norm(np.diff(outs, axis=0), axis=2)   # (L-1, N)
    sharp = step.max(0) / (span * (LAMS[1] - LAMS[0]))
    mid = outs[len(LAMS) // 2]
    bm = np.minimum(np.linalg.norm(mid - fa, axis=1),
                    np.linalg.norm(mid - fb, axis=1)) / span
    # path excess: total variation / span (1 = monotone morph; >1 = detour)
    tv = step.sum(0) / span
    print(f"BSHARP {name} sharp_p50 {np.median(sharp):.2f} p90 "
          f"{np.percentile(sharp,90):.2f} | midbranch_p50 {np.median(bm):.3f} "
          f"frac<0.25 {(bm < 0.25).mean():.2f} | tv_p50 {np.median(tv):.2f}",
          flush=True)
print("BSHARP done", flush=True)
