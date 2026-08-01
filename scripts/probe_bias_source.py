"""Is the emitted DC bias NEIGHBOR AVERAGING? For held-out states, compare
each arm's per-state bias vector b_arm = mean_t(pred_t - gt_t) with the
model-free kNN-regression bias b_knn = mean_t(a_knn_t - gt_t), where a_knn
is the inverse-distance-weighted average of the k=8 nearest TRAINING
windows' action chunks (MP-200 pool, z-scored obs metric).

Pre-registered: cos(b_L2, b_knn) > cos(b_MIP, b_knn), both positive; the
alignment is strongest where |b_knn| is large (pockets/align). Prints BSRC
lines.
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
from mip.samplers import get_sampler

D2 = "data/tool_hang_full2ins_mp_200.hdf5"
HELD = "data/tool_hang_full2ins_mp_20k.hdf5"
OK = ["object", "robot0_eef_pos", "robot0_eef_quat", "robot0_gripper_qpos"]
ARMS = [
    ("L2", "regression", "logs/mp200_l2_s1000/models/snap_300000.pt"),
    ("NONOISE", "mip_nonoise", "logs/mp200_nonoise_s1000/models/snap_300000.pt"),
    ("MIP", "mip", "logs/mp200_mip_s1000/models/snap_300000.pt"),
]
NS = int(os.environ.get("BS_N", "4000"))
AS_LO, AS_HI = 1, 9
K = 8

# ---- training pool (windows + raw executed chunks) -----------------------
h = h5py.File(D2, "r")
names = sorted(h["data"].keys(), key=lambda d: int(d.split("_")[1]))
PX, PA = [], []
for dn in names:
    o = h[f"data/{dn}/obs"]
    S = np.concatenate([np.asarray(o[k]) for k in OK], 1).astype(np.float32)
    A = np.asarray(h[f"data/{dn}/actions"], dtype=np.float32)
    L = len(S)
    if L < 12:
        continue
    for i in range(1, L - 9):
        PX.append(np.stack([S[i - 1], S[i]]).reshape(-1))
        PA.append(A[i - 1 + AS_LO:i - 1 + AS_HI])
h.close()
PX, PA = np.stack(PX), np.stack(PA)
xmu, xsd = PX.mean(0), PX.std(0) + 1e-6
tree = cKDTree((PX - xmu) / xsd)
print(f"BSRC pool {len(PX)}", flush=True)

# ---- held-out states (same protocol as probe_dc_bias) --------------------
h = h5py.File(HELD, "r")
hnames = sorted(h["data"].keys(), key=lambda d: int(d.split("_")[1]))[2000:2600]
rng = np.random.RandomState(0)
W, GA, PH = [], [], []
per = max(1, NS // len(hnames))
for dn in hnames:
    o = h[f"data/{dn}/obs"]
    S = np.concatenate([np.asarray(o[k]) for k in OK], 1).astype(np.float32)
    A = np.asarray(h[f"data/{dn}/actions"], dtype=np.float32)
    L = len(S)
    if L < 20:
        continue
    idx = rng.choice(np.arange(1, L - 10), min(per, L - 11), replace=False)
    for i in idx:
        W.append(np.stack([S[i - 1], S[i]]))
        GA.append(A[i - 1 + AS_LO:i - 1 + AS_HI])
        PH.append(i / L)
h.close()
W, GA, PH = np.stack(W), np.stack(GA), np.asarray(PH)
print(f"BSRC held-out {len(W)}", flush=True)

# ---- model-free kNN bias -------------------------------------------------
Wn = (W.reshape(len(W), -1) - xmu) / xsd
d, nb = tree.query(Wn, k=K)
wgt = 1.0 / (d + 1e-6)
wgt = wgt / wgt.sum(1, keepdims=True)
a_knn = (PA[nb] * wgt[:, :, None, None]).sum(1)          # (N, 8, 7)
b_knn = (a_knn[:, :, 0:3] - GA[:, :, 0:3]).mean(1)       # (N, 3)
bkn = np.linalg.norm(b_knn, axis=1)
print(f"BSRC knn |b|_p50 {np.median(bkn):.5f} p90 "
      f"{np.percentile(bkn,90):.5f}", flush=True)

for name, loss, ck in ARMS:
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
    sampler = get_sampler(loss)
    no, na = ds.normalizer["obs"]["state"], ds.normalizer["action"]
    dev = cfg.optimization.device

    B = []
    for i in range(0, len(W), 256):
        xb = torch.tensor(np.stack([no.normalize(w) for w in W[i:i + 256]]),
                          device=dev, dtype=torch.float32)
        with torch.no_grad():
            a0 = torch.zeros((len(xb), H, 10), device=dev)
            an = sampler(cfg.optimization, ag.flow_map_ema,
                         ag.encoder_ema, a0, {"state": xb})
        pe = np.asarray(ds.undo_transform_action(
            na.unnormalize(an.cpu().numpy())[:, AS_LO:AS_HI]))
        B.append((pe[:, :, 0:3] - GA[i:i + 256, :, 0:3]).mean(1))
    B = np.stack([r for b in B for r in b])
    bn = np.linalg.norm(B, axis=1)
    cos = (B * b_knn).sum(1) / (bn * bkn + 1e-12)
    m = (bn > 1e-4) & (bkn > 1e-4)
    big = m & (bkn > np.percentile(bkn, 66))
    print(f"BSRC {name} cos(b_arm, b_knn) p50 {np.median(cos[m]):+.3f} "
          f"(n={m.sum()}) | big-knn-bias stratum p50 "
          f"{np.median(cos[big]):+.3f} | frac_cos>0.5 "
          f"{(cos[m] > 0.5).mean():.2f} | |b|/|b_knn| p50 "
          f"{np.median(bn[m]/bkn[m]):.2f}", flush=True)
print("BSRC done", flush=True)
