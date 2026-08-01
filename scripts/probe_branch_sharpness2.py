"""Branch-commitment, span-stratified (the user's correction to PART
CCCLXXXVI): commitment should appear only for pairs whose branch actions
REALLY differ, and midpoint statistics miss off-center jumps.

Per pair, over the 21-point interpolation path:
  dis     : GT executed-action disagreement of the pair (selection score)
  sharp   : max local rate / linear rate (as before)
  jump3   : fraction of total path variation in the best 3 consecutive
            steps (linear morph -> 3/20 = 0.15; single step -> ~1.0)
  branch  : mean over lam of min(d(f,fA), d(f,fB))/span
            (linear morph -> 0.25; plateau-jump-plateau -> ~0.05-0.1)
  lam*    : location of the max step (transition center)
Stratified by dis terciles. Prints BS2 lines per arm x tercile + lam*
deciles. Env: BR_N.
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
NP_ = int(os.environ.get("BR_N", "360"))
LAMS = np.linspace(0, 1, 21)

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
cand = rng.choice(len(F), 9000, replace=False)
pairs, diss = [], []
for q in cand:
    d, nb = tree.query(Fn[q], k=24)
    m = (PT[nb] != PT[q]) & (d < 1.5) & (d > 0.05)
    nb2 = nb[m]
    if len(nb2) == 0:
        continue
    dis = np.abs(PA[nb2][:, :, 0:3] - PA[q][None, :, 0:3]).mean((1, 2))
    j = int(np.argmax(dis))
    if dis[j] > 0.005:                 # accept a WIDE disagreement range
        pairs.append((q, int(nb2[j])))
        diss.append(dis[j])
    if len(pairs) >= NP_:
        break
pairs, diss = np.asarray(pairs), np.asarray(diss)
ter = np.digitize(diss, np.quantile(diss, [1 / 3, 2 / 3]))
print(f"BS2 pairs {len(pairs)} dis terciles "
      f"{np.quantile(diss, [0, 1/3, 2/3, 1]).round(3)}", flush=True)
A_, B_ = PX[pairs[:, 0]], PX[pairs[:, 1]]

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

    outs = np.stack([act((1 - l) * A_ + l * B_) for l in LAMS])
    fa, fb = outs[0], outs[-1]
    span = np.linalg.norm(fb - fa, axis=1) + 1e-9
    step = np.linalg.norm(np.diff(outs, axis=0), axis=2)      # (20, N)
    tv = step.sum(0) + 1e-12
    sharp = step.max(0) / (span * (LAMS[1] - LAMS[0]))
    # best-3-consecutive-step share of total variation
    k3 = step[0:-2] + step[1:-1] + step[2:]
    jump3 = k3.max(0) / tv
    lam_star = LAMS[1:][np.argmax(step, axis=0)]
    # mean over path of distance to the NEARER endpoint output
    dA = np.linalg.norm(outs - fa[None], axis=2)
    dB = np.linalg.norm(outs - fb[None], axis=2)
    branch = np.minimum(dA, dB).mean(0) / span
    for t, lab in ((0, "loT"), (1, "midT"), (2, "hiT")):
        m = ter == t
        print(f"BS2 {name} {lab} n={m.sum()} sharp_p50 "
              f"{np.median(sharp[m]):.2f} jump3_p50 {np.median(jump3[m]):.2f} "
              f"branch_p50 {np.median(branch[m]):.3f} "
              f"(committed<0.15, linear~0.25)", flush=True)
    hi = ter == 2
    print(f"BS2 {name} lam* deciles(hiT) "
          f"{np.percentile(lam_star[hi], [10, 30, 50, 70, 90]).round(2)}",
          flush=True)
print("BS2 done", flush=True)
