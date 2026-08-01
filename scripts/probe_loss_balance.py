"""Is the MP training signal scale/loss-imbalanced, so that rebalancing
small-error components is the active ingredient of the winning objectives?

Measurements on MP-200 TRAINING windows (same normalizers as training):
  1. LOSS-MASS DISTRIBUTION of the converged L2 policy: share of total MSE
     by trajectory-progress decile and by action dim. If the precision-
     critical late phase (insertion/hang) holds a tiny share, plain MSE
     allocates almost no gradient there.
  2. TARGET ACTIVITY per decile: mean squared normalized target action —
     the scale of the signal itself (imbalance of the DATA, not the fit).
  3. HT'S LEARNED REBALANCING: sigma(x) from the scalar head at the same
     states; implied per-state weight w = 1/sigma^2 (normalized to mean 1)
     per decile. This is what the adaptive objective CHOSE to upweight.
  4. Same residual profile for MIP-200 (reference).
Prints LB lines. Env: LB_N (windows), checkpoints hardcoded to MP-200 arms.
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

from mip.agent import TrainingAgent
from mip.datasets.robomimic_dataset import make_dataset
from mip.samplers import get_sampler

D2 = os.environ.get("LB_DATA", "data/tool_hang_full2ins_mp_200.hdf5")
ARMS = [
    ("L2-200", "regression", "logs/mp200_l2_s1000/models/snap_300000.pt"),
    ("HT-200", "regression_hetero_t", "logs/mp200_ht_s1000/models/snap_300000.pt"),
    # sigma collapses to its floor at convergence; read the learned
    # rebalancing at training-time snapshots as well
    ("HT-20k", "regression_hetero_t", "logs/mp200_ht_s1000/models/snap_20000.pt"),
    ("HT-60k", "regression_hetero_t", "logs/mp200_ht_s1000/models/snap_60000.pt"),
    ("MIP-200", "mip", "logs/mp200_mip_s1000/models/snap_300000.pt"),
]
if os.environ.get("LB_ARMS"):
    ARMS = [tuple(a.split(":")) for a in os.environ["LB_ARMS"].split(",")]
NW = int(os.environ.get("LB_N", "6000"))
OK = ["object", "robot0_eef_pos", "robot0_eef_quat", "robot0_gripper_qpos"]

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

    h = h5py.File(D2, "r")
    names = sorted(h["data"].keys(), key=lambda d: int(d.split("_")[1]))
    rng = np.random.RandomState(0)
    W, T, P = [], [], []
    per_traj = max(1, NW // len(names))
    for dn in names:
        o = h[f"data/{dn}/obs"]
        S = np.concatenate([np.asarray(o[k]) for k in OK], 1).astype(np.float32)
        A = np.asarray(h[f"data/{dn}/actions"], dtype=np.float32)
        L = len(S)
        if L < H + 4:
            continue
        idx = rng.choice(np.arange(1, L - H - 1), min(per_traj, L - H - 2),
                         replace=False)
        for i in idx:
            W.append(np.stack([S[i - 1], S[i]]))
            # chunk index 0 aligns with the FIRST obs frame (i-1): the eval
            # harness executes an[:, obs_steps-1 : obs_steps-1+AS]
            T.append(A[i - 1:i - 1 + H])
            P.append(i / L)
    h.close()
    W, P = np.stack(W), np.asarray(P)
    # normalized target chunks: raw 7-dim -> rot6d 10-dim -> normalize
    def fwd(a):  # (H, 7) pos3+axisangle3+grip1
        rot = ds.rotation_transformer.forward(a[..., 3:6])
        return np.concatenate([a[..., :3], rot, a[..., [6]]], -1)

    Tn = np.stack([na.normalize(fwd(t)) for t in T])
    print(f"LB {name} windows {len(W)}", flush=True)

    R2, SIG = [], []
    for i in range(0, len(W), 256):
        xb = torch.tensor(np.stack([no.normalize(w) for w in W[i:i + 256]]),
                          device=dev, dtype=torch.float32)
        tb = torch.tensor(Tn[i:i + 256], device=dev, dtype=torch.float32)
        with torch.no_grad():
            a0 = torch.zeros((len(xb), H, 10), device=dev)
            an = sampler(cfg.optimization, ag.flow_map_ema,
                         ag.encoder_ema, a0, {"state": xb})
            R2.append(((an - tb) ** 2).cpu().numpy())
            if name.startswith("HT"):
                t0 = torch.zeros(len(xb), device=dev)
                emb = ag.encoder_ema({"state": xb}, None)
                _, s_raw = ag.flow_map_ema.net(a0, t0, t0, emb)
                sig = (torch.nn.functional.softplus(s_raw)
                       .reshape(len(xb), -1).mean(1) + 1e-3)
                SIG.append(sig.cpu().numpy())
    R2 = np.concatenate(R2)                     # (N, H, 10)
    per_state = R2.mean(axis=(1, 2))
    dec = np.clip((P * 10).astype(int), 0, 9)

    # 1. loss-mass share + mean per-state MSE by decile
    tot = per_state.sum()
    shares = [per_state[dec == d].sum() / tot for d in range(10)]
    means = [per_state[dec == d].mean() for d in range(10)]
    print("LB {} loss_share_decile {}".format(
        name, " ".join(f"{s:.3f}" for s in shares)), flush=True)
    print("LB {} mse_mean_decile {}".format(
        name, " ".join(f"{m:.2e}" for m in means)), flush=True)
    # per-dim share (pos 0:3, rot6d 3:9, grip 9)
    dsh = R2.mean(axis=1).sum(0) / R2.mean(axis=1).sum()
    print("LB {} dim_share pos {:.3f} rot {:.3f} grip {:.3f}".format(
        name, dsh[0:3].sum(), dsh[3:9].sum(), dsh[9]), flush=True)

    # 2. target activity by decile (signal scale, arm-independent; print once)
    if name == "L2-200":
        act2 = (Tn ** 2).mean(axis=(1, 2))
        amean = [act2[dec == d].mean() for d in range(10)]
        print("LB TARGET act2_mean_decile {}".format(
            " ".join(f"{m:.3f}" for m in amean)), flush=True)

    # 3. HT learned weights by decile
    if name.startswith("HT") and SIG:
        SIG = np.concatenate(SIG)
        wgt = 1.0 / SIG ** 2
        wgt = wgt / wgt.mean()
        wm = [wgt[dec == d].mean() for d in range(10)]
        sm = [SIG[dec == d].mean() for d in range(10)]
        print(f"LB {name} sigma_mean_decile " + "{}".format(
            " ".join(f"{m:.4f}" for m in sm)), flush=True)
        print(f"LB {name} weight_mean_decile " + "{}".format(
            " ".join(f"{m:.2f}" for m in wm)), flush=True)
        wmed = [np.median(wgt[dec == d]) for d in range(10)]
        print(f"LB {name} weight_med_decile " + "{}".format(
            " ".join(f"{m:.2f}" for m in wmed)), flush=True)
print("LB done", flush=True)
