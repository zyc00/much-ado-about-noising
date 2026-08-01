"""Per-region residual vs training step over snapshot grids (companion to the
boundary-formation ladder): for each arm and snapshot, mean per-sample MSE (in
normalized action units, = training-loss units for L2) on SETTLE-window samples
and STROKE-window samples mined exactly as in probe_interp_curve.py.
Prints: SETLOSS <arm> <step> <settle_mse> <stroke_mse>
"""
import glob
import os
import sys

os.environ["MUJOCO_GL"] = "egl"
sys.path.insert(0, "scripts")
sys.path.insert(0, ".")
import numpy as np
import torch
from hydra import compose, initialize_config_dir
from omegaconf import OmegaConf

from mip.agent import TrainingAgent
from mip.datasets.robomimic_dataset import make_dataset


def load(loss):
    with initialize_config_dir(version_base=None,
                               config_dir=os.path.abspath("examples/configs")):
        cfg = compose(config_name="main", overrides=[
            "task=tool_hang_ph_state_delta_legacy",
            "+task.dataset_path=" + os.path.abspath("data/tool_hang_full2ins_2000.hdf5"),
            "network=chiunet", f"optimization.loss_type={loss}",
            "optimization.auto_resume=false", "log.wandb_mode=disabled"])
    OmegaConf.set_struct(cfg, False)
    cfg.task.obs_dim = 53
    cfg.task.horizon = int(2 ** np.ceil(np.log2(cfg.task.horizon)))
    ds = make_dataset(cfg.task)
    return cfg, ds, TrainingAgent(cfg)


cfg0, ds0, _ = load("regression")
rb = ds0.replay_buffer
ends = rb.episode_ends[:]
obs_entry = rb["obs"]
try:
    S_all = obs_entry["state"][:]
except (TypeError, IndexError, KeyError):
    S_all = obs_entry[:]
A_all = rb["action"][:]
H = int(cfg0.task.horizon)
AD = A_all.shape[1]

no = ds0.normalizer["obs"]["state"]
na = ds0.normalizer["action"]
dev = cfg0.optimization.device

# mine settle/stroke windows exactly like probe_interp_curve
sets, strs = [], []
start = 0
for e in range(min(120, len(ends))):
    end = int(ends[e])
    S, A = S_all[start:end], A_all[start:end]
    T = len(S)
    start = end
    if T < 60:
        continue
    an = np.linalg.norm(A, axis=1)
    lo, hi = T // 3, 2 * T // 3
    settle = lo + int(np.argmin(an[lo:hi]))
    seg = an[settle:min(settle + 40, T - H - 1)]
    if len(seg) < 6:
        continue
    stroke = settle + int(np.argmax(seg))
    if stroke - settle >= 5 and settle >= 1 and stroke + H < T - end + end and stroke + H < T:
        sets.append((np.stack([S[settle - 1], S[settle]]), A[settle:settle + H]))
        strs.append((np.stack([S[stroke - 1], S[stroke]]), A[stroke:stroke + H]))
print(f"windows: settle={len(sets)} stroke={len(strs)}", flush=True)


def region_mse(ag, wins):
    errs = []
    for w, a in wins:
        x = torch.tensor(no.normalize(w[None]), device=dev,
                         dtype=torch.float32).reshape(1, 2, 53)
        an_t = torch.tensor(na.normalize(a), device=dev, dtype=torch.float32)
        emb = ag.encoder_ema({"state": x}, None)
        t = torch.zeros(1, device=dev)
        act0 = torch.zeros(1, H, AD, device=dev)
        with torch.no_grad():
            pred, _ = ag.flow_map_ema.net(act0, t, t, emb)
        errs.append(float(((pred[0, :len(an_t)] - an_t) ** 2).mean()))
    return float(np.mean(errs))


ARMS = [("L2", "regression", "logs/f2i_l2_s1000"),
        ("HG", "regression_hetero_gauss", "logs/f2i_hg_s1000"),
        ("HT", "regression_hetero_t", "logs/f2i_ht_s1000")]
for name, loss, d in ARMS:
    cfg, ds, ag = load(loss)
    for ck in sorted(glob.glob(f"{d}/models/snap_*.pt"),
                     key=lambda p: int(p.split("snap_")[1].split(".")[0])):
        step = int(ck.split("snap_")[1].split(".")[0])
        ag.load(ck, load_optimizer=False)
        ag.eval()
        m_set = region_mse(ag, sets)
        m_str = region_mse(ag, strs)
        print(f"SETLOSS {name} {step} {m_set:.3e} {m_str:.3e}", flush=True)
print("SETLOSS done", flush=True)
