"""Per-region LOSS and GRADIENT-NORM balance over snapshot grids.

Regions: settle-core windows (entire 16-step chunk quiet: max action norm
< 2.5x the episode minimum level) vs stroke windows (argmax norm after the
minimum). For each arm and snapshot, computes the arm's ACTUAL training loss
(mip.losses fn, live non-EMA nets) restricted to each region and the full
parameter gradient norm of that restricted loss.
Prints: GRADBAL <arm> <step> <loss_settle> <loss_stroke> <g_settle> <g_stroke>
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
from mip.losses import get_loss_fn


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

no = ds0.normalizer["obs"]["state"]
na = ds0.normalizer["action"]
dev = cfg0.optimization.device

sets, strs = [], []
start = 0
for e in range(min(300, len(ends))):
    end = int(ends[e])
    S, A = S_all[start:end], A_all[start:end]
    T = len(S)
    start = end
    if T < 60:
        continue
    an = np.linalg.norm(A, axis=1)
    lo, hi = T // 3, 2 * T // 3
    settle = lo + int(np.argmin(an[lo:hi]))
    # settle-core: whole chunk quiet (no onset tail inside the horizon)
    if settle >= 1 and settle + H < T:
        seg = an[settle:settle + H]
        if seg.max() < 2.5 * max(an[settle], 1e-6) and len(sets) < 128:
            sets.append((np.stack([S[settle - 1], S[settle]]), A[settle:settle + H]))
    seg2 = an[settle:min(settle + 40, T - H - 1)]
    if len(seg2) >= 6:
        stroke = settle + int(np.argmax(seg2))
        if stroke - settle >= 5 and stroke + H < T and len(strs) < 128:
            strs.append((np.stack([S[stroke - 1], S[stroke]]), A[stroke:stroke + H]))
print(f"windows: settle-core={len(sets)} stroke={len(strs)}", flush=True)


def make_batch(wins):
    W = np.stack([w for w, _ in wins])          # (B,2,53)
    A = np.stack([a for _, a in wins])          # (B,H,AD)
    x = torch.tensor(np.stack([no.normalize(w) for w in W]), device=dev,
                     dtype=torch.float32)
    an_t = torch.tensor(np.stack([na.normalize(a) for a in A]), device=dev,
                        dtype=torch.float32)
    return {"state": x}, an_t


B_set = make_batch(sets)
B_str = make_batch(strs)


def region_stats(cfg, ag, loss_fn, batch):
    obs, act = batch
    dt = torch.zeros(len(act), device=dev)
    for p in list(ag.flow_map.parameters()) + list(ag.encoder.parameters()):
        p.grad = None
    loss, _ = loss_fn(cfg.optimization, ag.flow_map, ag.encoder,
                      ag.interpolant, act, obs, dt)
    loss.backward()
    g2 = 0.0
    for p in list(ag.flow_map.parameters()) + list(ag.encoder.parameters()):
        if p.grad is not None:
            g2 += float((p.grad ** 2).sum())
    return float(loss), g2 ** 0.5


ARMS = [("L2", "regression", "logs/f2i_l2_s1000"),
        ("HG", "regression_hetero_gauss", "logs/f2i_hg_s1000")]
for name, loss, d in ARMS:
    cfg, ds, ag = load(loss)
    fn = get_loss_fn(loss)
    for ck in sorted(glob.glob(f"{d}/models/snap_*.pt"),
                     key=lambda p: int(p.split("snap_")[1].split(".")[0])):
        step = int(ck.split("snap_")[1].split(".")[0])
        ag.load(ck, load_optimizer=False)
        l_set, g_set = region_stats(cfg, ag, fn, B_set)
        l_str, g_str = region_stats(cfg, ag, fn, B_str)
        print(f"GRADBAL {name} {step} {l_set:.3e} {l_str:.3e} "
              f"{g_set:.3e} {g_str:.3e}", flush=True)
print("GRADBAL done", flush=True)
