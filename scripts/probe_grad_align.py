"""Alignment probe: is the settle-region gradient anti-aligned with the
actual (frequency-weighted) batch gradient during the retreat?
Per L2 snapshot: cos(g_settle, g_total) with g_total from 1024 uniformly
sampled windows (real batch composition), plus the settle batch's internal
coherence ||sum g_i|| / sum ||g_i|| over per-window gradients.
Prints: GRADALIGN <step> <cos> <coherence>
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
            "network=chiunet", "optimization.loss_type=regression",
            "optimization.auto_resume=false", "log.wandb_mode=disabled"])
    OmegaConf.set_struct(cfg, False)
    cfg.task.obs_dim = 53
    cfg.task.horizon = int(2 ** np.ceil(np.log2(cfg.task.horizon)))
    ds = make_dataset(cfg.task)
    return cfg, ds, TrainingAgent(cfg)


cfg, ds, ag = load("regression")
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
dev = cfg.optimization.device
fn = get_loss_fn("regression")

# settle-core windows (same rule as probe_grad_balance)
sets = []
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
    if settle >= 1 and settle + H < T:
        seg = an[settle:settle + H]
        if seg.max() < 2.5 * max(an[settle], 1e-6) and len(sets) < 128:
            sets.append((np.stack([S[settle - 1], S[settle]]), A[settle:settle + H]))

# uniform batch windows (real training composition)
rng = np.random.RandomState(0)
uni = []
starts = np.concatenate([[0], ends[:-1]])
while len(uni) < 1024:
    e = rng.randint(0, min(300, len(ends)))
    s0, e0 = int(starts[e]), int(ends[e])
    T = e0 - s0
    if T < H + 3:
        continue
    i = rng.randint(1, T - H)
    S, A = S_all[s0:e0], A_all[s0:e0]
    uni.append((np.stack([S[i - 1], S[i]]), A[i:i + H]))
print(f"windows: settle-core={len(sets)} uniform={len(uni)}", flush=True)


def make_batch(wins):
    W = np.stack([w for w, _ in wins])
    A = np.stack([a for _, a in wins])
    x = torch.tensor(np.stack([no.normalize(w) for w in W]), device=dev,
                     dtype=torch.float32)
    an_t = torch.tensor(np.stack([na.normalize(a) for a in A]), device=dev,
                        dtype=torch.float32)
    return {"state": x}, an_t


B_set = make_batch(sets)
B_uni = make_batch(uni)
params = None


def grad_vec(batch):
    global params
    obs, act = batch
    dt = torch.zeros(len(act), device=dev)
    params = list(ag.flow_map.parameters()) + list(ag.encoder.parameters())
    for p in params:
        p.grad = None
    loss, _ = fn(cfg.optimization, ag.flow_map, ag.encoder, ag.interpolant,
                 act, obs, dt)
    loss.backward()
    return torch.cat([(p.grad if p.grad is not None else torch.zeros_like(p)).reshape(-1)
                      for p in params])


for ck in sorted(glob.glob("logs/f2i_l2_s1000/models/snap_*.pt"),
                 key=lambda p: int(p.split("snap_")[1].split(".")[0])):
    step = int(ck.split("snap_")[1].split(".")[0])
    ag.load(ck, load_optimizer=False)
    g_set = grad_vec(B_set)
    g_tot = grad_vec(B_uni)
    cos = float(torch.dot(g_set, g_tot) /
                (g_set.norm() * g_tot.norm() + 1e-12))
    # settle-batch internal coherence over 8 sub-chunks of 16 windows
    subs = []
    for k in range(0, 128, 16):
        subs.append(grad_vec(make_batch(sets[k:k + 16])))
    subs = torch.stack(subs)
    coh = float(subs.sum(0).norm() / (subs.norm(dim=1).sum() + 1e-12))
    print(f"GRADALIGN {step} {cos:+.4f} {coh:.4f}", flush=True)
print("GRADALIGN done", flush=True)
