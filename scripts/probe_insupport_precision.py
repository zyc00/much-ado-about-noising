"""High-precision IN-SUPPORT held-out comparison.

Earlier held-out numbers (192 states) disagreed between samples, so nothing
could be concluded about in-support accuracy. This probe fixes that:
  * many more states (default 4000) from demos no arm has seen
  * PAIRED per-state comparison (every arm sees identical states), which
    removes state-to-state variance and is far more sensitive than
    comparing independent means
  * stratified by PHASE, with the insertion/precision phase separated out
  * position error converted to MILLIMETRES of commanded motion, since that
    is what decides whether the frame seats
Reports per arm: mean/median error, and vs the L2 baseline the paired mean
difference with its standard error and a sign test.
Env: IP_ARMS "name:loss:ckpt:dataset", IP_N, IP_BASE. Prints IPREC lines.
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

ARMS = [tuple(a.split(":")) for a in os.environ["IP_ARMS"].split(",")]
NST = int(os.environ.get("IP_N", "4000"))
BASE = os.environ.get("IP_BASE", "L2-200")
HELD = "data/tool_hang_full2ins_mp_20k.hdf5"
OK = ["object", "robot0_eef_pos", "robot0_eef_quat", "robot0_gripper_qpos"]
# robomimic OSC_POSE: action in [-1,1] maps to +-0.05 m per step
POS_SCALE_MM = 50.0

h = h5py.File(HELD, "r")
names = sorted(h["data"].keys(), key=lambda d: int(d.split("_")[1]))[2000:2600]
rng = np.random.RandomState(0)
W, Y, Z = [], [], []
for dn in names:
    o = h[f"data/{dn}/obs"]
    S = np.concatenate([np.asarray(o[k]) for k in OK], 1).astype(np.float32)
    A = np.asarray(h[f"data/{dn}/actions"]).astype(np.float32)
    P = np.asarray(o["robot0_eef_pos"]).astype(np.float32)
    if len(S) < 20:
        continue
    for i in range(1, len(S) - 9, 3):
        W.append(np.stack([S[i - 1], S[i]]))
        Y.append(A[i:i + 8])
        Z.append(P[i, 2])
h.close()
W, Y, Z = np.stack(W), np.stack(Y), np.array(Z)
sel = rng.choice(len(W), min(NST, len(W)), replace=False)
W, Y, Z = W[sel], Y[sel], Z[sel]
INS = Z < 0.86
TRA = (Z >= 0.86) & (Z < 1.02)
APP = Z >= 1.02
print(f"held-out states {len(W)} from {len(names)} unseen demos | "
      f"insertion {int(INS.sum())} transit {int(TRA.sum())} approach "
      f"{int(APP.sum())}", flush=True)

ERR = {}
for name, loss, ck, dset in ARMS:
    with initialize_config_dir(version_base=None,
                               config_dir=os.path.abspath("examples/configs")):
        cfg = compose(config_name="main", overrides=[
            "task=tool_hang_ph_state_delta_legacy",
            "+task.dataset_path=" + os.path.abspath(dset), "network=chiunet",
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
    pe_all = []
    for i in range(0, len(W), 256):
        xb = torch.tensor(np.stack([no.normalize(w) for w in W[i:i + 256]]),
                          device=dev, dtype=torch.float32)
        with torch.no_grad():
            a0 = torch.zeros((len(xb), H, 10), device=dev)
            an = sampler(cfg.optimization, ag.flow_map_ema, ag.encoder_ema,
                         a0, {"state": xb})
        pe_all.append(np.asarray(ds.undo_transform_action(
            na.unnormalize(an.cpu().numpy())[:, 1:9])))
    P = np.concatenate(pe_all)                       # (N, 8, 7)
    d = np.abs(P - Y)
    ERR[name] = dict(
        full=d.mean(axis=(1, 2)),
        pos1=np.linalg.norm((P - Y)[:, 0, 0:3], axis=1) * POS_SCALE_MM,
        pos8=np.linalg.norm((P - Y)[:, :, 0:3], axis=2).mean(1) * POS_SCALE_MM,
        grip=d[:, :, 6].mean(1))
    e = ERR[name]
    for lab, m in (("all", np.ones(len(W), bool)), ("insert", INS),
                   ("transit", TRA), ("approach", APP)):
        print(f"IPREC {name} {lab} n={int(m.sum())} "
              f"err {e['full'][m].mean():.5f} (med {np.median(e['full'][m]):.5f}) "
              f"pos1_mm {e['pos1'][m].mean():.3f} (med "
              f"{np.median(e['pos1'][m]):.3f}) pos8_mm {e['pos8'][m].mean():.3f} "
              f"grip {e['grip'][m].mean():.4f}", flush=True)

print("", flush=True)
for name in ERR:
    if name == BASE:
        continue
    for lab, m in (("all", np.ones(len(W), bool)), ("insert", INS)):
        for key, unit in (("full", ""), ("pos1", "mm"), ("pos8", "mm")):
            a, b = ERR[BASE][key][m], ERR[name][key][m]
            dd = b - a
            se = dd.std(ddof=1) / np.sqrt(len(dd))
            frac = float((dd < 0).mean())
            print(f"IPAIR {name}-vs-{BASE} {lab} {key} mean_diff "
                  f"{dd.mean():+.5f}{unit} +-{se:.5f} (t={dd.mean()/se:+.1f}) "
                  f"median_diff {np.median(dd):+.5f} better_on "
                  f"{frac*100:.1f}% of states", flush=True)
print("IPREC done", flush=True)
