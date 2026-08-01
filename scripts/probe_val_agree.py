"""Pairwise policy agreement on HELD-OUT states (env action units).

Does MIP at 200 demos behave like MSE with 10x the data? For 192 held-out
states (demos >=2000 of the 20k MP file, unseen by every arm) we predict with
every arm and report (a) each arm's distance to GT, (b) the full pairwise
|da| matrix, (c) the same split by distance to the 200-demo training support.
Prints VAGREE lines.
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

NST = int(os.environ.get("VA_N", "192"))
ARMS = [tuple(a.split(":")) for a in os.environ["VA_ARMS"].split(",")]
OK = ["object", "robot0_eef_pos", "robot0_eef_quat", "robot0_gripper_qpos"]
HELD = "data/tool_hang_full2ins_mp_20k.hdf5"
TRAIN200 = "data/tool_hang_full2ins_mp_200.hdf5"


def states_from(path, lo, hi, seed, per=3):
    h = h5py.File(path, "r")
    names = sorted(h["data"].keys(), key=lambda d: int(d.split("_")[1]))[lo:hi]
    rng = np.random.RandomState(seed)
    Wl, Yl = [], []
    for dn in names:
        o = h[f"data/{dn}/obs"]
        S = np.concatenate([np.asarray(o[k]) for k in OK],
                           axis=1).astype(np.float32)
        Aa = np.asarray(h[f"data/{dn}/actions"]).astype(np.float32)
        if len(S) < 12:
            continue
        for i in rng.choice(np.arange(1, len(S) - 9), per, replace=False):
            Wl.append(np.stack([S[i - 1], S[i]]))
            Yl.append(Aa[i:i + 8])
    h.close()
    return np.stack(Wl), np.stack(Yl)


HW, HY = states_from(HELD, 2000, 2400, 0)
sel = np.random.RandomState(0).choice(len(HW), min(NST, len(HW)), replace=False)
HW, HY = HW[sel], HY[sel]
TW, _ = states_from(TRAIN200, 0, 200, 1, per=4)
print(f"held-out {len(HW)} | train-200 ref {len(TW)}", flush=True)

PRED = {}
for name, loss, ck, dset in ARMS:
    with initialize_config_dir(version_base=None,
                               config_dir=os.path.abspath("examples/configs")):
        cfg = compose(config_name="main", overrides=[
            "task=tool_hang_ph_state_delta_legacy",
            "+task.dataset_path=" + os.path.abspath(dset),
            "network=chiunet", f"optimization.loss_type={loss}",
            "optimization.auto_resume=false", "log.wandb_mode=disabled"])
    OmegaConf.set_struct(cfg, False)
    cfg.task.obs_dim = 53
    H = int(2 ** np.ceil(np.log2(cfg.task.horizon)))
    cfg.task.horizon = H
    ds = make_dataset(cfg.task)
    ag = TrainingAgent(cfg)
    ag.load(ck, load_optimizer=False)
    ag.eval()
    sampler = get_sampler(loss)
    no, na = ds.normalizer["obs"]["state"], ds.normalizer["action"]
    dev = cfg.optimization.device
    out = []
    for i in range(0, len(HW), 128):
        xb = torch.tensor(
            np.stack([no.normalize(w).reshape(-1) for w in HW[i:i + 128]]),
            device=dev, dtype=torch.float32)
        with torch.no_grad():
            a0 = torch.zeros((len(xb), H, 10), device=dev)
            an = sampler(cfg.optimization, ag.flow_map_ema, ag.encoder_ema,
                         a0, {"state": xb.reshape(-1, 2, 53)})
        out.append(np.asarray(ds.undo_transform_action(
            na.unnormalize(an.cpu().numpy())[:, 1:9])))
    PRED[name] = np.concatenate(out)
    if name == "L2-200":                      # common distance reference
        tn = np.stack([no.normalize(w).reshape(-1) for w in TW])
        hn = np.stack([no.normalize(w).reshape(-1) for w in HW])
        DIST, _ = cKDTree(tn).query(hn)
    print(f"VAGREE-GT {name} err {np.abs(PRED[name] - HY).mean():.4f}",
          flush=True)

names = list(PRED)
print("VAGREE matrix (mean |da|, env units, held-out states):", flush=True)
print("VAGREE " + " ".join(f"{n:>9}" for n in ["", *names, "GT"]), flush=True)
for a in names:
    row = [f"{np.abs(PRED[a] - PRED[b]).mean():9.4f}" for b in names]
    row.append(f"{np.abs(PRED[a] - HY).mean():9.4f}")
    print(f"VAGREE {a:>9} " + " ".join(row), flush=True)

med = np.median(DIST)
for lab, m in [("near", DIST <= med), ("far", DIST > med)]:
    pairs = [("MIP-200", "L2-2k"), ("MIP-200", "L2-200"), ("MIP-200", "MIP-2k"),
             ("L2-200", "L2-2k"), ("HG-200", "L2-2k"), ("HT-200", "L2-2k")]
    s = " ".join(f"{a}~{b}:{np.abs(PRED[a][m] - PRED[b][m]).mean():.4f}"
                 for a, b in pairs if a in PRED and b in PRED)
    g = " ".join(f"{n}~GT:{np.abs(PRED[n][m] - HY[m]).mean():.4f}"
                 for n in names)
    print(f"VAGREE-{lab} n={int(m.sum())} {s} || {g}", flush=True)
print("VAGREE done", flush=True)
