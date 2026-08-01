"""Train vs held-out action error per arm (env units), with per-component
breakdown — why is MSE-200's clean held-out error the largest of the
200-demo arms even though it is the objective that minimizes exactly this?
TRAIN states come from each arm's OWN training demos; HELD from demos
>=2000 of the 20k file (unseen by every arm). Prints GAP lines.
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

NST = int(os.environ.get("GG_N", "192"))
ARMS = [tuple(a.split(":")) for a in os.environ["GG_ARMS"].split(",")]
OK = ["object", "robot0_eef_pos", "robot0_eef_quat", "robot0_gripper_qpos"]
HELD = "data/tool_hang_full2ins_mp_20k.hdf5"


def states_from(path, lo, hi, seed):
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
        for i in rng.choice(np.arange(1, len(S) - 9), 3, replace=False):
            Wl.append(np.stack([S[i - 1], S[i]]))
            Yl.append(Aa[i:i + 8])
    h.close()
    W, Y = np.stack(Wl), np.stack(Yl)
    sel = rng.choice(len(W), min(NST, len(W)), replace=False)
    return W[sel], Y[sel]


HW, HY = states_from(HELD, 2000, 2400, 0)
print(f"held-out {len(HW)} states", flush=True)
from scipy.spatial import cKDTree

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
    no = ds.normalizer["obs"]["state"]
    na = ds.normalizer["action"]
    dev = cfg.optimization.device
    ndemo = 200 if "_200" in dset else 2000
    TW, TY = states_from(dset, 0, ndemo, 1)

    def err(W, Y):
        errs = []
        for i in range(0, len(W), 128):
            xb = torch.tensor(
                np.stack([no.normalize(w).reshape(-1) for w in W[i:i + 128]]),
                device=dev, dtype=torch.float32)
            with torch.no_grad():
                a0 = torch.zeros((len(xb), H, 10), device=dev)
                an = sampler(cfg.optimization, ag.flow_map_ema, ag.encoder_ema,
                             a0, {"state": xb.reshape(-1, 2, 53)})
            pe = np.asarray(ds.undo_transform_action(
                na.unnormalize(an.cpu().numpy())[:, 1:9]))
            errs.append(np.abs(pe - Y[i:i + 128]))
        e = np.concatenate(errs)                       # (N, 8, 7)
        per = e.mean(axis=(1, 2))
        return (per.mean(), np.median(per), e[..., 0:3].mean(),
                e[..., 3:6].mean(), e[..., 6].mean())

    # held-out error binned by distance to the TRAINING support
    tn = np.stack([no.normalize(w).reshape(-1) for w in TW])
    hn = np.stack([no.normalize(w).reshape(-1) for w in HW])
    dist, _ = cKDTree(tn).query(hn)
    qs = np.quantile(dist, [0.25, 0.5, 0.75])
    bins = np.digitize(dist, qs)
    eall = []
    for i in range(0, len(HW), 128):
        xb = torch.tensor(hn[i:i + 128], device=dev, dtype=torch.float32)
        with torch.no_grad():
            a0 = torch.zeros((len(xb), H, 10), device=dev)
            an = sampler(cfg.optimization, ag.flow_map_ema, ag.encoder_ema,
                         a0, {"state": xb.reshape(-1, 2, 53)})
        pe = np.asarray(ds.undo_transform_action(
            na.unnormalize(an.cpu().numpy())[:, 1:9]))
        eall.append(np.abs(pe - HY[i:i + 128]).mean(axis=(1, 2)))
    eall = np.concatenate(eall)
    qtxt = " ".join(f"q{b+1}(d~{np.median(dist[bins==b]):.2f}):"
                    f"{eall[bins==b].mean():.4f}" for b in range(4))
    print(f"DIST {name} {qtxt}", flush=True)
    tr, hd = err(TW, TY), err(HW, HY)
    print(f"GAP {name} train[mean {tr[0]:.4f} med {tr[1]:.4f} pos {tr[2]:.4f} "
          f"rot {tr[3]:.4f} grip {tr[4]:.4f}] "
          f"held[mean {hd[0]:.4f} med {hd[1]:.4f} pos {hd[2]:.4f} "
          f"rot {hd[3]:.4f} grip {hd[4]:.4f}] "
          f"gap {hd[0]-tr[0]:+.4f} ratio {hd[0]/max(tr[0],1e-9):.2f}",
          flush=True)
print("GAP done", flush=True)
