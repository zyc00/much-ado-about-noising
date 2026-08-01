"""Descent-phase z fixed-point probe: at pre-grasp rollout states, impose
z-offsets on the eef obs dims and read the commanded first-step z-delta;
fit slope (z-stiffness) and zero-crossing per state; report per-arm slope
p50 and zero-crossing dispersion. Prints ZF lines."""
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
from mip.samplers import get_sampler

D2 = "data/tool_hang_full2ins_mp_200.hdf5"
ARMS = [
    ("HT", "regression_hetero_t", "logs/mp200_ht_s1000/models/snap_300000.pt",
     "logs/rd_ht"),
    ("HG", "regression_hetero_gauss",
     "logs/mp200_hg_s1000/models/snap_300000.pt", "logs/rd_hg"),
    ("MIP", "mip", "logs/mp200_mip_s1000/models/snap_300000.pt",
     "logs/rd_mip"),
]
OFFS = np.array([-0.05, -0.03, -0.015, 0.0, 0.015, 0.03, 0.05])

with initialize_config_dir(version_base=None,
                           config_dir=os.path.abspath("examples/configs")):
    cfg0 = compose(config_name="main", overrides=[
        "task=tool_hang_ph_state_delta_legacy",
        "+task.dataset_path=" + os.path.abspath(D2), "network=chiunet",
        "optimization.auto_resume=false", "log.wandb_mode=disabled"])
OmegaConf.set_struct(cfg0, False)
cfg0.task.obs_dim = 53
cfg0.task.horizon = int(2 ** np.ceil(np.log2(cfg0.task.horizon)))
Hn = int(cfg0.task.horizon)
ds0 = make_dataset(cfg0.task)
no, na0 = ds0.normalizer["obs"]["state"], ds0.normalizer["action"]
start = int(cfg0.task.obs_steps) - 1

for arm, loss, ck, dr in ARMS:
    cfg = OmegaConf.create(OmegaConf.to_container(cfg0))
    cfg.optimization.loss_type = loss
    ag = TrainingAgent(cfg)
    ag.load(ck, load_optimizer=False)
    ag.eval()
    sampler = get_sampler(loss)
    dev = cfg.optimization.device
    slopes, cross, epzero = [], [], []
    for f in sorted(glob.glob(f"{dr}/ep_*.npz")):
        z = np.load(f)
        obs = z["obs"]
        g = obs[:, 51:53].sum(1)
        thr = 0.5 * (g.max() + g.min())
        closed = np.where(g < thr)[0]
        if not len(closed):
            continue
        tg = int(closed[0])
        st = [i for i in range(max(1, tg - 24), tg, 8)]
        ep_c = []
        for i in st:
            w = np.stack([obs[i - 1], obs[i]])
            zc = []
            for off in OFFS:
                wp = w.copy()
                wp[:, 46] += off
                xb = torch.tensor(no.normalize(wp)[None], device=dev,
                                  dtype=torch.float32)
                with torch.no_grad():
                    a0 = torch.zeros((1, Hn, 10), device=dev)
                    an = sampler(cfg.optimization, ag.flow_map_ema,
                                 ag.encoder_ema, a0, {"state": xb})
                pe = np.asarray(ds0.undo_transform_action(
                    na0.unnormalize(an.cpu().numpy())[:, start:start + 4]))
                zc.append(pe[0, :, 2].mean())
            zc = np.asarray(zc)
            A = np.stack([OFFS, np.ones_like(OFFS)], 1)
            (m, b), *_ = np.linalg.lstsq(A, zc, rcond=None)
            slopes.append(m)
            if abs(m) > 1e-4:
                x0 = -b / m
                if abs(x0) < 0.2:
                    cross.append(x0)
                    ep_c.append(x0)
        if ep_c:
            epzero.append(np.mean(ep_c))
    slopes, cross = np.asarray(slopes), np.asarray(cross)
    epz = np.asarray(epzero)
    print(f"ZF {arm} z-stiffness p50 {np.median(slopes):+.4f} p10 "
          f"{np.percentile(slopes, 10):+.4f} | crossing spread(all) "
          f"{1000 * np.std(cross):.1f}mm | per-episode crossing std "
          f"{1000 * np.std(epz):.1f}mm (n_ep={len(epz)})", flush=True)
print("ZF done", flush=True)
