"""Gradient-capture anatomy: x = per-sample loss scale (raw pos inference
residual, log bins), y = sample distribution vs gradient-weight share, per
arm, both regimes. Gradient weight = ||grad_theta loss_i|| via batch-of-1
backward under the arm's OWN objective on the non-EMA nets (4-draw average
for stochastic losses). Prints GW lines."""
import os
import sys

os.environ["MUJOCO_GL"] = "egl"
sys.path.insert(0, "scripts")
sys.path.insert(0, ".")
import numpy as np
import torch
from hydra import compose, initialize_config_dir
from omegaconf import OmegaConf
from torch.utils.data import DataLoader, Subset

from mip.agent import TrainingAgent
from mip.datasets.robomimic_dataset import make_dataset
from mip.losses import get_loss_fn
from mip.samplers import get_sampler

N = int(os.environ.get("GW_N", "1200"))
AS_LO, AS_HI = 1, 9
REGIMES = {
    "script": ("data/tool_hang_full2ins_mp_200.hdf5", [
        ("L2", "regression", "logs/mp200_l2_s1000/models/snap_300000.pt"),
        ("HT", "regression_hetero_t",
         "logs/mp200_ht_s1000/models/snap_300000.pt"),
        ("HG", "regression_hetero_gauss",
         "logs/mp200_hg_s1000/models/snap_300000.pt"),
        ("MIP", "mip", "logs/mp200_mip_s1000/models/snap_300000.pt"),
    ], [3e-5, 1e-4, 3e-4, 1e-3]),
    "human": ("data/tool_hang_human_lowdim_up.hdf5", [
        ("hMSE", "regression", "logs/hbase2/models/model_latest.pt"),
        ("hHT", "regression_hetero_t",
         "logs/hheterot_s1000/models/model_latest.pt"),
        ("hMIP", "mip", "logs/hmip0/models/model_latest.pt"),
    ], [3e-4, 1e-3, 3e-3, 1e-2]),
}

for regime, (dpath, arms, edges) in REGIMES.items():
    for arm, loss, ck in arms:
        with initialize_config_dir(version_base=None,
                                   config_dir=os.path.abspath(
                                       "examples/configs")):
            cfg = compose(config_name="main", overrides=[
                "task=tool_hang_ph_state_delta_legacy",
                "+task.dataset_path=" + os.path.abspath(dpath),
                "network=chiunet", f"optimization.loss_type={loss}",
                "optimization.auto_resume=false", "log.wandb_mode=disabled"])
        OmegaConf.set_struct(cfg, False)
        cfg.task.obs_dim = 53
        cfg.task.horizon = int(2 ** np.ceil(np.log2(cfg.task.horizon)))
        Hn = int(cfg.task.horizon)
        ds = make_dataset(cfg.task)
        ag = TrainingAgent(cfg)
        ag.load(ck, load_optimizer=False)
        sampler = get_sampler(loss)
        lfn = get_loss_fn(loss)
        na = ds.normalizer["action"]
        dev = cfg.optimization.device
        idxs = np.linspace(0, len(ds) - 1, N).astype(int)
        dl = DataLoader(Subset(ds, idxs.tolist()), batch_size=1, shuffle=False)
        params = [p for p in ag.flow_map.parameters()] + \
                 [p for p in ag.encoder.parameters()]
        R, G = [], []
        stoch = loss.startswith("mip")
        for bi, data in enumerate(dl):
            act = data["action"].to(dev)[:, :Hn]
            obs_t = data["obs"]["state"].to(dev)[:, :int(cfg.task.obs_steps)]
            obs = {"state": obs_t}
            delta_t = torch.full((act.shape[0],),
                                 float(cfg.optimization.max_value), device=dev)
            with torch.no_grad():
                a0 = torch.zeros_like(act)
                an = sampler(cfg.optimization, ag.flow_map_ema,
                             ag.encoder_ema, a0, obs)
                pe = np.asarray(ds.undo_transform_action(
                    na.unnormalize(an.cpu().numpy())[:, AS_LO:AS_HI]))
                ge = np.asarray(ds.undo_transform_action(
                    na.unnormalize(act.cpu().numpy())[:, AS_LO:AS_HI]))
            R.append(float(np.abs(pe[:, :, 0:3] - ge[:, :, 0:3]).mean()))
            gs = []
            for d in range(4 if stoch else 1):
                torch.manual_seed(1000 * bi + d)
                for p in params:
                    p.grad = None
                lv, _ = lfn(cfg.optimization, ag.flow_map, ag.encoder,
                            ag.interpolant, act, obs_t, delta_t)
                lv.backward()
                gs.append(float(torch.sqrt(sum(
                    (p.grad ** 2).sum() for p in params
                    if p.grad is not None)).item()))
            G.append(float(np.mean(gs)))
        R, G = np.asarray(R), np.asarray(G)
        bins = np.digitize(R, edges)
        srow, grow = [], []
        for b in range(len(edges) + 1):
            m = bins == b
            srow.append(100 * m.mean())
            grow.append(100 * G[m].sum() / G.sum() if m.any() else 0.0)
        top = R >= np.percentile(R, 90)
        print(f"GW {regime} {arm} bins(edges {edges}) "
              f"samples% {' '.join(f'{v:.1f}' for v in srow)} | grad% "
              f"{' '.join(f'{v:.1f}' for v in grow)} | top10%res grad-share "
              f"{100 * G[top].sum() / G.sum():.1f} | r_p50 {np.median(R):.5f}",
              flush=True)
print("GW done", flush=True)
