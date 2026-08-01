"""Per-dimension Jacobian gain profile (for plotting).

For each arm and band, computes the mean over states of the per-input-feature
Jacobian energy sqrt(||J[:, j]||^2) for all 106 window dims, plus the sorted
(rank-ordered) profile normalized by its total — the direct picture of how
isotropic / concentrated each policy's input dependence is.
Saves DIMPROF npz to /mnt/pfs/yuchen/dimprof.npz and prints summary lines.
Env: DP_ARMS, DP_N.
"""
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

TAG = os.environ.get("DP_TAG", "l2mp200v2")
NPB = int(os.environ.get("DP_N", "40"))
ARMS = [tuple(a.split(":")) for a in os.environ["DP_ARMS"].split(",")]
BANDS = [("on", 0, 2), ("far", 10, 1e9)]

z = np.load(f"analysis/failvids/{TAG}_trajs.npz")
seeds = sorted({int(k[1:]) for k in z.files if k.startswith("W")})
W = np.concatenate([z[f"W{sd}"] for sd in seeds])
D = np.concatenate([z[f"D{sd}"] for sd in seeds])
rng = np.random.RandomState(0)
SEL = {}
for bname, lo, hi in BANDS:
    idx = np.where((D >= lo) & (D < hi))[0]
    SEL[bname] = W[rng.choice(idx, min(NPB, len(idx)), replace=False)]

OUT = {}
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
    dev = cfg.optimization.device
    for bname, Wb in SEL.items():
        prof, sortp = [], []
        for w in Wb:
            x = torch.tensor(no.normalize(w).reshape(-1), device=dev,
                             dtype=torch.float32)

            def f(t_):
                a0 = torch.zeros((1, H, 10), device=dev)
                return sampler(cfg.optimization, ag.flow_map_ema,
                               ag.encoder_ema, a0,
                               {"state": t_.reshape(1, 2, 53)}).reshape(-1)

            J = torch.autograd.functional.jacobian(f, x, vectorize=True)
            e = (J.reshape(-1, x.numel()) ** 2).sum(0).sqrt().cpu().numpy()
            prof.append(e)
            s = np.sort(e)[::-1]
            sortp.append(s / (s.sum() + 1e-30))
        OUT[f"{name}|{bname}|prof"] = np.mean(prof, 0)
        OUT[f"{name}|{bname}|sorted"] = np.mean(sortp, 0)
        sp = np.mean(sortp, 0)
        cum = np.cumsum(sp)
        n50 = int(np.argmax(cum >= 0.5)) + 1
        n90 = int(np.argmax(cum >= 0.9)) + 1
        print(f"DIMPROF {name} {bname} dims_for_50%={n50} dims_for_90%={n90} "
              f"top1={sp[0]:.3f} (isotropic would be 1/106=0.009)", flush=True)
np.savez_compressed("/mnt/pfs/yuchen/dimprof.npz", **OUT)
print("DIMPROF done", flush=True)
