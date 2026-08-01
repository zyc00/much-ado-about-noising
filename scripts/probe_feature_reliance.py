"""Direct test of "higher PR = output induced by more features = robust".

Two things the singular-value PR does NOT measure, both measured here:
 (1) FEATURE-space participation: per-input-dim Jacobian energy
     e_j = ||J[:, j]||^2 -> PR_feat = (sum e)^2 / sum e^2 ("how many of the
     106 window features the action is actually induced by"), plus the share
     carried by the single most-important feature (top1) and the top-5 share.
 (2) The robustness that claim implies, measured CAUSALLY: corrupt one
     sensor group at a time (object-state 44 / eef-pos 3 / eef-quat 4 /
     gripper 2, both frames) with Gaussian noise, and measure the induced
     |da|; plus worst-case SINGLE-feature ablation (zero one dim, max over
     dims). If the action is induced by many features, no single feature or
     group can swing it much.
Reported at on-support and far-from-support states, per arm.
Env: FR_TAG, FR_ARMS, FR_N, FR_SIG. Prints FEAT lines.
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

TAG = os.environ.get("FR_TAG", "l2mp200v2")
NPB = int(os.environ.get("FR_N", "32"))
SIG = float(os.environ.get("FR_SIG", "0.5"))
NDRAW = int(os.environ.get("FR_DRAW", "12"))
ARMS = [tuple(a.split(":")) for a in os.environ["FR_ARMS"].split(",")]
BANDS = [("on(<2)", 0, 2), ("far(>10)", 10, 1e9)]
# obs layout per frame: object 44 | eef_pos 3 | eef_quat 4 | gripper 2
GROUPS = {"object": (0, 44), "eefpos": (44, 47), "eefquat": (47, 51),
          "grip": (51, 53)}

z = np.load(f"analysis/failvids/{TAG}_trajs.npz")
seeds = sorted({int(k[1:]) for k in z.files if k.startswith("W")})
W = np.concatenate([z[f"W{sd}"] for sd in seeds])
D = np.concatenate([z[f"D{sd}"] for sd in seeds])
rng = np.random.RandomState(int(os.environ.get("FR_SEED", "0")))
SEL = {}
for bname, lo, hi in BANDS:
    idx = np.where((D >= lo) & (D < hi))[0]
    if len(idx) >= 8:
        SEL[bname] = W[rng.choice(idx, min(NPB, len(idx)), replace=False)]


def load(loss, dset, ck):
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
    return cfg, ds, ag, get_sampler(loss), H


for name, loss, ck, dset in ARMS:
    cfg, ds, ag, sampler, H = load(loss, dset, ck)
    no = ds.normalizer["obs"]["state"]
    dev = cfg.optimization.device
    gen = torch.Generator(device=dev)

    def fwd(xflat):
        a0 = torch.zeros((len(xflat), H, 10), device=dev)
        return sampler(cfg.optimization, ag.flow_map_ema, ag.encoder_ema, a0,
                       {"state": xflat.reshape(-1, 2, 53)}).reshape(len(xflat), -1)

    for bname, Wb in SEL.items():
        prf, top1, top5, ablmax, gsens, gshare = [], [], [], [], {g: [] for g in GROUPS}, {g: [] for g in GROUPS}
        for w in Wb:
            x = torch.tensor(no.normalize(w).reshape(-1), device=dev,
                             dtype=torch.float32)
            J = torch.autograd.functional.jacobian(
                lambda t_: fwd(t_.reshape(1, -1)).reshape(-1), x,
                vectorize=True).reshape(-1, x.numel())
            e = (J ** 2).sum(0).double()                    # per-feature energy
            tot = float(e.sum()) + 1e-30
            prf.append(float(e.sum() ** 2 / (e ** 2).sum()))
            es = torch.sort(e, descending=True).values
            top1.append(float(es[0]) / tot)
            top5.append(float(es[:5].sum()) / tot)
            for g, (a, b) in GROUPS.items():
                idxg = list(range(a, b)) + list(range(53 + a, 53 + b))
                gshare[g].append(float(e[idxg].sum()) / tot)
            with torch.no_grad():
                base = fwd(x.reshape(1, -1))
                # single-feature ablation (zero one normalized dim), worst case
                Xa = x.repeat(len(x), 1)
                Xa[torch.arange(len(x)), torch.arange(len(x))] = 0.0
                da = (fwd(Xa) - base).abs().mean(1)
                ablmax.append(float(da.max()))
                # per-group Gaussian corruption
                for g, (a, b) in GROUPS.items():
                    idxg = torch.tensor(list(range(a, b)) +
                                        list(range(53 + a, 53 + b)), device=dev)
                    Xg = x.repeat(NDRAW, 1)
                    gen.manual_seed(3)
                    Xg[:, idxg] += SIG * torch.randn(
                        (NDRAW, len(idxg)), device=dev, generator=gen)
                    gsens[g].append(float((fwd(Xg) - base).abs().mean()))
        gs = " ".join(f"{g}:{np.mean(gshare[g]):.3f}" for g in GROUPS)
        gc = " ".join(f"{g}:{np.mean(gsens[g]):.4f}" for g in GROUPS)
        print(f"FEAT {name} {bname} n={len(Wb)} "
              f"PRfeat {np.mean(prf):.2f}+-{np.std(prf)/np.sqrt(len(prf)):.2f} "
              f"top1 {np.mean(top1):.3f} top5 {np.mean(top5):.3f} "
              f"abl_worst {np.mean(ablmax):.4f} | share[{gs}] | "
              f"corrupt[{gc}]", flush=True)
print("FEAT done", flush=True)
