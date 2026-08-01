"""Gain vs rank at the REAL off-support states.

Takes the captured rollout states (analysis/failvids/l2mp200v2_trajs.npz —
the states MSE-200 actually visits, including its runaway orbits), strata by
distance-to-support d, and for each arm computes the Jacobian of the
DEPLOYED policy map (full sampler, so MIP's two steps are included) wrt the
observation window: ||J||_F (gain), PR / k90 / smax-smed (shape), and the
output magnitude |a|.

Discriminating cell: HT-200 has a RICHER spectrum than MIP-200 but 12 pts
less SR. If HT's off-support GAIN is high like L2's, gain (not rank) is the
operative quantity; if it is low like MIP's, neither statistic separates it.
Env: OJ_TAG (state source), OJ_ARMS. Prints OFFJAC lines.
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

TAG = os.environ.get("OJ_TAG", "l2mp200v2")
NPB = int(os.environ.get("OJ_N", "24"))
ARMS = [tuple(a.split(":")) for a in os.environ["OJ_ARMS"].split(",")]
BANDS = [("on(<2)", 0, 2), ("annulus(2-4)", 2, 4), ("mid(4-10)", 4, 10),
         ("far(>10)", 10, 1e9)]

z = np.load(f"analysis/failvids/{TAG}_trajs.npz")
seeds = sorted({int(k[1:]) for k in z.files if k.startswith("W")})
W = np.concatenate([z[f"W{sd}"] for sd in seeds])
D = np.concatenate([z[f"D{sd}"] for sd in seeds])
rng = np.random.RandomState(int(os.environ.get("OJ_SEED", "0")))
SEL = {}
for bname, lo, hi in BANDS:
    idx = np.where((D >= lo) & (D < hi))[0]
    if len(idx) >= 8:
        SEL[bname] = W[rng.choice(idx, min(NPB, len(idx)), replace=False)]
    print(f"BAND {bname} n_states={len(idx)}", flush=True)


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
    na = ds.normalizer["action"]
    dev = cfg.optimization.device
    for bname, Wb in SEL.items():
        fro, prs, k90s, rats, mags = [], [], [], [], []
        for w in Wb:
            x = torch.tensor(no.normalize(w).reshape(-1), device=dev,
                             dtype=torch.float32)

            def f(inp):
                a0 = torch.zeros((1, H, 10), device=dev)
                return sampler(cfg.optimization, ag.flow_map_ema,
                               ag.encoder_ema, a0,
                               {"state": inp.reshape(1, 2, 53)}).reshape(-1)

            J = torch.autograd.functional.jacobian(f, x, vectorize=True)
            J = J.reshape(-1, x.numel())
            s = torch.linalg.svdvals(J.double())
            s2 = s ** 2
            fro.append(float(s2.sum().sqrt()))
            prs.append(float(s2.sum() ** 2 / (s2 ** 2).sum()))
            c = torch.cumsum(s2, 0) / s2.sum()
            k90s.append(int((c < 0.90).sum()) + 1)
            rats.append(float(s[0] / (s[len(s) // 2] + 1e-12)))
            with torch.no_grad():
                a = f(x).reshape(H, 10)
            mags.append(float(np.abs(na.unnormalize(
                a.cpu().numpy()[None])[0][1:9, 0:3]).mean()))
        se = np.std(prs) / np.sqrt(len(prs))
        print(f"OFFJAC {name} {bname} n={len(Wb)} "
              f"Jfro {np.mean(fro):.2f} (med {np.median(fro):.2f}) "
              f"PR {np.mean(prs):.3f}+-{se:.3f} k90 {np.mean(k90s):.1f} "
              f"ratio {np.median(rats):.1f} |apos| {np.mean(mags):.4f}",
              flush=True)
print("OFFJAC done", flush=True)
