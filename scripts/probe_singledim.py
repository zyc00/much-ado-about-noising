"""Single-dimension perturbation sensitivity, arm by arm.

For each state, perturb EACH of the 106 window features individually by
+/- delta (normalized units) and record |da| in env action units. Report
  mean_dim  : average single-dim sensitivity (how much one feature moves it)
  max_dim   : WORST single-dim sensitivity (the quantity that matters if one
              feature goes bad)
  p90_dim   : 90th percentile over dims
  conc      : max/mean (concentration of dependence on one feature)
  topdims   : which feature groups the worst dims belong to
On-support and far-from-support states. Prints SDIM lines.
Env: SD_ARMS, SD_N, SD_DELTA.
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

TAG = os.environ.get("SD_TAG", "l2mp200v2")
NPB = int(os.environ.get("SD_N", "24"))
DELTA = float(os.environ.get("SD_DELTA", "0.5"))
ARMS = [tuple(a.split(":")) for a in os.environ["SD_ARMS"].split(",")]
BANDS = [("on(<2)", 0, 2), ("far(>10)", 10, 1e9)]
GRP = [("object", 0, 44), ("eefpos", 44, 47), ("eefquat", 47, 51),
       ("grip", 51, 53)]


def group_of(j):
    k = j % 53
    for g, a, b in GRP:
        if a <= k < b:
            return g
    return "?"


z = np.load(f"analysis/failvids/{TAG}_trajs.npz")
seeds = sorted({int(k[1:]) for k in z.files if k.startswith("W")})
W = np.concatenate([z[f"W{sd}"] for sd in seeds])
D = np.concatenate([z[f"D{sd}"] for sd in seeds])
rng = np.random.RandomState(0)
SEL = {}
for bname, lo, hi in BANDS:
    idx = np.where((D >= lo) & (D < hi))[0]
    if len(idx) >= 8:
        SEL[bname] = W[rng.choice(idx, min(NPB, len(idx)), replace=False)]

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

    def act(xf):
        with torch.no_grad():
            a0 = torch.zeros((len(xf), H, 10), device=dev)
            an = sampler(cfg.optimization, ag.flow_map_ema, ag.encoder_ema,
                         a0, {"state": xf.reshape(-1, 2, 53)})
        return np.asarray(ds.undo_transform_action(
            na.unnormalize(an.cpu().numpy())[:, 1:9]))

    for bname, Wb in SEL.items():
        mx, mn, p90, tops = [], [], [], []
        for w in Wb:
            x = torch.tensor(no.normalize(w).reshape(-1), device=dev,
                             dtype=torch.float32)
            base = act(x.reshape(1, -1))
            sens = np.zeros(len(x))
            for sgn in (+1.0, -1.0):
                Xp = x.repeat(len(x), 1)
                Xp[torch.arange(len(x)), torch.arange(len(x))] += sgn * DELTA
                dd = np.abs(act(Xp) - base).mean(axis=(1, 2))
                sens = np.maximum(sens, dd)
            mx.append(sens.max())
            mn.append(sens.mean())
            p90.append(np.percentile(sens, 90))
            tops.extend([group_of(j) for j in np.argsort(sens)[-3:]])
        from collections import Counter
        tc = Counter(tops).most_common(3)
        print(f"SDIM {name} {bname} n={len(Wb)} delta={DELTA} "
              f"mean_dim {np.mean(mn):.4f} max_dim {np.mean(mx):.4f} "
              f"(med {np.median(mx):.4f}) p90_dim {np.mean(p90):.4f} "
              f"conc {np.mean(mx)/max(np.mean(mn),1e-9):.1f} "
              f"topgroups {tc}", flush=True)
print("SDIM done", flush=True)
