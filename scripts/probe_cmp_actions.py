"""Cross-policy action comparison on captured trajectories.
Loads the four *_trajs.npz (raw obs windows W + tube distance D) and the four
policies; evaluates every policy on every arm's states, stratified by
d-band; reports pairwise mean |delta a| (unnormalized units) and per-arm
trajectory stats. Prints CMP lines.
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

ARMS = [
    ("L2-200", "regression", "logs/mp200_l2_s1000/models/snap_300000.pt",
     "data/tool_hang_full2ins_mp_200.hdf5", "l2mp200v2"),
    ("MIP-200", "mip", "logs/mp200_mip_s1000/models/snap_300000.pt",
     "data/tool_hang_full2ins_mp_200.hdf5", "mipmp200"),
    ("L2-2k", "regression", "logs/mpf2i_l2_s1000/models/snap_300000.pt",
     "data/tool_hang_full2ins_mp_2000.hdf5", "l2mp2k"),
    ("MIP-2k", "mip", "logs/mpf2i_mip_s1000/models/snap_300000.pt",
     "data/tool_hang_full2ins_mp_2000.hdf5", "mipmp2k"),
]

POL = {}
for name, loss, ck, dset, tag in ARMS:
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
    POL[name] = (cfg, ds, ag, get_sampler(loss), H)

start = 1  # obs_steps - 1


def act_of(name, W_raw):
    cfg, ds, ag, sampler, H = POL[name]
    no = ds.normalizer["obs"]["state"]
    na = ds.normalizer["action"]
    outs = []
    B = 256
    for i in range(0, len(W_raw), B):
        w = W_raw[i:i + B]
        x = torch.tensor(np.stack([no.normalize(wi) for wi in w]),
                         device=cfg.optimization.device, dtype=torch.float32)
        with torch.no_grad():
            a0 = torch.zeros((len(w), H, 10), device=x.device)
            an = sampler(cfg.optimization, ag.flow_map_ema, ag.encoder_ema,
                         a0, {"state": x})
        outs.append(na.unnormalize(an.detach().cpu().numpy())[:, start:start + 8])
    return np.concatenate(outs)


BANDS = [("on(d<2)", 0, 2), ("annulus(2-4)", 2, 4), ("far(d>10)", 10, 1e9)]
for src_name, _, _, _, tag in ARMS:
    d = np.load(f"analysis/failvids/{tag}_trajs.npz")
    seeds = sorted({int(k[1:]) for k in d.files if k.startswith("W")})
    W = np.concatenate([d[f"W{sd}"] for sd in seeds])
    Dd = np.concatenate([d[f"D{sd}"] for sd in seeds])
    # per-arm trajectory stats
    oks = [bool(d[f"O{sd}"][0]) for sd in seeds]
    steps = [len(d[f"D{sd}"]) for sd in seeds]
    print(f"CMPSTAT {src_name} SR {np.mean(oks):.2f} "
          f"steps(succ) {np.mean([s for s, o in zip(steps, oks) if o]):.0f} "
          f"maxd_med {np.median([d[f'D{sd}'].max() for sd in seeds]):.1f}",
          flush=True)
    for bname, lo, hi in BANDS:
        m = (Dd >= lo) & (Dd < hi)
        if m.sum() < 20:
            print(f"CMP {src_name} {bname} n={int(m.sum())} (skip)", flush=True)
            continue
        idx = np.where(m)[0]
        if len(idx) > 400:
            idx = idx[np.linspace(0, len(idx) - 1, 400).astype(int)]
        acts = {n: act_of(n, W[idx]) for n in POL}
        pairs = [("L2-200", "L2-2k"), ("L2-200", "MIP-200"),
                 ("L2-2k", "MIP-2k"), ("MIP-200", "MIP-2k"),
                 ("L2-2k", "MIP-200")]
        s = " ".join(
            f"{a}~{b}:{np.abs(acts[a] - acts[b]).mean():.4f}"
            for a, b in pairs)
        mag = " ".join(f"|{n}|:{np.abs(acts[n]).mean():.4f}" for n in POL)
        print(f"CMP {src_name} {bname} n={len(idx)} {s} || {mag}", flush=True)
print("CMPACT done", flush=True)
