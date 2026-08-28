"""Learned response field on a 2D slice, per checkpoint.

For a segment of consecutive dataset windows (real states), displace the
eef-position dims of both frames by dn along a fixed direction n
perpendicular to the local motion, decode the policy's first action, and
project its positional component onto (motion direction, n). Output: a
quiver grid (progress index x normal offset) per model — the learned
counterpart of the schematic panels.

Known instrument limitation (stated on the figure): displaced windows shift
both frames equally; object-relative dims stay at their on-support values.

Env: MODELS="tag:ckpt:loss,...", NORMDS, SEG_DEMO (demo index), SEG_LO,
SEG_N (windows), DN_MM (max offset, default 30), OUT (png path).
"""
import os
import sys

os.environ.setdefault("MUJOCO_GL", "egl")
sys.path.insert(0, "scripts")
sys.path.insert(0, ".")

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
from hydra import compose, initialize_config_dir
from omegaconf import OmegaConf

from mip.agent import TrainingAgent
from mip.datasets.robomimic_dataset import make_dataset

NORMDS = os.environ.get("NORMDS", "data/tool_hang_full2ins_mp_200.hdf5")
SEG_LO = int(os.environ.get("SEG_LO", "80"))
SEG_N = int(os.environ.get("SEG_N", "26"))
DN_MM = float(os.environ.get("DN_MM", "30"))
OUT = os.environ.get("OUT", "analysis/paper/learned_field_slice.png")
POS = slice(44, 47)
H = 16
cfgdir = os.path.abspath("examples/configs")

specs = [sp.split(":")[:3] for sp in os.environ["MODELS"].split(",")]
fig, axes = plt.subplots(1, len(specs), figsize=(5.2 * len(specs), 4.6))
if len(specs) == 1:
    axes = [axes]

ds = None
for ax, (tag, ckpt, loss) in zip(axes, specs):
    with initialize_config_dir(version_base=None, config_dir=cfgdir):
        cfg = compose(config_name="main", overrides=[
            "task=tool_hang_ph_state_delta_legacy",
            "+task.dataset_path=" + os.path.abspath(NORMDS),
            "network=chiunet", f"optimization.loss_type={loss}",
            "task.horizon=16",
            "optimization.auto_resume=false", "log.wandb_mode=disabled"])
    OmegaConf.set_struct(cfg, False)
    cfg.task.obs_dim = 53
    if ds is None:
        ds = make_dataset(cfg.task)
        # consecutive windows from one demo segment (raw, unnormalized)
        wins = []
        for i in range(SEG_LO, SEG_LO + SEG_N):
            w = np.asarray(ds[i]["obs"]["state"])[:2]
            wins.append(ds.normalizer["obs"]["state"].unnormalize(w))
        wins = np.stack(wins)                      # (T, 2, 53) raw
        eef = wins[:, 1, POS]                      # (T, 3)
        motion = np.gradient(eef, axis=0)
        motion = motion / (np.linalg.norm(motion, axis=1, keepdims=True)
                           + 1e-9)
        up = np.array([0.0, 0.0, 1.0])
        nvec = np.cross(motion, up)
        nvec = nvec / (np.linalg.norm(nvec, axis=1, keepdims=True) + 1e-9)
        dns = np.linspace(-DN_MM, DN_MM, 9) / 1000.0
    dev = cfg.optimization.device
    no = ds.normalizer["obs"]["state"]
    na = ds.normalizer["action"]
    ag = TrainingAgent(cfg)
    ag.load(ckpt, load_optimizer=False)
    ag.eval()
    g = torch.Generator(device="cpu").manual_seed(0)
    act0 = torch.randn((1, H, 10), generator=g).to(dev)

    U = np.zeros((len(dns), SEG_N))
    V = np.zeros((len(dns), SEG_N))
    for ti in range(SEG_N):
        for di, dn in enumerate(dns):
            w = wins[ti].copy()
            w[:, POS] = w[:, POS] + dn * nvec[ti]
            x = torch.tensor(no.normalize(w)[None], device=dev,
                             dtype=torch.float32)
            with torch.no_grad():
                an = ag.sample(act_0=act0.clone(), obs=x, use_ema=True)
            a = ds.undo_transform_action(
                na.unnormalize(an.detach().cpu().numpy())[:, 1:2])[0][0]
            apos = a[:3]
            U[di, ti] = float(apos @ motion[ti])
            V[di, ti] = float(apos @ nvec[ti])
    X, Y = np.meshgrid(np.arange(SEG_N), dns * 1000.0)
    mag = np.sqrt(U ** 2 + V ** 2)
    ax.quiver(X, Y, U, V, mag, cmap="coolwarm", scale=np.percentile(mag, 95) * 18,
              width=0.004)
    ax.axhline(0, color="k", lw=2)
    ax.fill_between([0, SEG_N - 1], -9, 9, color="tab:orange", alpha=0.15)
    ax.set_title(f"{tag}  (learned field, projected)", fontsize=11)
    ax.set_xlabel("progress along segment (window index)")
    ax.set_ylabel("normal offset (mm)")
    print(f"MODEL {tag} field done; |a| on-support "
          f"{mag[4].mean():.3f}, at +/-{DN_MM:.0f}mm "
          f"{mag[[0, -1]].mean():.3f}; V at +30mm mean {V[-1].mean():+.4f}, "
          f"V at -30mm mean {V[0].mean():+.4f}", flush=True)

fig.suptitle("Learned action fields on a (progress, normal-offset) slice — "
             "arrows: (along-motion, normal) action components; black line: "
             "demo path; band: +/-9mm", fontsize=12, y=1.00)
fig.tight_layout()
fig.savefig(OUT, dpi=160, bbox_inches="tight")
print("saved", OUT)
print("FIELD-DONE")
