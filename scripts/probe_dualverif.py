"""Dual-verification real-data leg (PART CCCXV): settle-canon encoder-Jacobian
column-gain GROUP SHARES for the repriced losses (HG x3 seeds, HT) vs L2
controls on the f2i scripted protocol. The pre-registered question: does HG's
settle chart shift toward the separator groups (grip / hand-centric) relative
to L2's height-family loading, or is it height-loaded too (which would refute
survivor-selection as the real-data cure mechanism)?
Modeled on probe_colcos.py (same load pattern, queries, naming)."""
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


def load(loss):
    with initialize_config_dir(version_base=None,
                               config_dir=os.path.abspath("examples/configs")):
        cfg = compose(config_name="main", overrides=[
            "task=tool_hang_ph_state_delta_legacy",
            "+task.dataset_path=" + os.path.abspath("data/tool_hang_full2ins_2000.hdf5"),
            "network=chiunet", f"optimization.loss_type={loss}",
            "optimization.auto_resume=false", "log.wandb_mode=disabled"])
    OmegaConf.set_struct(cfg, False)
    cfg.task.obs_dim = 53
    ds = make_dataset(cfg.task)
    return cfg, ds, TrainingAgent(cfg)


cfg0, ds0, _ = load("regression")
no = ds0.normalizer["obs"]["state"]
dev = cfg0.optimization.device
Q = np.load("scripts/jac_queries.npz")

# per-frame semantic groups (53-dim frame layout, cf. probe_colcos NAME53)
GROUPS = {
    "height_static": [2, 30],          # base_relz, tool_relz
    "grip": [51, 52],
    "eef": [44, 45, 46],
    "frame_rel": [14, 15, 16],
}
IDX = {k: [c + f * 53 for c in v for f in (0, 1)] for k, v in GROUPS.items()}

MODELS = [
    ("L2_s1000", "regression", "logs/f2i_l2_s1000/models/model_latest.pt"),
    ("L2_s42", "regression", "logs/f2i_l2_s42/models/model_latest.pt"),
    ("HG_s1000", "regression_hetero_gauss", "logs/f2i_hg_s1000/models/model_latest.pt"),
    ("HG_s42", "regression_hetero_gauss", "logs/f2i_hg_s42/models/model_latest.pt"),
    ("HG_s5", "regression_hetero_gauss", "logs/f2i_hg_s5/models/model_latest.pt"),
    ("HT_s1000", "regression_hetero_t", "logs/f2i_ht_s1000/models/model_latest.pt"),
]

for name, loss, ck in MODELS:
    if not os.path.exists(ck):
        print(f"DUALVERIF {name}: MISSING {ck}", flush=True)
        continue
    cfg, ds, ag = load(loss)
    ag.load(ck, load_optimizer=False)
    ag.eval()
    enc = ag.encoder_ema

    def fea(inp):
        return enc({"state": inp.reshape(1, 2, 53)}, None).reshape(-1)

    shares = {k: [] for k in IDX}
    shares["other"] = []
    for w in Q["canon"]:
        x = torch.tensor(no.normalize(w[None]), device=dev,
                         dtype=torch.float32).reshape(1, -1)
        J = torch.autograd.functional.jacobian(fea, x, vectorize=True)
        J = J.squeeze(1).detach().cpu().numpy()
        g = np.linalg.norm(J, axis=0) ** 2
        tot = g.sum() + 1e-12
        acc = 0.0
        for k, cols in IDX.items():
            sh = g[cols].sum() / tot
            shares[k].append(sh)
            acc += sh
        shares["other"].append(1.0 - acc)
    out = {k: float(np.mean(v)) for k, v in shares.items()}
    print(f"DUALVERIF {name}: " + " ".join(f"{k}={v:.3f}" for k, v in out.items()),
          flush=True)
print("DUALVERIF done", flush=True)
