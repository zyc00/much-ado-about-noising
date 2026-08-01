"""Encoder-Jacobian PR at settle states, over snapshot grids, f2i family.
Protocol matches PART LXXXIV: J = d(phi)/d(input window) by autograd at
settle-core states; PR(s^2) = (sum s^2)^2 / sum s^4 over singular values;
also k90 (PCs to 90% of sum s^2) and smax/smed.
Prints: JACLAD <arm> <step> <PR> <k90> <smax/smed>
"""
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


DATASET = os.environ.get("JL_DATASET", "data/tool_hang_full2ins_2000.hdf5")


def load(loss):
    with initialize_config_dir(version_base=None,
                               config_dir=os.path.abspath("examples/configs")):
        cfg = compose(config_name="main", overrides=[
            "task=tool_hang_ph_state_delta_legacy",
            "+task.dataset_path=" + os.path.abspath(DATASET),
            "network=chiunet", f"optimization.loss_type={loss}",
            "optimization.auto_resume=false", "log.wandb_mode=disabled"])
    OmegaConf.set_struct(cfg, False)
    cfg.task.obs_dim = 53
    cfg.task.horizon = int(2 ** np.ceil(np.log2(cfg.task.horizon)))
    ds = make_dataset(cfg.task)
    return cfg, ds, TrainingAgent(cfg)


cfg0, ds0, _ = load("regression")
rb = ds0.replay_buffer
ends = rb.episode_ends[:]
obs_entry = rb["obs"]
try:
    S_all = obs_entry["state"][:]
except (TypeError, IndexError, KeyError):
    S_all = obs_entry[:]
A_all = rb["action"][:]
H = int(cfg0.task.horizon)
no = ds0.normalizer["obs"]["state"]
dev = cfg0.optimization.device

# settle-core states (same mining as probe_grad_balance)
wins = []
start = 0
for e in range(min(300, len(ends))):
    end = int(ends[e])
    S, A = S_all[start:end], A_all[start:end]
    T = len(S)
    start = end
    if T < 60:
        continue
    an = np.linalg.norm(A, axis=1)
    lo, hi = T // 3, 2 * T // 3
    settle = lo + int(np.argmin(an[lo:hi]))
    if settle >= 1 and settle + H < T:
        seg = an[settle:settle + H]
        if seg.max() < 2.5 * max(an[settle], 1e-6) and len(wins) < 20:
            wins.append(np.stack([S[settle - 1], S[settle]]))
print(f"settle states: {len(wins)}", flush=True)
X = torch.tensor(np.stack([no.normalize(w) for w in wins]), device=dev,
                 dtype=torch.float32)  # (20, 2, 53)


def spec_stats(ag):
    prs, k90s, ratios = [], [], []
    for i in range(len(X)):
        x = X[i:i + 1].clone().requires_grad_(True)

        def f(inp):
            return ag.encoder_ema({"state": inp}, None).reshape(-1)

        J = torch.autograd.functional.jacobian(f, x, vectorize=True)
        J = J.reshape(J.shape[0], -1)  # (emb, 106)
        s = torch.linalg.svdvals(J)
        s2 = (s ** 2)
        pr = float(s2.sum() ** 2 / (s2 ** 2).sum())
        c = torch.cumsum(s2, 0) / s2.sum()
        k90 = int((c < 0.90).sum()) + 1
        ratios.append(float(s[0] / (s[len(s) // 2] + 1e-12)))
        prs.append(pr)
        k90s.append(k90)
    return float(np.mean(prs)), float(np.mean(k90s)), float(np.median(ratios))


ARMS = [("L2", "regression", "logs/f2i_l2_s1000"),
        ("HG", "regression_hetero_gauss", "logs/f2i_hg_s1000"),
        ("HT", "regression_hetero_t", "logs/f2i_ht_s1000"),
        ("MIP", "mip", "logs/full_mip_2000_s2")]
if os.environ.get("JL_ARMS"):
    ARMS = [tuple(a.split(":")) for a in os.environ["JL_ARMS"].split(",")]
for name, loss, d in ARMS:
    cfg, ds, ag = load(loss)
    cks = sorted(glob.glob(f"{d}/models/snap_*.pt"),
                 key=lambda p: int(p.split("snap_")[1].split(".")[0]))
    if not cks:
        cks = [f"{d}/models/model_latest.pt"]
    for ck in cks:
        if not os.path.exists(ck):
            continue
        step = (int(ck.split("snap_")[1].split(".")[0])
                if "snap_" in ck else 300000)
        ag.load(ck, load_optimizer=False)
        ag.eval()
        pr, k90, rat = spec_stats(ag)
        print(f"JACLAD {name} {step} {pr:.2f} {k90:.1f} {rat:.1f}", flush=True)
print("JACLAD done", flush=True)
