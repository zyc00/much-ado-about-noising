"""Why does MIP win at MP-200? Discriminating probe (manifold story rejected).
H-gain: noised-action training bounds the state-side gain off-support
  (L2's off/on ||J_x|| ratio grows with displacement; MIP's stays flat)
H-track: MIP's outputs remain locked to nearest-demo actions off-support
  (the 'return-to-data' signature a manifold/tracking story needs)
For 30 support states, displace along random normalized-state rays at
delta in {0, 1, 2, 4, 8}; report per arm: ||J_x||_F, ||a_pred||,
|a_pred - a_NNdemo|. Prints WHYMIP lines.
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
from scipy.spatial import cKDTree

from mip.agent import TrainingAgent
from mip.datasets.robomimic_dataset import make_dataset

DSET = os.environ.get("WM_DATASET", "data/tool_hang_full2ins_mp_200.hdf5")
ARMS = [tuple(a.split(":")) for a in os.environ["WM_ARMS"].split(",")]


def load(loss):
    with initialize_config_dir(version_base=None,
                               config_dir=os.path.abspath("examples/configs")):
        cfg = compose(config_name="main", overrides=[
            "task=tool_hang_ph_state_delta_legacy",
            "+task.dataset_path=" + os.path.abspath(DSET),
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
AD = A_all.shape[1]
no = ds0.normalizer["obs"]["state"]
na = ds0.normalizer["action"]
dev = cfg0.optimization.device

starts = np.concatenate([[0], ends[:-1]])
W_list, A_list = [], []
for e in range(min(200, len(ends))):
    s0, e0 = int(starts[e]), int(ends[e])
    T = e0 - s0
    if T < H + 3:
        continue
    S, A = S_all[s0:e0], A_all[s0:e0]
    for i in range(1, T - H, 2):
        W_list.append(no.normalize(np.stack([S[i - 1], S[i]])).reshape(-1))
        A_list.append(na.normalize(A[i:i + H]).reshape(-1))
WN = np.stack(W_list)
AN = np.stack(A_list)
tree = cKDTree(WN)
rng = np.random.RandomState(0)
IDX = rng.choice(len(WN), 30, replace=False)
DELTAS = [0.0, 1.0, 2.0, 4.0, 8.0]
RAYS = {i: rng.randn(len(IDX), WN.shape[1]) for i in range(1)}
R = RAYS[0] / np.linalg.norm(RAYS[0], axis=1, keepdims=True)
# scale: normalized-window units; kdtree distances in same space
SIG = WN.std(0).mean()

for name, loss, ck in ARMS:
    cfg, ds, ag = load(loss)
    ag.load(ck, load_optimizer=False)
    ag.eval()

    def f_out(x_np):
        x = torch.tensor(x_np, device=dev, dtype=torch.float32)
        t = torch.zeros(len(x), device=dev)
        act0 = torch.zeros(len(x), H, AD, device=dev)
        emb = ag.encoder_ema({"state": x.reshape(-1, 2, 53)}, None)
        with torch.no_grad():
            pred, _ = ag.flow_map_ema.net(act0, t, t, emb)
        return pred.reshape(len(x), -1).cpu().numpy()

    def jfro(x_np):
        x = torch.tensor(x_np, device=dev, dtype=torch.float32)

        def g(inp):
            t = torch.zeros(1, device=dev)
            act0 = torch.zeros(1, H, AD, device=dev)
            emb = ag.encoder_ema({"state": inp.reshape(1, 2, 53)}, None)
            pred, _ = ag.flow_map_ema.net(act0, t, t, emb)
            return pred.reshape(-1)

        J = torch.autograd.functional.jacobian(g, x.reshape(-1),
                                               vectorize=True)
        return float((J ** 2).sum() ** 0.5)

    for dl in DELTAS:
        X = WN[IDX] + dl * SIG * R
        out = f_out(X)
        _, nn = tree.query(X)
        a_nn = AN[nn]
        jn = np.mean([jfro(X[i:i + 1]) for i in range(0, len(X), 3)])
        anorm = float(np.abs(out).mean())
        atrack = float(np.abs(out - a_nn).mean())
        print(f"WHYMIP {name} delta {dl:.0f} Jfro {jn:.2f} "
              f"|a| {anorm:.4f} |a-aNN| {atrack:.4f}", flush=True)
print("WHYMIP done", flush=True)
