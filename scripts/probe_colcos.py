"""Cosine similarity between peak-colgain Jacobian columns at settle canon.

For each model, per settle-canon query: J = d(embedding)/d(state) (D x 106); rank columns by
gain ||J[:,c]||; among the top-10 columns compute (a) pairwise |cos| of the column vectors,
(b) |cos| of each column with u1 (top left singular vector of J). Controls: (c) mean pairwise
|cos| among 10 random nonzero-gain columns, (d) pairwise |cos| within the static height-proxy
set {base_relz, tool_relz} x 2 frames. Near-parallel top columns = many redundant input
columns feeding ONE embedding feature (rank-1 chart); spread top columns = genuinely
multi-coordinate chart.
"""
import os, sys
os.environ["MUJOCO_GL"] = "egl"
sys.path.insert(0, "scripts"); sys.path.insert(0, ".")
import numpy as np, torch
from hydra import initialize_config_dir, compose
from omegaconf import OmegaConf
from mip.datasets.robomimic_dataset import make_dataset
from mip.agent import TrainingAgent

def load(loss):
    with initialize_config_dir(version_base=None, config_dir=os.path.abspath("examples/configs")):
        cfg = compose(config_name="main", overrides=["task=tool_hang_ph_state_delta_legacy",
            "+task.dataset_path=" + os.path.abspath("data/tool_hang_full2ins_2000.hdf5"), "network=chiunet",
            f"optimization.loss_type={loss}", "optimization.auto_resume=false", "log.wandb_mode=disabled"])
    OmegaConf.set_struct(cfg, False); cfg.task.obs_dim = 53
    ds = make_dataset(cfg.task)
    return cfg, ds, TrainingAgent(cfg)

cfg, ds, _ = load("regression")
no = ds.normalizer["obs"]["state"]; dev = cfg.optimization.device
Q = np.load("scripts/jac_queries.npz")

NAME53 = {}
for i, n in list(zip(range(0, 3), ["base_relx", "base_rely", "base_relz"])) + \
           list(zip(range(3, 7), [f"base_relq{i}" for i in range(4)])) + \
           list(zip(range(7, 10), ["base_px", "base_py", "base_pz"])) + \
           list(zip(range(10, 14), [f"base_q{i}" for i in range(4)])) + \
           list(zip(range(14, 17), ["frame_relx", "frame_rely", "frame_relz"])) + \
           list(zip(range(17, 21), [f"frame_relq{i}" for i in range(4)])) + \
           list(zip(range(21, 24), ["frame_px", "frame_py", "frame_pz"])) + \
           list(zip(range(24, 28), [f"frame_q{i}" for i in range(4)])) + \
           list(zip(range(28, 31), ["tool_relx", "tool_rely", "tool_relz"])) + \
           list(zip(range(31, 35), [f"tool_relq{i}" for i in range(4)])) + \
           list(zip(range(35, 38), ["tool_px", "tool_py", "tool_pz"])) + \
           list(zip(range(38, 42), [f"tool_q{i}" for i in range(4)])) + \
           [(42, "flag0"), (43, "flag1")] + \
           list(zip(range(44, 47), ["eef_px", "eef_py", "eef_pz"])) + \
           list(zip(range(47, 51), [f"eef_q{i}" for i in range(4)])) + \
           [(51, "grip0"), (52, "grip1")]:
    NAME53[i] = n
def cname(c):
    return f"f{c // 53}.{NAME53[c % 53]}"

HEIGHT = [2, 30, 2 + 53, 30 + 53]   # base_relz, tool_relz, both frames

MODELS = [("MSE", "regression", "logs/full_regression_2000/models/model_latest.pt"),
          ("MIPs1", "mip", "logs/full_mip_2000_s1/models/model_latest.pt"),
          ("fadehint", "regression_fadehint", "logs/orig_fadehint/models/model_latest.pt"),
          ("stress10", "regression_stressreg", "logs/orig_stress10/models/model_latest.pt")]

rng = np.random.RandomState(0)
for name, loss, ck in MODELS:
    if not os.path.exists(ck):
        print(f"COLCOS {name}: MISSING {ck}", flush=True); continue
    cfg, ds, ag = load(loss)
    ag.load(ck, load_optimizer=False); ag.eval()
    enc = ag.encoder_ema
    def fea(inp):
        return enc({"state": inp.reshape(1, 2, 53)}, None).reshape(-1)
    top_pair, top_u1, rand_pair, hgt_pair, top_names = [], [], [], [], {}
    for w in Q["canon"]:
        x = torch.tensor(no.normalize(w[None]), device=dev, dtype=torch.float32).reshape(1, -1)
        J = torch.autograd.functional.jacobian(fea, x, vectorize=True).squeeze(1).detach().cpu().numpy()
        g = np.linalg.norm(J, axis=0)
        nz = np.where(g > 1e-8 * g.max())[0]
        top = nz[np.argsort(g[nz])[-10:]]
        for c in top: top_names[cname(c)] = top_names.get(cname(c), 0) + 1
        U, sv, _ = np.linalg.svd(J, full_matrices=False)
        u1 = U[:, 0]
        def paircos(cols):
            C = J[:, cols] / (np.linalg.norm(J[:, cols], axis=0, keepdims=True) + 1e-12)
            S = np.abs(C.T @ C)
            iu = np.triu_indices(len(cols), 1)
            return float(S[iu].mean())
        top_pair.append(paircos(top))
        top_u1.append(float(np.abs((J[:, top] / (np.linalg.norm(J[:, top], axis=0, keepdims=True) + 1e-12)).T @ u1).mean()))
        rand_pair.append(paircos(rng.choice(nz, 10, replace=False)))
        hgt_pair.append(paircos(np.array(HEIGHT)))
    tn = sorted(top_names.items(), key=lambda kv: -kv[1])[:8]
    print(f"COLCOS {name}: top10 pairwise|cos| p50={np.median(top_pair):.2f} | top10-vs-u1 |cos| p50={np.median(top_u1):.2f} | random10 pair p50={np.median(rand_pair):.2f} | heightset pair p50={np.median(hgt_pair):.2f}", flush=True)
    print(f"COLCOS {name} topcols: {', '.join(f'{k}({v})' for k, v in tn)}", flush=True)
print("COLCOS-DONE")
