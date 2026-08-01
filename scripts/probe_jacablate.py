"""Semantic-group ablation of the encoder Jacobian: zero the columns of a semantic group
and measure the surviving principal gain s1_abl/s1 and PR. Tests redundancy: a feature
spread over many same-semantic columns survives single-column ablation but dies with the
group. Models: MSE, MIPs1, fadehint. Query: canon + dep."""
import os, sys
os.environ["MUJOCO_GL"] = "egl"
sys.path.insert(0, "scripts"); sys.path.insert(0, ".")
import numpy as np, torch
from hydra import initialize_config_dir, compose
from omegaconf import OmegaConf
from mip.datasets.robomimic_dataset import make_dataset
from mip.agent import TrainingAgent

# per-frame semantic groups (53-dim layout)
Z = {"relz": [2, 16, 30], "posz": [9, 23, 37], "eefz": [46]}
G53 = {
    "HEIGHT": Z["relz"] + Z["posz"] + Z["eefz"],
    "XYREL":  [0, 1, 14, 15, 28, 29],
    "XYPOS":  [7, 8, 21, 22, 35, 36, 44, 45],
    "QUAT":   list(range(3, 7)) + list(range(10, 14)) + list(range(17, 21)) + list(range(24, 28)) + list(range(31, 35)) + list(range(38, 42)) + list(range(47, 51)),
    "GRIP":   [51, 52],
    "FLAGS":  [42, 43],
}
GROUPS = {k: v + [d + 53 for d in v] for k, v in G53.items()}
GROUPS["FRAME0"] = list(range(53))
GROUPS["FRAME1"] = list(range(53, 106))

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

MODELS = [
    ("MSE", "regression", "logs/full_regression_2000/models/model_latest.pt"),
    ("MIPs1", "mip", "logs/full_mip_2000_s1/models/model_latest.pt"),
    ("fadehint", "regression_fadehint", "logs/orig_fadehint/models/model_latest.pt"),
]
for name, loss, ck in MODELS:
    cfg, ds, ag = load(loss)
    ag.load(ck, load_optimizer=False); ag.eval(); enc = ag.encoder_ema
    for qname in ["canon", "dep"]:
        res = {g: [] for g in GROUPS}
        base_s1 = []
        for w in Q[qname]:
            x = torch.tensor(no.normalize(w[None]), device=dev, dtype=torch.float32).reshape(1, -1)
            def f(inp): return enc({"state": inp.reshape(1, 2, 53)}, None).reshape(-1)
            J = torch.autograd.functional.jacobian(f, x, vectorize=True).squeeze(1).detach().cpu().numpy()
            s1 = np.linalg.svd(J, compute_uv=False)[0]
            base_s1.append(s1)
            for g, dims in GROUPS.items():
                Ja = J.copy(); Ja[:, dims] = 0
                res[g].append(float(np.linalg.svd(Ja, compute_uv=False)[0] / (s1 + 1e-12)))
        line = f"JACABL {name} {qname}: s1 p50={np.median(base_s1):.2f} |"
        for g in GROUPS:
            line += f" -{g}={np.median(res[g]):.2f}"
        print(line, flush=True)
print("JACABL-DONE")
