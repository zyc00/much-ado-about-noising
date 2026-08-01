"""Per-column explained variance of the embedding: for each obs dim d (per frame, summed
over frames), the fraction of total embedding variance linearly explained by that single
column, over settle-phase bank states. Prints a 53-vector per model."""
import os, sys
os.environ["MUJOCO_GL"] = "egl"
sys.path.insert(0, "scripts"); sys.path.insert(0, ".")
import numpy as np, h5py, torch
from hydra import initialize_config_dir, compose
from omegaconf import OmegaConf
from mip.datasets.robomimic_dataset import make_dataset
from mip.agent import TrainingAgent

OK = ["object", "robot0_eef_pos", "robot0_eef_quat", "robot0_gripper_qpos"]
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
h5 = h5py.File("data/tool_hang_full2ins_2000.hdf5", "r")
keys = sorted(h5["data"].keys(), key=lambda k: int(k.split("_")[-1]))
W = []
for k in keys[:200]:
    o = h5[f"data/{k}/obs"]
    ov = np.concatenate([np.asarray(o[q]) for q in OK], axis=1).astype(np.float32)
    a = np.clip(np.asarray(h5[f"data/{k}/actions"]), -1, 1); g = a[:, 6]
    cl = [t for t in range(1, len(g)) if g[t-1] < 0 and g[t] >= 0]
    if not cl: continue
    c1 = cl[0]
    for t in range(max(1, c1-12), c1):   # settle window
        W.append(np.stack([ov[t-1], ov[t]]))
h5.close()
W = np.stack(W)
print(f"settle bank n={len(W)}")

MINMASK = "0,1,2,3,4,5,6,7,8,9,10,11,12,13,21,22,23,24,25,26,27,28,29,30,31,32,33,34,35,36,37,38,39,40,41,42,43"
MODELS = [
    ("MSE", "regression", "logs/full_regression_2000/models/model_latest.pt", None),
    ("minimal", "regression", "logs/obl_minimal/models/model_latest.pt", MINMASK),
    ("fadehint", "regression_fadehint", "logs/orig_fadehint/models/model_latest.pt", None),
]
for name, loss, ck, mask in MODELS:
    if mask: os.environ["OBS_MASK"] = mask
    elif "OBS_MASK" in os.environ: del os.environ["OBS_MASK"]
    cfg, ds, ag = load(loss)
    ag.load(ck, load_optimizer=False); ag.eval()
    outs = []
    for i in range(0, len(W), 512):
        x = torch.tensor(no.normalize(W[i:i+512]), device=dev, dtype=torch.float32)
        with torch.no_grad():
            e = ag.encoder_ema({"state": x}, None)
        outs.append(e.reshape(len(x), -1).cpu().numpy())
    E = np.concatenate(outs)
    E = E - E.mean(0, keepdims=True)
    totvar = (E ** 2).sum()
    X = no.normalize(W).reshape(len(W), -1)   # (n, 106) normalized inputs
    X = X - X.mean(0, keepdims=True)
    r2 = np.zeros(53)
    for d in range(53):
        for fr in (0, 1):
            xd = X[:, fr * 53 + d]
            v = (xd ** 2).sum()
            if v < 1e-9: continue
            cov = E.T @ xd                     # (D,)
            r2[d] += float((cov ** 2).sum() / v / totvar)
    print(f"COLVAR {name}: " + ",".join(f"{v:.4f}" for v in r2), flush=True)
print("COLVAR-DONE")
