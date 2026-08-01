"""On-manifold plateau test: pre-NN -> post-NN line, g zero-crossing alpha*.
Usage: CKPT=... LOSS=... python scripts/probe_plateau.py"""
import os, sys
os.environ["MUJOCO_GL"] = "egl"
sys.path.insert(0, "scripts"); sys.path.insert(0, ".")
import numpy as np, h5py, torch
from hydra import initialize_config_dir, compose
from omegaconf import OmegaConf
from mip.datasets.robomimic_dataset import make_dataset
from mip.agent import TrainingAgent
OK = ["object", "robot0_eef_pos", "robot0_eef_quat", "robot0_gripper_qpos"]
DS = os.environ.get("DS", "data/tool_hang_full2ins_2000.hdf5")
with initialize_config_dir(version_base=None, config_dir=os.path.abspath("examples/configs")):
    cfg = compose(config_name="main", overrides=["task=tool_hang_ph_state_delta_legacy",
        f"+task.dataset_path={os.path.abspath(DS)}", "network=chiunet",
        f"optimization.loss_type={os.environ['LOSS']}", "optimization.auto_resume=false", "log.wandb_mode=disabled"])
OmegaConf.set_struct(cfg, False); cfg.task.obs_dim = 53
ds = make_dataset(cfg.task)
ag = TrainingAgent(cfg); ag.load(os.environ["CKPT"], load_optimizer=False); ag.eval()
dev = cfg.optimization.device; start = cfg.task.obs_steps - 1
no = ds.normalizer["obs"]["state"]; na = ds.normalizer["action"]
def g_of(wins):
    ot = {"state": torch.tensor(no.normalize(wins), device=dev, dtype=torch.float32)}
    with torch.no_grad():
        an = ag.sample(act_0=torch.randn((len(wins), 16, 10), device=dev), obs=ot, use_ema=True)
    return ds.undo_transform_action(na.unnormalize(an.detach().cpu().numpy())[:, start:start+8])[:, 0, 6]
h = h5py.File(DS, "r"); keys = sorted(h["data"].keys(), key=lambda k: int(k.split("_")[-1]))
pairs = []
for k in keys[:60]:
    o = h[f"data/{k}/obs"]
    ov = np.concatenate([np.asarray(o[q]) for q in OK], axis=1).astype(np.float32)
    a = np.clip(np.asarray(h[f"data/{k}/actions"]), -1, 1); g = a[:, 6]
    cl = [t for t in range(1, len(g)) if g[t-1] < 0 and g[t] >= 0]
    if not cl: continue
    c1 = cl[0]
    if c1 - 7 < 1 or c1 + 9 >= len(ov): continue
    pairs.append((np.stack([ov[c1-7], ov[c1-6]]), np.stack([ov[c1+7], ov[c1+8]])))
h.close(); pairs = pairs[:40]
alphas = np.linspace(0, 1, 21); cross = []
for wpre, wpost in pairs:
    wins = np.stack([(1-al)*wpre + al*wpost for al in alphas])
    gs = g_of(wins); idx = np.where(gs > 0)[0]
    cross.append(alphas[idx[0]] if len(idx) else np.nan)
cross = np.array(cross)
print(f"PLATEAU {os.environ.get('TAG','')}: alpha* p25={np.nanpercentile(cross,25):.2f} p50={np.nanmedian(cross):.2f} p75={np.nanpercentile(cross,75):.2f} (MSE ref 0.15, MIP ref 0.10)")
