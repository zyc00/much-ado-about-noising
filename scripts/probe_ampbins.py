"""Amplitude-binned teacher-forced residual table. Env: CKPT, LOSS, TAG."""
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
dev = cfg.optimization.device; start = cfg.task.obs_steps - 1; AS = cfg.task.act_steps
no = ds.normalizer["obs"]["state"]; na = ds.normalizer["action"]
def predict(wins):
    outs = []
    for i in range(0, len(wins), 256):
        w = wins[i:i+256]
        ot = {"state": torch.tensor(no.normalize(w), device=dev, dtype=torch.float32)}
        with torch.no_grad():
            an = ag.sample(act_0=torch.randn((len(w), 16, 10), device=dev), obs=ot, use_ema=True)
        outs.append(ds.undo_transform_action(na.unnormalize(an.detach().cpu().numpy())[:, start:start+AS]))
    return np.concatenate(outs)
h = h5py.File(DS, "r")
wins, labs = [], []
for i in range(24):
    o = h[f"data/demo_{i}/obs"]
    ov = np.concatenate([np.asarray(o[q]) for q in OK], axis=1).astype(np.float32)
    a = np.clip(np.asarray(h[f"data/demo_{i}/actions"]), -1, 1)
    for t in range(4, len(a) - 18, 3):
        wins.append(np.stack([ov[t-1], ov[t]])); labs.append(a[t:t+AS])
h.close()
wins = np.stack(wins); labs = np.stack(labs)
p = predict(wins)
e = np.linalg.norm(p[:, :, :3] - labs[:, :, :3], axis=2).mean(1)
amp = np.linalg.norm(labs[:, :, :3], axis=2).mean(1)
row = []
for lo, hi in [(0, 0.02), (0.02, 0.1), (0.1, 0.4), (0.4, 1.0), (1.0, 3)]:
    m = (amp >= lo) & (amp < hi)
    row.append(f"{np.median(e[m]):.4f}" if m.sum() >= 10 else "n/a")
print(f"AMPBINS {os.environ.get('TAG','')}: " + " ".join(row) + "  (bins 0-.02/.02-.1/.1-.4/.4-1/1-3)")
