"""Teacher-forced rotation-leak probe: feed demo obs windows from late settle
(t = c1-8 .. c1-1), predict the 16-step chunk, and compare rotation-channel
magnitude |a_rot| (axis-angle dims 3:6 of the 7-dim env action) against the demo
actions, binned by chunk-step position relative to closure (t+k-c1).
Query obs always come from full2ins demos (grasp segment identical across slices);
each policy normalizes with its OWN training dataset (DS env).
Env: CKPT, LOSS, DS, TAG."""
import os, sys
os.environ["MUJOCO_GL"] = "egl"
sys.path.insert(0, "scripts"); sys.path.insert(0, ".")
import numpy as np, torch, h5py
from hydra import initialize_config_dir, compose
from omegaconf import OmegaConf
from mip.datasets.robomimic_dataset import make_dataset
from mip.agent import TrainingAgent

QSRC = "data/tool_hang_full2ins_2000.hdf5"
OKEYS = ["object", "robot0_eef_pos", "robot0_eef_quat", "robot0_gripper_qpos"]
with initialize_config_dir(version_base=None, config_dir=os.path.abspath("examples/configs")):
    cfg = compose(config_name="main", overrides=["task=tool_hang_ph_state_delta_legacy",
        "+task.dataset_path=" + os.path.abspath(os.environ["DS"]), "network=chiunet",
        f"optimization.loss_type={os.environ['LOSS']}", "optimization.auto_resume=false", "log.wandb_mode=disabled"])
OmegaConf.set_struct(cfg, False); cfg.task.obs_dim = 53
ds = make_dataset(cfg.task)
ag = TrainingAgent(cfg); ag.load(os.environ["CKPT"], load_optimizer=False); ag.eval()
dev = cfg.optimization.device; start = cfg.task.obs_steps - 1
no = ds.normalizer["obs"]["state"]; na = ds.normalizer["action"]

h = h5py.File(QSRC, "r")
keys = sorted(h["data"].keys(), key=lambda k: int(k.split("_")[-1]))[:200]
wins, rels, gts = [], [], []
for k in keys:
    d = h[f"data/{k}"]
    a = np.clip(np.asarray(d["actions"]), -1, 1); g = a[:, 6]
    cl = [t for t in range(1, len(g)) if g[t-1] < 0 and g[t] >= 0]
    if not cl: continue
    c1 = cl[0]
    if c1 < 10 or c1 + 16 >= len(g): continue
    ov = np.concatenate([np.asarray(d["obs"][q]) for q in OKEYS], axis=1).astype(np.float32)
    for t in range(c1 - 8, c1):
        wins.append(ov[t-1:t+1])
        rels.append(np.arange(15) + t - c1)          # chunk-step position relative to closure
        gts.append(np.linalg.norm(np.asarray(d["actions"])[t:t+15, 3:6], axis=1))
h.close()
W = np.stack(wins); REL = np.stack(rels); GT = np.stack(gts)
print(f"queries: {len(W)}")

preds = []
B = 256
for i in range(0, len(W), B):
    ot = {"state": torch.tensor(no.normalize(W[i:i+B]), device=dev, dtype=torch.float32)}
    with torch.no_grad():
        an = ag.sample(act_0=torch.randn((len(ot["state"]), 16, 10), device=dev), obs=ot, use_ema=True)
    ch = ds.undo_transform_action(na.unnormalize(an.detach().cpu().numpy()))  # (B, 16+, 7)
    preds.append(np.linalg.norm(ch[:, start:start+15, 3:6], axis=2))
P = np.concatenate(preds)[:, :15]

BINS = [(-8, -5), (-4, -1), (0, 3), (4, 7), (8, 15)]
tag = os.environ.get("TAG", "")
for lo, hi in BINS:
    m = (REL >= lo) & (REL <= hi)
    print(f"ROTLEAK {tag} rel[{lo:+d},{hi:+d}]: pred|rot| p50={np.median(P[m]):.4f} p90={np.percentile(P[m],90):.4f} | data p50={np.median(GT[m]):.4f} p90={np.percentile(GT[m],90):.4f}")
print("ROTLEAK-DONE")
