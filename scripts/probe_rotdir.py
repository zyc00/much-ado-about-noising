"""Directional teacher-forced rotation probe: does the predicted rotation residual
(pred - data) at approach/settle states align with the transit-phase mean rotation
direction? Env: CKPT, LOSS, DS, TAG. Queries from full2ins demos, t = c1-24 .. c1-1.
Transit direction computed from the policy's own training dataset (DS)."""
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

# transit-phase mean rotation direction from the policy's own training data
ht = h5py.File(os.environ["DS"], "r")
tkeys = sorted(ht["data"].keys(), key=lambda k: int(k.split("_")[-1]))[:600]
rots = []
for k in tkeys:
    d = ht[f"data/{k}"]
    a = np.clip(np.asarray(d["actions"]), -1, 1); g = a[:, 6]
    cl = [t for t in range(1, len(g)) if g[t-1] < 0 and g[t] >= 0]
    if not cl: continue  # sliced sub-demos without closure: skip (transit slices handled below)
    c1 = cl[0]
    seg = np.asarray(d["actions"])[c1+16:c1+70, 3:6]
    if len(seg): rots.append(seg)
# for datasets where transit lives in separate sub-demos (no closure inside), also take
# any demo whose max |rot| is large
if not rots:
    for k in tkeys:
        seg = np.asarray(ht[f"data/{k}/actions"])[:, 3:6]
        if np.linalg.norm(seg, axis=1).max() > 0.02: rots.append(seg)
ht.close()
R = np.concatenate(rots)
R = R[np.linalg.norm(R, axis=1) > 0.005]
tdir = R.mean(0); tdir /= (np.linalg.norm(tdir) + 1e-12)
print(f"transit rot steps n={len(R)}  mean|rot|={np.linalg.norm(R,axis=1).mean():.4f}  dir={np.array2string(tdir, precision=3)}")

h = h5py.File(QSRC, "r")
keys = sorted(h["data"].keys(), key=lambda k: int(k.split("_")[-1]))[:200]
wins, rels, gts = [], [], []
for k in keys:
    d = h[f"data/{k}"]
    a = np.clip(np.asarray(d["actions"]), -1, 1); g = a[:, 6]
    cl = [t for t in range(1, len(g)) if g[t-1] < 0 and g[t] >= 0]
    if not cl: continue
    c1 = cl[0]
    if c1 < 26 or c1 + 16 >= len(g): continue
    ov = np.concatenate([np.asarray(d["obs"][q]) for q in OKEYS], axis=1).astype(np.float32)
    for t in range(c1 - 24, c1, 2):
        wins.append(ov[t-1:t+1]); rels.append(t - c1)
        gts.append(np.asarray(d["actions"])[t, 3:6])
h.close()
W = np.stack(wins); REL = np.array(rels); GT = np.stack(gts)

preds = []
for i in range(0, len(W), 256):
    ot = {"state": torch.tensor(no.normalize(W[i:i+256]), device=dev, dtype=torch.float32)}
    with torch.no_grad():
        an = ag.sample(act_0=torch.randn((len(ot["state"]), 16, 10), device=dev), obs=ot, use_ema=True)
    ch = ds.undo_transform_action(na.unnormalize(an.detach().cpu().numpy()))
    preds.append(ch[:, start, 3:6])   # first executed action's rotation
P = np.concatenate(preds)

tag = os.environ.get("TAG", "")
for lo, hi in [(-24, -17), (-16, -9), (-8, -1)]:
    m = (REL >= lo) & (REL <= hi)
    res = (P[m] - GT[m]).mean(0)
    proj = float(res @ tdir); mag = float(np.linalg.norm(res))
    cosv = proj / (mag + 1e-12)
    print(f"ROTDIR {tag} rel[{lo:+d},{hi:+d}]: mean-residual |r|={mag:.5f}  proj_on_transit={proj:+.5f}  cos={cosv:+.2f}  (pred mean|rot|={np.linalg.norm(P[m],axis=1).mean():.4f}, data {np.linalg.norm(GT[m],axis=1).mean():.4f})")
print("ROTDIR-DONE")
