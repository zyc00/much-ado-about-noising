"""Multi-block FD Jacobian at canonical vs deployed-failure states.

Input blocks (exact kinematics, rel = R_eef^T (obj - eef)):
  eef   : eef_pos += d, each rel_pos += -R^T d      (physical eef displacement)
  base/frame/tool : obj_pos += d, its rel_pos += +R^T d  (physical object displacement)
  grip  : gripper qpos dims perturbed individually
Outputs: first action's position channels (J_pos) and gripper channel (g_grad).
Env: CKPT, LOSS, TAG.
"""
import os, sys
os.environ["MUJOCO_GL"] = "egl"
sys.path.insert(0, "scripts"); sys.path.insert(0, ".")
import numpy as np, torch
from hydra import initialize_config_dir, compose
from omegaconf import OmegaConf
from mip.datasets.robomimic_dataset import make_dataset
from mip.agent import TrainingAgent

with initialize_config_dir(version_base=None, config_dir=os.path.abspath("examples/configs")):
    cfg = compose(config_name="main", overrides=["task=tool_hang_ph_state_delta_legacy",
        "+task.dataset_path=" + os.path.abspath(os.environ.get("DS", "data/tool_hang_full2ins_2000.hdf5")), "network=chiunet",
        f"optimization.loss_type={os.environ['LOSS']}", "optimization.auto_resume=false", "log.wandb_mode=disabled"])
OmegaConf.set_struct(cfg, False); cfg.task.obs_dim = 53
PAD = os.environ.get("PHASE_PAD")  # e.g. "1,0,0": appended to each frame (oracle-phase models)
if PAD:
    PAD = np.array([float(x) for x in PAD.split(",")])
    cfg.task.obs_dim = 53 + len(PAD)
ds = make_dataset(cfg.task)
ag = TrainingAgent(cfg); ag.load(os.environ["CKPT"], load_optimizer=False); ag.eval()
dev = cfg.optimization.device; AS = cfg.task.act_steps; start = cfg.task.obs_steps - 1
no = ds.normalizer["obs"]["state"]; na = ds.normalizer["action"]

EEF_POS = [44, 45, 46]; EEF_QUAT = [47, 48, 49, 50]; GRIP = [51, 52]
REL = {"base": [0, 1, 2], "frame": [14, 15, 16], "tool": [28, 29, 30]}
OBJ = {"base": [7, 8, 9], "frame": [21, 22, 23], "tool": [35, 36, 37]}
DELTA = 0.002; DELTA_G = 0.001

def quat2R(q):
    x, y, z, w = q
    return np.array([
        [1-2*(y*y+z*z), 2*(x*y-z*w), 2*(x*z+y*w)],
        [2*(x*y+z*w), 1-2*(x*x+z*z), 2*(y*z-x*w)],
        [2*(x*z-y*w), 2*(y*z+x*w), 1-2*(x*x+y*y)]])

def shift_eef(w, axis, sgn):
    w = w.copy(); d = np.zeros(3); d[axis] = sgn * DELTA
    for fr in range(2):
        R = quat2R(w[fr, EEF_QUAT])
        w[fr, EEF_POS] += d
        rd = -R.T @ d
        for rel in REL.values(): w[fr, rel] += rd
    return w

def shift_obj(w, name, axis, sgn):
    w = w.copy(); d = np.zeros(3); d[axis] = sgn * DELTA
    for fr in range(2):
        R = quat2R(w[fr, EEF_QUAT])
        w[fr, OBJ[name]] += d
        w[fr, REL[name]] += R.T @ d
    return w

def shift_grip(w, k, sgn):
    w = w.copy()
    for fr in range(2): w[fr, GRIP[k]] += sgn * DELTA_G
    return w

def first_act(wins):
    if PAD is not None and wins.shape[-1] == 53:
        wins = np.concatenate([wins, np.tile(PAD, (len(wins), 2, 1))], axis=-1)
    ot = {"state": torch.tensor(no.normalize(wins), device=dev, dtype=torch.float32)}
    with torch.no_grad():
        an = ag.sample(act_0=torch.randn((len(wins), 16, 10), device=dev), obs=ot, use_ema=True)
    return ds.undo_transform_action(na.unnormalize(an.detach().cpu().numpy())[:, start:start+AS])[:, 0, :]

BLOCKS = ["eef", "base", "frame", "tool", "grip"]
Q = np.load("scripts/jac_queries.npz")
for qname in ["canon", "dep"]:
    stats = {b: {"fro": [], "sv": [], "gg": []} for b in BLOCKS}
    for w in Q[qname]:
        wins, spec = [], []
        for b in ["eef", "base", "frame", "tool"]:
            for ax in range(3):
                for sg in (+1, -1):
                    wins.append(shift_eef(w, ax, sg) if b == "eef" else shift_obj(w, b, ax, sg))
            spec.append((b, 3, DELTA))
        for k in range(2):
            for sg in (+1, -1): wins.append(shift_grip(w, k, sg))
        spec.append(("grip", 2, DELTA_G))
        A = first_act(np.stack(wins))
        i = 0
        for b, ncol, dl in spec:
            Jp = np.zeros((3, ncol)); Jg = np.zeros(ncol)
            for c in range(ncol):
                Jp[:, c] = (A[i, :3] - A[i+1, :3]) / (2 * dl)
                Jg[c] = (A[i, 6] - A[i+1, 6]) / (2 * dl)
                i += 2
            stats[b]["fro"].append(np.linalg.norm(Jp))
            stats[b]["sv"].append(np.linalg.svd(Jp, compute_uv=False)[0])
            stats[b]["gg"].append(np.linalg.norm(Jg))
    for b in BLOCKS:
        f, s, g = (np.array(stats[b][k]) for k in ("fro", "sv", "gg"))
        print(f"JAC2 {os.environ.get('TAG','')} {qname} {b:5s}: ‖J_pos‖_F p50/p90={np.median(f):.1f}/{np.percentile(f,90):.1f}  σ_max p50/p90={np.median(s):.1f}/{np.percentile(s,90):.1f}  ‖∂a_grip/∂x‖ p50/p90={np.median(g):.1f}/{np.percentile(g,90):.1f}")
print("DONE")
