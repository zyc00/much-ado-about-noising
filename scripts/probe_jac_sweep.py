"""FD Jacobian growth (eef/frame/grip blocks) swept over training snapshots.
Env: SNAP_GLOB (e.g. 'logs/mse_timeline/models/snap_*.pt'), LOSS, TAG, optional DS.
Loads dataset/agent once, iterates checkpoints, prints one line per (ckpt, block)."""
import os, sys, glob, re
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
ds = make_dataset(cfg.task)
ag = TrainingAgent(cfg)
dev = cfg.optimization.device; AS = cfg.task.act_steps; start = cfg.task.obs_steps - 1
no = ds.normalizer["obs"]["state"]; na = ds.normalizer["action"]

EEF_POS = [44, 45, 46]; EEF_QUAT = [47, 48, 49, 50]; GRIP = [51, 52]
REL = {"base": [0, 1, 2], "frame": [14, 15, 16], "tool": [28, 29, 30]}
OBJ = {"frame": [21, 22, 23]}
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

def shift_frame(w, axis, sgn):
    w = w.copy(); d = np.zeros(3); d[axis] = sgn * DELTA
    for fr in range(2):
        R = quat2R(w[fr, EEF_QUAT])
        w[fr, OBJ["frame"]] += d
        w[fr, REL["frame"]] += R.T @ d
    return w

def shift_grip(w, k, sgn):
    w = w.copy()
    for fr in range(2): w[fr, GRIP[k]] += sgn * DELTA_G
    return w

def first_act(wins):
    ot = {"state": torch.tensor(no.normalize(wins), device=dev, dtype=torch.float32)}
    with torch.no_grad():
        an = ag.sample(act_0=torch.randn((len(wins), 16, 10), device=dev), obs=ot, use_ema=True)
    return ds.undo_transform_action(na.unnormalize(an.detach().cpu().numpy())[:, start:start+AS])[:, 0, :]

Q = np.load("scripts/jac_queries.npz")
snaps = sorted(glob.glob(os.environ["SNAP_GLOB"]),
               key=lambda p: int(re.search(r"snap_(\d+)", p).group(1)) if re.search(r"snap_(\d+)", p) else 10**9)
for ck in snaps:
    m = re.search(r"snap_(\d+)", ck)
    step = m.group(1) if m else "latest"
    ag.load(ck, load_optimizer=False); ag.eval()
    for qname in ["canon", "dep"]:
        stats = {b: [] for b in ["eef", "frame", "grip"]}
        for w in Q[qname]:
            wins, spec = [], []
            for b, fn, ncol in [("eef", shift_eef, 3), ("frame", shift_frame, 3), ("grip", shift_grip, 2)]:
                for ax in range(ncol):
                    for sg in (+1, -1): wins.append(fn(w, ax, sg))
                spec.append((b, ncol, DELTA_G if b == "grip" else DELTA))
            A = first_act(np.stack(wins))
            i = 0
            for b, ncol, dl in spec:
                Jp = np.zeros((3, ncol))
                for c in range(ncol):
                    Jp[:, c] = (A[i, :3] - A[i+1, :3]) / (2 * dl)
                    i += 2
                stats[b].append(np.linalg.norm(Jp))
        for b in stats:
            f = np.array(stats[b])
            print(f"SNAPJAC {os.environ.get('TAG','')} step={step} {qname} {b:5s}: ‖J‖_F p50/p90={np.median(f):.1f}/{np.percentile(f,90):.1f}", flush=True)
print("SWEEP-DONE")
