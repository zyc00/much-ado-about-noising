"""Re-roll DART-6k on 4 chosen warmstart seeds recording BOTH d_clean and d_own
(distance to its own DART support) per step. Deterministic -> matches escape_stats'
DART d_clean curves. Saves analysis/recovery/dart_dualruler_4.npz for the 4-panel fig."""
import os
os.environ["MUJOCO_GL"] = "egl"
import numpy as np
import torch
import h5py
import robosuite
import sys
sys.path.insert(0, "scripts"); sys.path.insert(0, ".")
from collect_tool_hang_demos import ENV_KWARGS
from hydra import initialize_config_dir, compose
from omegaconf import OmegaConf
from scipy.spatial import cKDTree
from mip.datasets.robomimic_dataset import make_dataset
from mip.agent import TrainingAgent

OK = ["object", "robot0_eef_pos", "robot0_eef_quat", "robot0_gripper_qpos"]
KM = {"object": "object-state"}
IDX = [6, 12, 21, 34]   # warmstart demo indices -> seeds 21006/21012/21021/21034
H = 16; MAX_STEPS = 500


def stack(path, ndemo, stride=1):
    h = h5py.File(path, "r"); g = "data" if "data" in h else "demos"; ks = list(h[g])[:ndemo]
    out = [np.concatenate([np.asarray(h[f"{g}/{k}/obs"][key]) for key in OK], axis=1).astype(np.float32)[::stride] for k in ks]
    h.close(); return np.concatenate(out, 0)


cfgdir = os.path.abspath("examples/configs")
with initialize_config_dir(version_base=None, config_dir=cfgdir):
    cfg = compose(config_name="main", overrides=["task=tool_hang_ph_state_delta_legacy",
        "+task.dataset_path=" + os.path.abspath("data/tool_hang_full2ins_2000.hdf5"), "network=chiunet",
        "optimization.loss_type=regression", "optimization.auto_resume=false", "log.wandb_mode=disabled"])
OmegaConf.set_struct(cfg, False); cfg.task.obs_dim = 53
dev = cfg.optimization.device; AS = cfg.task.act_steps; start = cfg.task.obs_steps - 1
ds = make_dataset(cfg.task)
ag = TrainingAgent(cfg); ag.load("logs/puredart_mse/models/model_latest.pt", load_optimizer=False); ag.eval()
nD = torch.load("analysis/recovery/norm_puredart6k.pt", weights_only=False)
no = nD["obs"]["state"]; na = nD["action"]

df = h5py.File("data/warmstart_demos.hdf5", "r"); allk = list(df["demos"].keys())
man_keys = allk[40:70]
env = robosuite.make("ToolHang", horizon=4000, **ENV_KWARGS)
def ov(o): return np.concatenate([o[KM.get(k, k)] for k in OK]).astype(np.float32)

# clean cloud (same construction as escape_stats: replay manifold demos)
man = []
for k in man_keys:
    d = df["demos/" + k]; sd = int(d.attrs["seed"]); acts = np.clip(d["actions"][:], -1, 1); s0 = d["state0"][:]
    np.random.seed(sd); env.reset(); env.sim.set_state_from_flattened(s0); env.sim.forward()
    arm = env.robots[0].composite_controller.part_controllers["right"]; arm.update(); arm.reset_goal()
    man.append(ov(env._get_observations(force_update=True)))
    for t in range(len(acts)):
        o, _, _, _ = env.step(acts[t]); man.append(ov(o))
man = np.array(man); mu, sig = man.mean(0), man.std(0) + 1e-6
tc = cKDTree((man - mu) / sig)
# DART own-support cloud (proxy: local dart_full2ins_2000 stored obs), same mu/sig
dart_raw = stack("data/tool_hang_dart_full2ins_2000.hdf5", 60)
rng = np.random.RandomState(0)
di = rng.choice(len(dart_raw), min(len(man), len(dart_raw)), replace=False)
td = cKDTree((dart_raw[di] - mu) / sig)
def dc(v): return float(tc.query((v - mu) / sig)[0])
def dd(v): return float(td.query((v - mu) / sig)[0])

def predict(hist):
    w = np.stack(hist[-2:])[None]
    ot = {"state": torch.tensor(no.normalize(w), device=dev, dtype=torch.float32)}
    with torch.no_grad():
        an = ag.sample(act_0=torch.randn((1, H, 10), device=dev), obs=ot, use_ema=True)
    return ds.undo_transform_action(na.unnormalize(an.detach().cpu().numpy())[:, start:start + AS])[0]

out = {}
for j in IDX:
    k = allk[j]; d = df["demos/" + k]; sd = int(d.attrs["seed"])
    np.random.seed(sd); env.reset()
    arm = env.robots[0].composite_controller.part_controllers["right"]; arm.update(); arm.reset_goal()
    for _ in range(10):
        env.step(np.zeros(7))
    o = env._get_observations(force_update=True); hist = [ov(o), ov(o)]
    steps = 0; asm = False; DC, DO = [], []
    while steps < MAX_STEPS and not asm:
        for a in predict(hist):
            o, _, _, _ = env.step(a); steps += 1; v = ov(o); hist.append(v)
            DC.append(dc(v)); DO.append(dd(v))
            if env._check_frame_assembled(): asm = True; break
            if steps >= MAX_STEPS: break
    out[f"dc{j}"] = np.array(DC); out[f"do{j}"] = np.array(DO); out[f"asm{j}"] = asm
    print(f"seed {sd}: asm={asm} maxd_clean={max(DC):.1f} maxd_own={max(DO):.1f}", flush=True)
df.close()
np.savez("analysis/recovery/dart_dualruler_4.npz", **{k: v for k, v in out.items()})
print("saved analysis/recovery/dart_dualruler_4.npz")
