"""CAUSAL TEST: from identical pre-gate states (demo sim-state handover, ~2-6cm out), does
(a) a coherent straight-in servo (the 'radialized' direction executed perfectly) succeed, or
(b) only the demonstrated (curved) continuation? Also (c) hMSE policy from the same states.
If (a) succeeds at high rate -> radialization-direction is NOT the failure cause (hypothesis
dead); if (a) fails and (b) succeeds -> causal closure for the path-structure claim.
"""
import os
os.environ["MUJOCO_GL"] = "egl"
import numpy as np, torch, h5py, sys
sys.path.insert(0, "scripts"); sys.path.insert(0, ".")
import robosuite
from collect_tool_hang_demos import ENV_KWARGS
from hydra import initialize_config_dir, compose
from omegaconf import OmegaConf
from mip.datasets.robomimic_dataset import make_dataset
from mip.agent import TrainingAgent

DSP = "/home/jigu/.cache/huggingface/hub/datasets--ChaoyiPan--mip-dataset/blobs/c5cb01b119a109a3d4842ca515906e86f1f35a8ee1a333d22b3503c1a513a8f4"
OK = ["object", "robot0_eef_pos", "robot0_eef_quat", "robot0_gripper_qpos"]
KM = {"object": "object-state"}
OLD = "/home/jigu/projects/much-ado-about-noising-old/checkpoints"
BP = slice(7, 10); FP = slice(21, 24)

def load(loss):
    with initialize_config_dir(version_base=None, config_dir=os.path.abspath("examples/configs")):
        cfg = compose(config_name="main", overrides=["task=tool_hang_ph_state_delta_legacy",
            "+task.dataset_path=" + DSP, "network=chiunet",
            f"optimization.loss_type={loss}", "optimization.auto_resume=false", "log.wandb_mode=disabled"])
    OmegaConf.set_struct(cfg, False); cfg.task.obs_dim = 53
    ds = make_dataset(cfg.task)
    return cfg, ds, TrainingAgent(cfg)

cfg, ds, ag = load("regression")
ag.load(f"{OLD}/tool_hang_ph_state_regression_chiunet_256_seed5_success52.pt", load_optimizer=False); ag.eval()
no = ds.normalizer["obs"]["state"]; na = ds.normalizer["action"]; dev = cfg.optimization.device
AS = cfg.task.act_steps; start = cfg.task.obs_steps - 1

h = h5py.File(DSP, "r")
keys = sorted(h["data"].keys(), key=lambda k: int(k.split("_")[-1]))
# gate offset
offs = []
metas = []
for k in keys[:40]:
    d = h[f"data/{k}"]
    o = d["obs"]
    ov = np.concatenate([np.asarray(o[q]) for q in OK], axis=1).astype(np.float32)
    a = np.clip(np.asarray(d["actions"]), -1, 1); g = a[:, 6]; T = len(a)
    cl = [t for t in range(1, T) if g[t-1] < 0 and g[t] >= 0]
    op = [t for t in range(1, T) if g[t-1] >= 0 and g[t] < 0]
    if not cl: continue
    c1 = cl[0]
    r1 = next((t for t in op if t > c1), None)
    if not r1: continue
    offs.append(ov[r1-1, FP] - ov[r1-1, BP])
    metas.append((k, np.asarray(d["states"]), a, ov, c1, r1))
off = np.median(np.stack(offs), 0)
h.close()

EK = dict(ENV_KWARGS)
env = robosuite.make("ToolHang", horizon=4000, **EK)
def ov_env(o):
    return np.concatenate([np.asarray(o[KM.get(k, k)]) for k in OK]).astype(np.float32)

results = {"replay": [], "straight": [], "mse": []}
dists = []
for k, states, acts, ov_d, c1, r1 in metas[:20]:
    th = r1 - 45
    if th <= c1 + 10: continue
    d0 = ov_d[th, FP] - (ov_d[th, BP] + off)
    dl0 = np.linalg.norm(d0[:2])
    if not (0.015 <= dl0 <= 0.07): continue
    dists.append(dl0)
    for mode in ["replay", "straight", "mse"]:
        env.reset()
        env.sim.set_state_from_flattened(states[th]); env.sim.forward()
        arm = env.robots[0].composite_controller.part_controllers["right"]; arm.update(); arm.reset_goal()
        o = env._get_observations(force_update=True)
        hist = [ov_env(o), ov_env(o)]
        ok = False
        if mode == "replay":
            for t in range(th, min(r1 + 10, len(acts))):
                env.step(acts[t])
                if env._check_frame_assembled(): ok = True; break
        elif mode == "straight":
            for t in range(220):
                s = hist[-1]
                gate = s[BP] + off
                delta = gate - s[FP]
                a = np.zeros(7)
                a[:3] = np.clip(delta * 8.0, -0.35, 0.35)
                a[6] = 1.0
                o_, _, _, _ = env.step(a); hist.append(ov_env(o_))
                if env._check_frame_assembled(): ok = True; break
        else:
            enc = None
            steps = 0
            while steps < 220 and not ok:
                w = np.stack(hist[-2:])[None]
                ot = {"state": torch.tensor(no.normalize(w), device=dev, dtype=torch.float32)}
                with torch.no_grad():
                    an = ag.sample(act_0=torch.randn((1, 16, 10), device=dev), obs=ot, use_ema=True)
                chunk = ds.undo_transform_action(na.unnormalize(an.detach().cpu().numpy())[:, start:start + AS])[0]
                for a in chunk:
                    env.step(a); steps += 1
                    o_ = env._get_observations(force_update=True)
                    hist.append(ov_env(o_))
                    if env._check_frame_assembled(): ok = True; break
                    if steps >= 220: break
        results[mode].append(int(ok))
n = len(results["replay"])
print(f"CAUSAL n={n} handover states, lateral dist p50={np.median(dists)*1000:.0f}mm", flush=True)
for mode in ["replay", "straight", "mse"]:
    print(f"CAUSAL {mode}: assembled {sum(results[mode])}/{n}", flush=True)
print("CAUSAL-DONE")
