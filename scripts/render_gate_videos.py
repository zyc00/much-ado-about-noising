"""Gate-approach videos, birdview camera: 2 exact demo replays (sim-state playback) +
MSE-fail/MIP-success matched seed 31003 re-rollouts."""
import os
os.environ["MUJOCO_GL"] = "egl"
import numpy as np, torch, h5py, sys
sys.path.insert(0, "scripts"); sys.path.insert(0, ".")
import robosuite, imageio
from collect_tool_hang_demos import ENV_KWARGS
from hydra import initialize_config_dir, compose
from omegaconf import OmegaConf
from mip.datasets.robomimic_dataset import make_dataset
from mip.agent import TrainingAgent

DSP = "/home/jigu/.cache/huggingface/hub/datasets--ChaoyiPan--mip-dataset/blobs/c5cb01b119a109a3d4842ca515906e86f1f35a8ee1a333d22b3503c1a513a8f4"
OK = ["object", "robot0_eef_pos", "robot0_eef_quat", "robot0_gripper_qpos"]
KM = {"object": "object-state"}
OLD = "/home/jigu/projects/much-ado-about-noising-old/checkpoints"
OUT = "analysis/traj_vis/videos"
CAM = "birdview"
EK = dict(ENV_KWARGS); EK["has_offscreen_renderer"] = True
env = robosuite.make("ToolHang", horizon=4000, **EK)

# --- demo replays via sim states (gate segment: c1-20 .. r1+30)
h = h5py.File(DSP, "r")
for di in [0, 3]:
    d = h[f"data/demo_{di}"]
    a = np.clip(np.asarray(d["actions"]), -1, 1); g = a[:, 6]; T = len(a)
    cl = [t for t in range(1, T) if g[t-1] < 0 and g[t] >= 0]
    op = [t for t in range(1, T) if g[t-1] >= 0 and g[t] < 0]
    c1 = cl[0]; r1 = next((t for t in op if t > c1), T - 1)
    states = np.asarray(d["states"])
    env.reset()
    frames = []
    for t in range(max(0, c1 - 20), min(r1 + 30, T)):
        env.sim.set_state_from_flattened(states[t]); env.sim.forward()
        frames.append(env.sim.render(height=448, width=512, camera_name=CAM)[::-1])
    fp = f"{OUT}/demo{di}_gate_{CAM}.mp4"
    imageio.mimsave(fp, frames, fps=25, macro_block_size=1)
    print(f"VID demo{di}: {fp} ({len(frames)} frames)", flush=True)
h.close()

# --- policy rollouts, seed 31003, birdview
def load(loss):
    with initialize_config_dir(version_base=None, config_dir=os.path.abspath("examples/configs")):
        cfg = compose(config_name="main", overrides=["task=tool_hang_ph_state_delta_legacy",
            "+task.dataset_path=" + DSP, "network=chiunet",
            f"optimization.loss_type={loss}", "optimization.auto_resume=false", "log.wandb_mode=disabled"])
    OmegaConf.set_struct(cfg, False); cfg.task.obs_dim = 53
    ds = make_dataset(cfg.task)
    return cfg, ds, TrainingAgent(cfg)
for name, loss, ck in [("hMSE_s5", "regression", f"{OLD}/tool_hang_ph_state_regression_chiunet_256_seed5_success52.pt"),
                       ("hMIP_s5", "mip", f"{OLD}/tool_hang_ph_state_mip_chiunet_256_seed5_success82.pt")]:
    cfg, ds, ag = load(loss)
    ag.load(ck, load_optimizer=False); ag.eval()
    dev = cfg.optimization.device; AS = cfg.task.act_steps; start = cfg.task.obs_steps - 1
    no = ds.normalizer["obs"]["state"]; na = ds.normalizer["action"]
    def ov(o):
        return np.concatenate([np.asarray(o[KM.get(k, k)]) for k in OK]).astype(np.float32)
    def chunk(hist):
        w = np.stack(hist[-2:])[None]
        ot = {"state": torch.tensor(no.normalize(w), device=dev, dtype=torch.float32)}
        with torch.no_grad():
            an = ag.sample(act_0=torch.randn((1, 16, 10), device=dev), obs=ot, use_ema=True)
        return ds.undo_transform_action(na.unnormalize(an.detach().cpu().numpy())[:, start:start + AS])[0]
    sd = 31003
    np.random.seed(sd); torch.manual_seed(sd); env.reset()
    arm = env.robots[0].composite_controller.part_controllers["right"]; arm.update(); arm.reset_goal()
    for _ in range(10):
        env.step(np.zeros(7))
    o = env._get_observations(force_update=True); hist = [ov(o), ov(o)]
    frames, steps, succ = [], 0, False
    while steps < 800 and not succ:
        for a in chunk(hist):
            o, _, _, _ = env.step(a); steps += 1; hist.append(ov(o))
            if steps % 2 == 0:
                frames.append(env.sim.render(height=448, width=512, camera_name=CAM)[::-1])
            if env._check_success(): succ = True; break
            if steps >= 800: break
    fp = f"{OUT}/{name}_seed{sd}_{CAM}_{'success' if succ else 'timeout'}.mp4"
    imageio.mimsave(fp, frames, fps=30, macro_block_size=1)
    print(f"VID {name}: {fp} steps={steps} succ={succ}", flush=True)
print("VID-DONE", flush=True)
