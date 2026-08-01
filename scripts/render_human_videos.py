"""Render mp4 videos of hMSE_s5 / hMIP_s5 human-task rollouts at the SAME seeds as the
analyzed dumps (31000+), standalone robosuite env with offscreen renderer."""
import os
os.environ["MUJOCO_GL"] = "egl"
import numpy as np, torch, sys
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
os.makedirs(OUT, exist_ok=True)
SEEDS = [int(s) for s in os.environ.get("SEEDS", "31000,31001,31002,31003,31004,31005").split(",")]
CAM = os.environ.get("CAM", "frontview")

def load(loss):
    with initialize_config_dir(version_base=None, config_dir=os.path.abspath("examples/configs")):
        cfg = compose(config_name="main", overrides=["task=tool_hang_ph_state_delta_legacy",
            "+task.dataset_path=" + DSP, "network=chiunet",
            f"optimization.loss_type={loss}", "optimization.auto_resume=false", "log.wandb_mode=disabled"])
    OmegaConf.set_struct(cfg, False); cfg.task.obs_dim = 53
    ds = make_dataset(cfg.task)
    return cfg, ds, TrainingAgent(cfg)

MODELS = [("hMSE_s5", "regression", f"{OLD}/tool_hang_ph_state_regression_chiunet_256_seed5_success52.pt"),
          ("hMIP_s5", "mip", f"{OLD}/tool_hang_ph_state_mip_chiunet_256_seed5_success82.pt")]
EK = dict(ENV_KWARGS); EK["has_offscreen_renderer"] = True
for name, loss, ck in MODELS:
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
    env = robosuite.make("ToolHang", horizon=4000, **EK)
    for sd in SEEDS:
        np.random.seed(sd); torch.manual_seed(sd); env.reset()
        arm = env.robots[0].composite_controller.part_controllers["right"]; arm.update(); arm.reset_goal()
        for _ in range(10):
            env.step(np.zeros(7))
        o = env._get_observations(force_update=True); hist = [ov(o), ov(o)]
        frames = []
        steps, succ, asm = 0, False, False
        while steps < 800 and not succ:
            for a in chunk(hist):
                o, _, _, _ = env.step(a); steps += 1; hist.append(ov(o))
                if steps % 2 == 0:
                    frames.append(env.sim.render(height=384, width=512, camera_name=CAM)[::-1])
                if env._check_frame_assembled(): asm = True
                if env._check_success(): succ = True; break
                if steps >= 800: break
        tag = "success" if succ else ("asm_timeout" if asm else "timeout")
        fp = f"{OUT}/{name}_seed{sd}_{tag}.mp4"
        imageio.mimsave(fp, frames, fps=30, macro_block_size=1)
        print(f"VID {name} seed={sd}: {tag} steps={steps} -> {fp}", flush=True)
print("VID-DONE")
