"""Closed-loop rollout of a trained BC policy from demo_0's EXACT init.

Resets the collection env (robosuite.make + ENV_KWARGS) via seed-0 reset, which
reproduces demo_0's recorded init exactly (verified: state diff = 0, open-loop
replay succeeds). Then runs the policy closed-loop, replacing the recorded
actions with policy predictions each step.

Usage:
  MUJOCO_GL=egl python scripts/closed_loop_demo0.py \
     --ckpt logs/overfit_1demo/models/model_latest.pt \
     --dataset data/tool_hang_scripted_1demo.hdf5 \
     --seed 0
"""
import os
os.environ["MUJOCO_GL"] = "egl"
import argparse
import numpy as np
import torch
import imageio
import robosuite
import robosuite.utils.transform_utils as T
from hydra import initialize, compose
from omegaconf import OmegaConf

import sys
sys.path.insert(0, "scripts")
sys.path.insert(0, ".")
from collect_tool_hang_demos import ENV_KWARGS
from mip.datasets.robomimic_dataset import make_dataset
from mip.agent import TrainingAgent

OBS_KEYS = ["object", "robot0_eef_pos", "robot0_eef_quat", "robot0_gripper_qpos"]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--dataset", default="data/tool_hang_scripted_1demo.hdf5")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--video", default="logs/closed_loop_demo0.mp4")
    ap.add_argument("--max_steps", type=int, default=900)
    ap.add_argument("--settle_steps", type=int, default=0)
    args = ap.parse_args()

    # --- build config like train_robomimic ---
    with initialize(version_base=None, config_path="../examples/configs"):
        cfg = compose(config_name="main", overrides=[
            "task=tool_hang_ph_state_delta_legacy",
            f"+task.dataset_path={os.path.abspath(args.dataset)}",
            "network=mlp",
            "optimization.loss_type=regression",
            "optimization.norm_type=l2",
            "optimization.auto_resume=false",
            "log.wandb_mode=disabled",
        ])
    config = cfg
    OmegaConf.set_struct(config, False)
    config.task.obs_dim = 53
    device = config.optimization.device
    obs_steps = config.task.obs_steps          # 2
    horizon = config.task.horizon              # 10 for mlp
    act_steps = config.task.act_steps          # 8
    act_dim = config.task.act_dim              # 10

    # --- dataset (for normalizer) + agent ---
    dataset = make_dataset(config.task)
    agent = TrainingAgent(config)
    agent.load(args.ckpt, load_optimizer=False)
    agent.eval()
    print(f"loaded ckpt {args.ckpt}; obs_steps={obs_steps} horizon={horizon} act_steps={act_steps}")

    # --- env: collection config + offscreen for video + reproducible init ---
    env_kw = dict(ENV_KWARGS)
    env_kw["has_offscreen_renderer"] = True
    env_kw["camera_names"] = "sideview"
    env_kw["camera_heights"] = 256
    env_kw["camera_widths"] = 256
    env = robosuite.make("ToolHang", horizon=4000, **env_kw)
    sim = env.sim

    np.random.seed(args.seed)
    obs = env.reset()
    frames = [(sim.forward(), sim.render(256, 256, camera_name="sideview")[::-1])[1]]

    # live robosuite obs uses "object-state" for the "object" key
    key_map = {"object": "object-state"}

    def obs_vec(o):
        return np.concatenate(
            [o[key_map.get(k, k)] for k in OBS_KEYS]
        ).astype(np.float32)  # (53,)

    # settle with EXACT zero actions (deterministic) before starting the policy,
    # so the policy starts from the settled state (matching settled-demo training)
    for _ in range(args.settle_steps):
        obs, _, _, _ = env.step(np.zeros(7))
        frames.append((sim.forward(), sim.render(256, 256, camera_name="sideview")[::-1])[1])

    # obs history buffer (obs_steps frames)
    hist = [obs_vec(obs)] * obs_steps

    norm_obs = dataset.normalizer["obs"]["state"]
    norm_act = dataset.normalizer["action"]

    max_steps = args.max_steps
    t = 0
    success = False
    while t < max_steps:
        # build (1, obs_steps, 53)
        obs_np = np.stack(hist[-obs_steps:], axis=0)[None]  # (1, obs_steps, 53)
        obs_n = norm_obs.normalize(obs_np)
        obs_t = {"state": torch.tensor(obs_n, device=device, dtype=torch.float32)}
        act_0 = torch.randn((1, horizon, act_dim), device=device)
        with torch.no_grad():
            act_normed = agent.sample(act_0=act_0, obs=obs_t, num_steps=1, use_ema=True)
        act = norm_act.unnormalize(act_normed.detach().cpu().numpy())  # (1, horizon, 10)
        start = obs_steps - 1
        act = act[:, start:start + act_steps, :]                       # (1, act_steps, 10)
        act7 = dataset.undo_transform_action(act)[0]                   # (act_steps, 7)

        for a in act7:
            obs, _, _, _ = env.step(a)
            frames.append((sim.forward(), sim.render(256, 256, camera_name="sideview")[::-1])[1])
            hist.append(obs_vec(obs))
            t += 1
            if env._check_success():
                success = True
                break
        if success:
            break

    print(f"SUCCESS: {success}  (steps={t})")
    imageio.mimsave(args.video, frames, fps=30)
    print(f"saved {args.video} ({len(frames)} frames)")


if __name__ == "__main__":
    main()
