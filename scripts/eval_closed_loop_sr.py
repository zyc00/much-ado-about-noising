"""Closed-loop success-rate eval for a trained BC policy on scripted ToolHang.

For each seed: reset (seed-deterministic init), settle with exact-zero actions
(so the policy starts from the settled state, matching settled/markovian data),
then run the policy closed-loop until success or max_steps. Reports SR.

Usage:
  MUJOCO_GL=egl python scripts/eval_closed_loop_sr.py \
     --ckpt logs/reg_mark/models/model_latest.pt \
     --dataset data/tool_hang_markovian_200.hdf5 \
     --n_seeds 30 --settle_steps 10 --max_steps 1200
"""
import os
os.environ["MUJOCO_GL"] = "egl"
import argparse
import numpy as np
import torch
import robosuite
import sys
sys.path.insert(0, "scripts")
sys.path.insert(0, ".")
from collect_tool_hang_demos import ENV_KWARGS
from hydra import initialize_config_dir, compose
from omegaconf import OmegaConf
from mip.datasets.robomimic_dataset import make_dataset
from mip.agent import TrainingAgent

OBS_KEYS = ["object", "robot0_eef_pos", "robot0_eef_quat", "robot0_gripper_qpos"]
KM = {"object": "object-state"}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--dataset", required=True)
    ap.add_argument("--network", default="mlp")
    ap.add_argument("--loss_type", default="regression")
    ap.add_argument("--n_seeds", type=int, default=30)
    ap.add_argument("--settle_steps", type=int, default=10)
    ap.add_argument("--max_steps", type=int, default=1200)
    ap.add_argument("--num_sample_steps", type=int, default=1)
    args = ap.parse_args()

    cfgdir = os.path.abspath("examples/configs")
    with initialize_config_dir(version_base=None, config_dir=cfgdir):
        cfg = compose(config_name="main", overrides=[
            "task=tool_hang_ph_state_delta_legacy",
            f"+task.dataset_path={os.path.abspath(args.dataset)}",
            f"network={args.network}",
            f"optimization.loss_type={args.loss_type}",
            "optimization.batch_size=128",
            "optimization.auto_resume=false",
            "log.wandb_mode=disabled",
        ])
    OmegaConf.set_struct(cfg, False)
    cfg.task.obs_dim = 53
    dev = cfg.optimization.device
    obs_steps = cfg.task.obs_steps
    horizon = cfg.task.horizon
    act_steps = cfg.task.act_steps
    start = obs_steps - 1

    ds = make_dataset(cfg.task)
    agent = TrainingAgent(cfg)
    agent.load(args.ckpt, load_optimizer=False)
    agent.eval()
    no = ds.normalizer["obs"]["state"]
    na = ds.normalizer["action"]

    env_kw = dict(ENV_KWARGS)
    env = robosuite.make("ToolHang", horizon=4000, **env_kw)

    def ov(o):
        return np.concatenate([o[KM.get(k, k)] for k in OBS_KEYS]).astype(np.float32)

    succ = 0
    results = []
    for seed in range(args.n_seeds):
        np.random.seed(seed)
        obs = env.reset()
        for _ in range(args.settle_steps):
            obs, _, _, _ = env.step(np.zeros(7))
        hist = [ov(obs), ov(obs)]
        done_succ = False
        t = 0
        while t < args.max_steps:
            w = np.stack(hist[-obs_steps:])[None]
            ot = {"state": torch.tensor(no.normalize(w), device=dev, dtype=torch.float32)}
            with torch.no_grad():
                an = agent.sample(act_0=torch.randn((1, horizon, 10), device=dev),
                                  obs=ot, num_steps=args.num_sample_steps, use_ema=True)
            act = na.unnormalize(an.detach().cpu().numpy())[:, start:start + act_steps, :]
            act7 = ds.undo_transform_action(act)[0]
            for a in act7:
                obs, _, _, _ = env.step(a)
                hist.append(ov(obs))
                t += 1
                if env._check_success():
                    done_succ = True
                    break
            if done_succ:
                break
        succ += int(done_succ)
        results.append((seed, done_succ, t))
        print(f"seed {seed}: success={done_succ} steps={t}  (running SR {succ}/{seed+1})")
    print(f"\nSR: {succ}/{args.n_seeds} = {100*succ/args.n_seeds:.1f}%")


if __name__ == "__main__":
    main()
