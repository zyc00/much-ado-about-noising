"""Render a policy rollout that starts from a saved state (set_state + ctrl sync
+ settle), e.g. pick states. Records frames to mp4.

Usage:
  MUJOCO_GL=egl python scripts/render_from_state.py \
     --ckpt logs/pick2ins_regression_200/models/model_latest.pt \
     --dataset data/tool_hang_pick2ins_200.hdf5 --loss regression \
     --eval_states data/pick_eval_states.hdf5 --idx 0 1 2 \
     --out logs/roll_pick2ins_reg200 --max_chunks 30
"""
import os
os.environ["MUJOCO_GL"] = "egl"
import argparse
import numpy as np
import torch
import h5py
import imageio
import robosuite
import sys
sys.path.insert(0, "scripts"); sys.path.insert(0, ".")
from collect_tool_hang_demos import ENV_KWARGS
from hydra import initialize_config_dir, compose
from omegaconf import OmegaConf
from mip.datasets.robomimic_dataset import make_dataset
from mip.agent import TrainingAgent

OK = ["object", "robot0_eef_pos", "robot0_eef_quat", "robot0_gripper_qpos"]
KM = {"object": "object-state"}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", required=True); ap.add_argument("--dataset", required=True)
    ap.add_argument("--loss", default="regression")
    ap.add_argument("--eval_states", default="data/pick_eval_states.hdf5")
    ap.add_argument("--idx", nargs="+", type=int, default=[0, 1, 2])
    ap.add_argument("--settle", type=int, default=5)
    ap.add_argument("--max_chunks", type=int, default=30)
    ap.add_argument("--out", default="logs/roll"); ap.add_argument("--cam", default="sideview")
    ap.add_argument("--res", type=int, default=512)
    args = ap.parse_args()

    cfgdir = os.path.abspath("examples/configs")
    with initialize_config_dir(version_base=None, config_dir=cfgdir):
        cfg = compose(config_name="main", overrides=[
            "task=tool_hang_ph_state_delta_legacy",
            f"+task.dataset_path={os.path.abspath(args.dataset)}",
            "network=chiunet", f"optimization.loss_type={args.loss}",
            "optimization.auto_resume=false", "log.wandb_mode=disabled"])
    OmegaConf.set_struct(cfg, False); cfg.task.obs_dim = 53
    dev = cfg.optimization.device
    H = 16; AS = cfg.task.act_steps; start = cfg.task.obs_steps - 1
    ds = make_dataset(cfg.task)
    agent = TrainingAgent(cfg); agent.load(args.ckpt, load_optimizer=False); agent.eval()
    no = ds.normalizer["obs"]["state"]; na = ds.normalizer["action"]

    es = h5py.File(args.eval_states, "r"); states = es["states"][:]; seeds = es["seeds"][:]; es.close()
    kw = dict(ENV_KWARGS); kw["has_offscreen_renderer"] = True; kw["use_camera_obs"] = False
    kw["camera_names"] = args.cam; kw["camera_heights"] = args.res; kw["camera_widths"] = args.res
    env = robosuite.make("ToolHang", horizon=4000, **kw)

    def ov(o):
        return np.concatenate([o[KM.get(k, k)] for k in OK]).astype(np.float32)

    def fr():
        return env.sim.render(camera_name=args.cam, width=args.res, height=args.res)[::-1]

    for i in args.idx:
        np.random.seed(int(seeds[i])); env.reset()
        env.sim.set_state_from_flattened(states[i]); env.sim.forward()
        arm = env.robots[0].composite_controller.part_controllers["right"]; arm.update(); arm.reset_goal()
        o = env._get_observations(force_update=True)
        for _ in range(args.settle):
            o, _, _, _ = env.step(np.array([0, 0, 0, 0, 0, 0, 1.0]))
        hist = [ov(o), ov(o)]; frames = [fr()]; asm = False
        for _ in range(args.max_chunks):
            w = np.stack(hist[-2:])[None]
            ot = {"state": torch.tensor(no.normalize(w), device=dev, dtype=torch.float32)}
            with torch.no_grad():
                an = agent.sample(act_0=torch.randn((1, H, 10), device=dev), obs=ot, use_ema=True)
            act7 = ds.undo_transform_action(na.unnormalize(an.detach().cpu().numpy())[:, start:start+AS])[0]
            for a in act7:
                o, _, _, _ = env.step(a); hist.append(ov(o)); frames.append(fr())
                if env._check_frame_assembled():
                    asm = True; break
            if asm:
                break
        path = f"{args.out}_idx{i}_{'OK' if asm else 'FAIL'}.mp4"
        imageio.mimsave(path, frames, fps=20, macro_block_size=1)
        print(f"SAVED {path}  ({len(frames)} frames, assembled={asm})")


if __name__ == "__main__":
    main()
