"""Full-task-to-insertion eval: reset to a held-out initial state (seed 21000+),
let the policy run the whole task, success = _check_frame_assembled.

Usage:
  MUJOCO_GL=egl python scripts/eval_full.py \
     --ckpt logs/full_mse_200/models/model_latest.pt \
     --dataset data/tool_hang_full2ins_200.hdf5 --loss regression --n 100
"""
import os
os.environ["MUJOCO_GL"] = "egl"
import argparse
import numpy as np
import torch
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
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--dataset", required=True)
    ap.add_argument("--loss", default="regression")
    ap.add_argument("--n", type=int, default=100)
    ap.add_argument("--start_seed", type=int, default=21000)
    ap.add_argument("--seeds_file", default="")  # .npy list of held-out seeds; overrides start_seed
    ap.add_argument("--max_chunks", type=int, default=70)
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

    env = robosuite.make("ToolHang", horizon=4000, **ENV_KWARGS)

    def ov(o):
        return np.concatenate([o[KM.get(k, k)] for k in OK]).astype(np.float32)

    succ = 0
    if args.seeds_file:
        seed_list = np.load(args.seeds_file)[:args.n]
    else:
        seed_list = [args.start_seed + i for i in range(args.n)]
    for sd in seed_list:
        np.random.seed(int(sd))
        o = env.reset()
        hist = [ov(o), ov(o)]
        asm = False
        for _ in range(args.max_chunks):
            w = np.stack(hist[-2:])[None]
            ot = {"state": torch.tensor(no.normalize(w), device=dev, dtype=torch.float32)}
            with torch.no_grad():
                an = agent.sample(act_0=torch.randn((1, H, 10), device=dev), obs=ot, use_ema=True)
            act7 = ds.undo_transform_action(na.unnormalize(an.detach().cpu().numpy())[:, start:start+AS])[0]
            for a in act7:
                o, _, _, _ = env.step(a); hist.append(ov(o))
                if env._check_frame_assembled():
                    asm = True; break
            if asm:
                break
        succ += int(asm)
    N = len(seed_list)
    print(f"FULL_SR {succ}/{N} = {100*succ/N:.1f}%")


if __name__ == "__main__":
    main()
