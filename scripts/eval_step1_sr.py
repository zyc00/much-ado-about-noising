"""Decisive test: does MIP's SR come from the 2-step ITERATIVE COMPUTE at inference, or
from the TRAINED FUNCTION? Eval the MIP model with mip_step1_only (1 step, no iteration)
vs mip_sampler (2 steps), faithful eval_seedset rollout, same seeds.
  step1 ~ full  -> iterative compute is NOT the lever (it's the trained function)
  step1 ~ MSE   -> the 2nd step IS the lever"""
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
from mip.samplers import mip_step1_only_sampler, mip_sampler

OK = ["object", "robot0_eef_pos", "robot0_eef_quat", "robot0_gripper_qpos"]
KM = {"object": "object-state"}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", default="logs/full_mip_2000/models/model_latest.pt")
    ap.add_argument("--dataset", default="data/tool_hang_full2ins_2000.hdf5")
    ap.add_argument("--seed_lo", type=int, default=0); ap.add_argument("--seed_hi", type=int, default=60)
    ap.add_argument("--H", type=int, default=16); ap.add_argument("--settle", type=int, default=10)
    ap.add_argument("--max_steps", type=int, default=700)
    args = ap.parse_args()
    cfgdir = os.path.abspath("examples/configs")
    with initialize_config_dir(version_base=None, config_dir=cfgdir):
        cfg = compose(config_name="main", overrides=[
            "task=tool_hang_ph_state_delta_legacy", f"+task.dataset_path={os.path.abspath(args.dataset)}",
            "network=chiunet", "optimization.loss_type=mip",
            "optimization.auto_resume=false", "log.wandb_mode=disabled"])
    OmegaConf.set_struct(cfg, False); cfg.task.obs_dim = 53
    ds = make_dataset(cfg.task)
    ag = TrainingAgent(cfg); ag.load(args.ckpt, load_optimizer=False); ag.eval()
    dev = cfg.optimization.device; AS = cfg.task.act_steps; start = cfg.task.obs_steps - 1
    no = ds.normalizer["obs"]["state"]; na = ds.normalizer["action"]
    optim = cfg.optimization
    env = robosuite.make("ToolHang", horizon=4000, **ENV_KWARGS)

    def ov(o): return np.concatenate([o[KM.get(k, k)] for k in OK]).astype(np.float32)
    def chunk(hist, sampler):
        w = np.stack(hist[-2:])[None]
        ot = {"state": torch.tensor(no.normalize(w), device=dev, dtype=torch.float32)}
        a0 = torch.zeros((1, args.H, 10), device=dev)
        with torch.no_grad():
            with ag._inference_mode():
                an = sampler(optim, ag.flow_map_ema, ag.encoder_ema, a0, ot)
        return ds.undo_transform_action(na.unnormalize(an.detach().cpu().numpy())[:, start:start + AS])[0]

    def run(sampler, tag):
        succ = 0; N = 0
        for sd in range(args.seed_lo, args.seed_hi):
            np.random.seed(sd); env.reset()
            arm = env.robots[0].composite_controller.part_controllers["right"]; arm.update(); arm.reset_goal()
            for _ in range(args.settle):
                env.step(np.zeros(7))
            o = env._get_observations(force_update=True); hist = [ov(o), ov(o)]
            steps = 0; asm = False
            while steps < args.max_steps and not asm:
                for a in chunk(hist, sampler):
                    o, _, _, _ = env.step(a); steps += 1; hist.append(ov(o))
                    if env._check_frame_assembled(): asm = True; break
                    if steps >= args.max_steps: break
            succ += int(asm); N += 1
        print(f"  {tag:14} seeds[{args.seed_lo},{args.seed_hi}) = {succ}/{N} = {100*succ/N:.1f}%", flush=True)
        return succ

    print(f"MIP model, same seeds:  [ref pure MSE-2k ~68%]")
    run(mip_step1_only_sampler, "MIP-step1(1x)")
    run(mip_sampler, "MIP-full(2x)")


if __name__ == "__main__":
    main()
