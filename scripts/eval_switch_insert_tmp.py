"""Closed-loop eval of the NO-WEIGHT-SHARING composition: frozen MSE-2k proposes a chunk,
the separately-trained denoiser (denoise_only_2k) refines it, execute. Held-out seeds."""
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
from mip.samplers import regression_sampler

OK = ["object", "robot0_eef_pos", "robot0_eef_quat", "robot0_gripper_qpos"]
KM = {"object": "object-state"}

def load(ckpt, loss):
    cfgdir = os.path.abspath("examples/configs")
    with initialize_config_dir(version_base=None, config_dir=cfgdir):
        cfg = compose(config_name="main", overrides=["task=tool_hang_ph_state_delta_legacy",
            "+task.dataset_path=" + os.path.abspath("data/tool_hang_full2ins_2000.hdf5"),
            "network=chiunet", f"optimization.loss_type={loss}",
            "optimization.auto_resume=false", "log.wandb_mode=disabled"])
    OmegaConf.set_struct(cfg, False); cfg.task.obs_dim = 53
    ds = make_dataset(cfg.task)
    ag = TrainingAgent(cfg); ag.load(ckpt, load_optimizer=False); ag.eval()
    return cfg, ds, ag

ap = argparse.ArgumentParser()
ap.add_argument("--seed_lo", type=int, default=21000); ap.add_argument("--seed_hi", type=int, default=21100)
ap.add_argument("--H", type=int, default=16); ap.add_argument("--settle", type=int, default=10)
ap.add_argument("--max_steps", type=int, default=700)
args = ap.parse_args()
cfg, ds, mse = load("logs/full_regression_2000/models/model_latest.pt", "regression")
_, _, den = load("logs/denoise_only_2k/models/model_latest.pt", "denoise_only")
dev = cfg.optimization.device; AS = cfg.task.act_steps; start = cfg.task.obs_steps - 1
no = ds.normalizer["obs"]["state"]; na = ds.normalizer["action"]; optim = cfg.optimization
TAU = float(optim.t_two_step)
env = robosuite.make("ToolHang", horizon=4000, **ENV_KWARGS)
def ov(o): return np.concatenate([o[KM.get(k, k)] for k in OK]).astype(np.float32)
def chunk(hist):
    x = torch.tensor(no.normalize(np.stack(hist[-2:])[None]), device=dev, dtype=torch.float32)
    with torch.no_grad():
        with mse._inference_mode():
            an = regression_sampler(optim, mse.flow_map_ema, mse.encoder_ema, torch.zeros((1, args.H, 10), device=dev), {"state": x})
        with den._inference_mode():
            emb = den.encoder_ema({"state": x}, None)
            t = torch.full((1,), TAU, device=dev)
            an2 = den.flow_map_ema.get_velocity(t, an, emb)
    return ds.undo_transform_action(na.unnormalize(an2.detach().cpu().numpy())[:, start:start + AS])[0]
succ = 0; N = 0
for sd in range(args.seed_lo, args.seed_hi):
    np.random.seed(sd); env.reset()
    arm = env.robots[0].composite_controller.part_controllers["right"]; arm.update(); arm.reset_goal()
    for _ in range(args.settle):
        env.step(np.zeros(7))
    o = env._get_observations(force_update=True); hist = [ov(o), ov(o)]
    steps = 0; asm = False
    while steps < args.max_steps and not asm:
        for a in chunk(hist):
            o, _, _, _ = env.step(a); steps += 1; hist.append(ov(o))
            if env._check_frame_assembled(): asm = True; break
            if steps >= args.max_steps: break
    succ += int(asm); N += 1
print(f"COMPOSE frozen-MSE->standalone-denoiser seeds[{args.seed_lo},{args.seed_hi}) = {succ}/{N} = {100*succ/N:.1f}%")
