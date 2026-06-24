"""Two-stage eval: chain two separately-trained policies.
  stage 1 (front) : pick -> handoff
  stage 2 (back)  : handoff -> insertion  (align+insertion policy)
Reset to held-out pick state, run front policy for a budget, then switch to the
back policy until _check_frame_assembled. Covers pick->insertion (NOT initial).

Usage:
  MUJOCO_GL=egl python scripts/eval_twostage.py \
     --front_ckpt logs/front_regression_2000/models/model_latest.pt \
     --front_ds   data/tool_hang_front_2000.hdf5 \
     --back_ckpt  logs/ai_regression_2000/models/model_latest.pt \
     --back_ds    data/tool_hang_aligninsert_2000.hdf5 \
     --loss regression --n 100
"""
import os
os.environ["MUJOCO_GL"] = "egl"
import argparse
import numpy as np
import torch
import h5py
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


def load(ds_path, loss, dev):
    cfgdir = os.path.abspath("examples/configs")
    with initialize_config_dir(version_base=None, config_dir=cfgdir):
        cfg = compose(config_name="main", overrides=[
            "task=tool_hang_ph_state_delta_legacy",
            f"+task.dataset_path={os.path.abspath(ds_path)}",
            "network=chiunet", f"optimization.loss_type={loss}",
            "optimization.auto_resume=false", "log.wandb_mode=disabled"])
    OmegaConf.set_struct(cfg, False); cfg.task.obs_dim = 53
    ds = make_dataset(cfg.task)
    return cfg, ds


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--front_ckpt", required=True); ap.add_argument("--front_ds", required=True)
    ap.add_argument("--back_ckpt", required=True); ap.add_argument("--back_ds", required=True)
    ap.add_argument("--loss", default="regression")
    ap.add_argument("--n", type=int, default=100)
    ap.add_argument("--eval_states", default="data/pick_eval_states.hdf5")
    ap.add_argument("--settle", type=int, default=5)
    ap.add_argument("--front_chunks", type=int, default=14)
    ap.add_argument("--back_chunks", type=int, default=30)
    args = ap.parse_args()

    dev = "cuda"
    cfgF, dsF = load(args.front_ds, args.loss, dev)
    cfgB, dsB = load(args.back_ds, args.loss, dev)
    dev = cfgF.optimization.device
    H = 16; AS = cfgF.task.act_steps; start = cfgF.task.obs_steps - 1

    aF = TrainingAgent(cfgF); aF.load(args.front_ckpt, load_optimizer=False); aF.eval()
    aB = TrainingAgent(cfgB); aB.load(args.back_ckpt, load_optimizer=False); aB.eval()
    noF = dsF.normalizer["obs"]["state"]; naF = dsF.normalizer["action"]
    noB = dsB.normalizer["obs"]["state"]; naB = dsB.normalizer["action"]

    es = h5py.File(args.eval_states, "r")
    states = es["states"][:]; seeds = es["seeds"][:]; es.close()
    N = min(args.n, len(states))
    env = robosuite.make("ToolHang", horizon=4000, **ENV_KWARGS)

    def ov(o):
        return np.concatenate([o[KM.get(k, k)] for k in OK]).astype(np.float32)

    def act_chunk(agent, no, na, hist):
        w = np.stack(hist[-2:])[None]
        ot = {"state": torch.tensor(no.normalize(w), device=dev, dtype=torch.float32)}
        with torch.no_grad():
            an = agent.sample(act_0=torch.randn((1, H, 10), device=dev), obs=ot, use_ema=True)
        return dsF.undo_transform_action(na.unnormalize(an.detach().cpu().numpy())[:, start:start+AS])[0]

    succ = 0
    for i in range(N):
        np.random.seed(int(seeds[i])); env.reset()
        sim = env.sim; sim.set_state_from_flattened(states[i]); sim.forward()
        arm = env.robots[0].composite_controller.part_controllers["right"]
        arm.update(); arm.reset_goal()
        o = env._get_observations(force_update=True)
        for _ in range(args.settle):
            o, _, _, _ = env.step(np.array([0, 0, 0, 0, 0, 0, 1.0]))
        hist = [ov(o), ov(o)]; asm = False
        # stage 1: front
        for _ in range(args.front_chunks):
            for a in act_chunk(aF, noF, naF, hist):
                o, _, _, _ = env.step(a); hist.append(ov(o))
        # stage 2: back (align+insert)
        for _ in range(args.back_chunks):
            for a in act_chunk(aB, noB, naB, hist):
                o, _, _, _ = env.step(a); hist.append(ov(o))
                if env._check_frame_assembled():
                    asm = True; break
            if asm:
                break
        succ += int(asm)
    print(f"TWOSTAGE_SR {succ}/{N} = {100*succ/N:.1f}%")


if __name__ == "__main__":
    main()
