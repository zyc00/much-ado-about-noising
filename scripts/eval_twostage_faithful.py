"""Faithful two-stage chaining (NO set_state, natural init):
  stage 1 = grasp specialist, runs from init until grasp detected (or budget)
  stage 2 = pick2ins specialist (c1->insertion), takes over until assembled.
Tests whether two separately-trained MSE specialists COMPOSE into the full task.
Both stages share the natural OSC controller (stage1 drives to grasp naturally,
stage2 trained on natural [c1:insertion] -> in-distribution handoff).

Usage:
  MUJOCO_GL=egl python scripts/eval_twostage_faithful.py \
     --grasp_ckpt logs/grasp_regression_2000/models/model_latest.pt \
     --grasp_ds   data/tool_hang_init2grasp_2000.hdf5 \
     --back_ckpt  logs/pick2ins_regression_2000/models/model_latest.pt \
     --back_ds    data/tool_hang_pick2ins_2000.hdf5 \
     --loss regression --seeds_file data/full_eval_seeds.npy --n 100
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


def load(ds_path, loss):
    cfgdir = os.path.abspath("examples/configs")
    with initialize_config_dir(version_base=None, config_dir=cfgdir):
        cfg = compose(config_name="main", overrides=[
            "task=tool_hang_ph_state_delta_legacy",
            f"+task.dataset_path={os.path.abspath(ds_path)}",
            "network=chiunet", f"optimization.loss_type={loss}",
            "optimization.auto_resume=false", "log.wandb_mode=disabled"])
    OmegaConf.set_struct(cfg, False); cfg.task.obs_dim = 53
    return cfg, make_dataset(cfg.task)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--grasp_ckpt", required=True); ap.add_argument("--grasp_ds", required=True)
    ap.add_argument("--back_ckpt", required=True); ap.add_argument("--back_ds", required=True)
    ap.add_argument("--loss", default="regression")
    ap.add_argument("--grasp_loss", default=None)  # per-stage override (cross combos)
    ap.add_argument("--back_loss", default=None)
    ap.add_argument("--demos", default="data/warmstart_demos.hdf5")  # for settled frame0 init
    ap.add_argument("--init_mode", default="reset_settle")  # state0 | reset_settle | reset
    ap.add_argument("--stage1", default="policy")  # policy | replay_c1 | replay_grasp (expert grasp control)
    ap.add_argument("--n", type=int, default=100)
    ap.add_argument("--grasp_budget", type=int, default=160)  # max steps for stage1
    ap.add_argument("--max_steps", type=int, default=700)
    args = ap.parse_args()

    dev = "cuda"
    cfgG, dsG = load(args.grasp_ds, args.grasp_loss or args.loss)
    cfgB, dsB = load(args.back_ds, args.back_loss or args.loss)
    dev = cfgG.optimization.device
    H = 16; AS = cfgG.task.act_steps; start = cfgG.task.obs_steps - 1
    aG = TrainingAgent(cfgG); aG.load(args.grasp_ckpt, load_optimizer=False); aG.eval()
    aB = TrainingAgent(cfgB); aB.load(args.back_ckpt, load_optimizer=False); aB.eval()
    noG = dsG.normalizer["obs"]["state"]; naG = dsG.normalizer["action"]
    noB = dsB.normalizer["obs"]["state"]; naB = dsB.normalizer["action"]

    import h5py
    dfh = h5py.File(args.demos, "r")
    dkeys = list(dfh["demos"].keys())[:args.n]
    env = robosuite.make("ToolHang", horizon=4000, **ENV_KWARGS)

    def ov(o):
        return np.concatenate([o[KM.get(k, k)] for k in OK]).astype(np.float32)

    def grasped():
        g = env._check_grasp(gripper=env.robots[0].gripper["right"], object_geoms=env.frame.contact_geoms)
        fz = env.sim.data.site_xpos[env.sim.model.site_name2id("frame_mount_site")][2]
        return bool(g and fz > 0.86)

    def chunk(agent, no, na, hist):
        w = np.stack(hist[-2:])[None]
        ot = {"state": torch.tensor(no.normalize(w), device=dev, dtype=torch.float32)}
        with torch.no_grad():
            an = agent.sample(act_0=torch.randn((1, H, 10), device=dev), obs=ot, use_ema=True)
        return dsG.undo_transform_action(na.unnormalize(an.detach().cpu().numpy())[:, start:start+AS])[0]

    succ = 0; n_grasp = 0
    for k in dkeys:
        d = dfh["demos/" + k]; sd = int(d.attrs["seed"]); state0 = d["state0"][:]
        acts = d["actions"][:]; c1 = int(d.attrs["c1"])
        replay_stage1 = args.stage1 in ("replay_c1", "replay_grasp")
        np.random.seed(sd); env.reset()
        if args.init_mode == "state0" or replay_stage1:
            env.sim.set_state_from_flattened(state0); env.sim.forward()
            arm = env.robots[0].composite_controller.part_controllers["right"]
            arm.update(); arm.reset_goal()
        else:  # reset (no settle) or reset_settle (+10x zero settle)
            arm = env.robots[0].composite_controller.part_controllers["right"]
            arm.update(); arm.reset_goal()
            if args.init_mode == "reset_settle":
                for _ in range(10):
                    env.step(np.zeros(7))
        o = env._get_observations(force_update=True)
        hist = [ov(o), ov(o)]; steps = 0; gdone = False; asm = False
        # stage 1: grasp
        if args.stage1 == "policy":  # grasp specialist until grasped or budget
            while steps < args.grasp_budget and not gdone:
                for a in chunk(aG, noG, naG, hist):
                    o, _, _, _ = env.step(a); steps += 1; hist.append(ov(o))
                    if grasped():
                        gdone = True; break
                    if steps >= args.grasp_budget:
                        break
        elif args.stage1 == "replay_c1":  # expert replay up to c1 (pick2ins's native start)
            for t in range(c1):
                o, _, _, _ = env.step(np.clip(acts[t], -1, 1)); steps += 1; hist.append(ov(o))
            gdone = True
        elif args.stage1 == "replay_grasp":  # expert replay until grasped() (same handoff depth as policy)
            for t in range(len(acts)):
                o, _, _, _ = env.step(np.clip(acts[t], -1, 1)); steps += 1; hist.append(ov(o))
                if grasped():
                    gdone = True; break
        n_grasp += int(gdone)
        # stage 2: pick2ins specialist until assembled
        while steps < args.max_steps and not asm:
            for a in chunk(aB, noB, naB, hist):
                o, _, _, _ = env.step(a); steps += 1; hist.append(ov(o))
                if env._check_frame_assembled():
                    asm = True; break
                if steps >= args.max_steps:
                    break
        succ += int(asm)
    N = len(dkeys)
    print(f"TWOSTAGE_FAITHFUL grasp={n_grasp}/{N} assembled={succ}/{N} = {100*succ/N:.1f}%")


if __name__ == "__main__":
    main()
