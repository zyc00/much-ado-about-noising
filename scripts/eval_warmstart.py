"""Warm-start eval (faithful, no mid-traj set_state):
  set_state(demo frame0) -> sync controller -> replay demo actions up to depth K
  -> policy takes over -> run to horizon -> success = _check_frame_assembled.
K is chosen per-demo from a phase name so the policy is handed a NATURALLY-reached
state (matching the OSC controller ref of sliced [K:insertion] training data).

--warm_to: 0 | c1 | align_done | <int>   (frame depth to replay before policy)
           "all" replays everything with NO policy (mechanism self-check: ~100%).

Usage:
  MUJOCO_GL=egl python scripts/eval_warmstart.py \
     --ckpt logs/chi_mip_clean2000/models/model_best.pt \
     --dataset data/tool_hang_clean_2000.hdf5 --loss mip \
     --demos data/warmstart_demos.hdf5 --warm_to align_done --n 100
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


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--dataset", required=True)
    ap.add_argument("--loss", default="regression")
    ap.add_argument("--demos", default="data/warmstart_demos.hdf5")
    ap.add_argument("--warm_to", default="align_done")  # 0|c1|align_done|mid_reach|mid_align|<int>|all
    ap.add_argument("--success", default="assembled")  # assembled | grasp
    ap.add_argument("--init_mode", default="state0")  # state0 | reset | reset_settle
    ap.add_argument("--phase_input", action="store_true")  # append oracle 3-phase to obs
    ap.add_argument("--settle", type=int, default=8)  # settle steps for reset_settle
    ap.add_argument("--n", type=int, default=100)
    ap.add_argument("--max_steps", type=int, default=700)
    args = ap.parse_args()

    policy_mode = args.warm_to != "all"
    if policy_mode:
        cfgdir = os.path.abspath("examples/configs")
        with initialize_config_dir(version_base=None, config_dir=cfgdir):
            ov_ = ["task=tool_hang_ph_state_delta_legacy",
                f"+task.dataset_path={os.path.abspath(args.dataset)}",
                "network=chiunet", f"optimization.loss_type={args.loss}",
                "optimization.auto_resume=false", "log.wandb_mode=disabled"]
            if args.phase_input:
                ov_.append("+task.phase_input=true")
            cfg = compose(config_name="main", overrides=ov_)
        OmegaConf.set_struct(cfg, False); cfg.task.obs_dim = 56 if args.phase_input else 53
        dev = cfg.optimization.device
        H = 16; AS = cfg.task.act_steps; start = cfg.task.obs_steps - 1
        ds = make_dataset(cfg.task)
        agent = TrainingAgent(cfg); agent.load(args.ckpt, load_optimizer=False); agent.eval()
        no = ds.normalizer["obs"]["state"]; na = ds.normalizer["action"]

    df = h5py.File(args.demos, "r")
    keys = list(df["demos"].keys())[:args.n]
    env = robosuite.make("ToolHang", horizon=4000, **ENV_KWARGS)

    def oracle_phase():
        sim = env.sim
        fz = sim.data.site_xpos[sim.model.site_name2id("frame_mount_site")]
        hc = np.mean([sim.data.geom_xpos[sim.model.geom_name2id(f"stand_wall{i}")] for i in range(4)], axis=0)
        lifted = fz[2] > 0.86; near = np.linalg.norm((fz - hc)[:2]) < 0.05
        ph = np.zeros(3, dtype=np.float32)
        ph[2 if (lifted and near) else (1 if lifted else 0)] = 1.0
        return ph

    def ov(o):
        base = np.concatenate([o[KM.get(k, k)] for k in OK]).astype(np.float32)
        if args.phase_input:
            base = np.concatenate([base, oracle_phase()]).astype(np.float32)
        return base

    def is_success():
        if args.success == "grasp":
            grasped = env._check_grasp(gripper=env.robots[0].gripper["right"],
                                       object_geoms=env.frame.contact_geoms)
            fz = env.sim.data.site_xpos[env.sim.model.site_name2id("frame_mount_site")][2]
            return bool(grasped and fz > 0.86)  # grasped AND lifted off rest (init z=0.812)
        return bool(env._check_frame_assembled())

    succ = 0; N = 0
    for k in keys:
        d = df["demos/" + k]
        seed = int(d.attrs["seed"]); acts = d["actions"][:]; state0 = d["state0"][:]
        c1 = int(d.attrs["c1"]); ad = int(d.attrs["align_done"])
        if args.warm_to == "all":
            K = len(acts)
        elif args.warm_to in ("c1", "align_done"):
            K = int(d.attrs[args.warm_to])
        elif args.warm_to == "mid_reach":
            K = c1 // 2
        elif args.warm_to == "mid_align":
            K = (c1 + ad) // 2
        else:
            K = min(int(args.warm_to), len(acts))
        np.random.seed(seed); env.reset()
        if args.init_mode == "state0":
            sim = env.sim; sim.set_state_from_flattened(state0); sim.forward()
            arm = env.robots[0].composite_controller.part_controllers["right"]
            arm.update(); arm.reset_goal()
            o = env._get_observations(force_update=True)
        else:  # reset (unsettled) or reset_settle (env.reset + a few settle steps)
            arm = env.robots[0].composite_controller.part_controllers["right"]
            arm.update(); arm.reset_goal()
            o = env._get_observations(force_update=True)
            if args.init_mode == "reset_settle":
                for _ in range(args.settle):
                    o, _, _, _ = env.step(np.zeros(7))  # match scripted leading settle (10x zeros)
        prev = ov(o); cur = ov(o)
        asm = False; steps = 0
        # warm-start replay
        for t in range(K):
            o, _, _, _ = env.step(np.clip(acts[t], -1, 1)); steps += 1
            prev = cur; cur = ov(o)
            if is_success():
                asm = True; break
        # policy takeover
        if policy_mode and not asm:
            hist = [prev, cur]
            while steps < args.max_steps and not asm:
                w = np.stack(hist[-2:])[None]
                ot = {"state": torch.tensor(no.normalize(w), device=dev, dtype=torch.float32)}
                with torch.no_grad():
                    an = agent.sample(act_0=torch.randn((1, H, 10), device=dev), obs=ot, use_ema=True)
                a7 = ds.undo_transform_action(na.unnormalize(an.detach().cpu().numpy())[:, start:start+AS])[0]
                for a in a7:
                    o, _, _, _ = env.step(a); steps += 1; hist.append(ov(o))
                    if is_success():
                        asm = True; break
                    if steps >= args.max_steps:
                        break
        succ += int(asm); N += 1
    df.close()
    tag = "REPLAY_ALL" if args.warm_to == "all" else f"warm_to={args.warm_to}"
    print(f"WARMSTART {tag} {succ}/{N} = {100*succ/N:.1f}%")


if __name__ == "__main__":
    main()
