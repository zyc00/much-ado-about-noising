"""Convert the delta-action full2ins dataset to ABSOLUTE actions, exactly.
For every step: restore the exact recorded pre-state, execute the original delta action,
and read the OSC controller's OWN absolute goal (goal_pos, goal_ori) — no frame guesswork.
Abs action = [goal_pos (m), mat2axisangle(goal_ori) (rad), gripper]  (robomimic abs format).
Copies obs/states/rewards/dones and attrs; patches env_args controller to absolute.
--validate N: replay N converted demos in an ABSOLUTE-controller env from state0 and report
final assembled rate + eef tracking error vs recorded states."""
import os
os.environ["MUJOCO_GL"] = "egl"
import argparse
import json
import numpy as np
import h5py
import robosuite
import robosuite.utils.transform_utils as T
import sys
sys.path.insert(0, "scripts"); sys.path.insert(0, ".")
from collect_tool_hang_demos import ENV_KWARGS
import copy


def abs_env_kwargs():
    ek = copy.deepcopy(ENV_KWARGS)
    ek["controller_configs"]["body_parts"]["right"]["input_type"] = "absolute"
    ek["controller_configs"]["body_parts"]["right"]["input_max"] = 1
    ek["controller_configs"]["body_parts"]["right"]["input_min"] = -1
    return ek


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--src", default="data/tool_hang_full2ins_2000.hdf5")
    ap.add_argument("--dst", default="data/tool_hang_full2ins_abs_2000.hdf5")
    ap.add_argument("--n", type=int, default=0, help="0=all demos")
    ap.add_argument("--validate", type=int, default=0, help="replay N converted demos with abs controller")
    args = ap.parse_args()

    src = h5py.File(args.src, "r")
    ks = sorted(src["data"].keys(), key=lambda k: int(k.split("_")[1]))
    if args.n: ks = ks[:args.n]

    env = robosuite.make("ToolHang", horizon=4000, **ENV_KWARGS)
    arm = None

    dst = h5py.File(args.dst, "w")
    dg = dst.create_group("data")
    for a, v in src["data"].attrs.items():
        if a == "env_args":
            meta = json.loads(v)
            bp = meta["env_kwargs"]["controller_configs"]["body_parts"]["right"]
            bp["input_type"] = "absolute"; bp["input_max"] = 1; bp["input_min"] = -1
            dg.attrs["env_args"] = json.dumps(meta)
        else:
            dg.attrs[a] = v

    for n_done, k in enumerate(ks):
        d = src[f"data/{k}"]
        acts = np.clip(np.asarray(d["actions"]), -1, 1)
        states = np.asarray(d["states"])
        Tn = len(acts)
        abs_acts = np.zeros((Tn, 7), dtype=np.float32)
        env.reset()
        arm = env.robots[0].composite_controller.part_controllers["right"]
        for t in range(Tn):
            env.sim.set_state_from_flattened(states[t]); env.sim.forward()
            arm.update(); arm.reset_goal()
            env.step(acts[t])
            gp = np.array(arm.goal_pos).ravel()
            go = np.array(arm.goal_ori)
            aa = T.quat2axisangle(T.mat2quat(go))
            abs_acts[t, :3] = gp; abs_acts[t, 3:6] = aa; abs_acts[t, 6] = acts[t, 6]
        g = dg.create_group(k)
        for a, v in d.attrs.items(): g.attrs[a] = v
        g.create_dataset("actions", data=abs_acts)
        for f in ["states", "rewards", "dones"]:
            if f in d: g.create_dataset(f, data=np.asarray(d[f]))
        src.copy(f"data/{k}/obs", g, name="obs")
        if (n_done + 1) % 50 == 0:
            print(f"converted {n_done+1}/{len(ks)}", flush=True)
    dst.close()
    print(f"saved {args.dst} ({len(ks)} demos)")

    if args.validate:
        env2 = robosuite.make("ToolHang", horizon=4000, **abs_env_kwargs())
        h = h5py.File(args.dst, "r")
        succ = 0; errs = []
        vks = list(h["data"].keys())[:args.validate]
        for k in vks:
            d = h[f"data/{k}"]
            acts = np.asarray(d["actions"]); states = np.asarray(d["states"])
            env2.reset()
            env2.sim.set_state_from_flattened(states[0]); env2.sim.forward()
            arm2 = env2.robots[0].composite_controller.part_controllers["right"]
            arm2.update(); arm2.reset_goal()
            for t in range(len(acts)):
                env2.step(acts[t])
                if t + 1 < len(states):
                    ref = states[t + 1]
                    cur = env2.sim.get_state().flatten()
                    m = min(len(ref), len(cur))
                    errs.append(float(np.abs(ref[:m] - cur[:m]).max()))
            succ += int(env2._check_frame_assembled())
        print(f"VALIDATE abs replay: assembled {succ}/{len(vks)}  "
              f"state err p50 {np.percentile(errs,50):.4f} p95 {np.percentile(errs,95):.4f}")
        h.close()


if __name__ == "__main__":
    main()
