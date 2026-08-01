"""Closed-loop SR of the pursuit action field itself (the 'expert' BC will imitate).
From each demo's stored init (model_file + states[0]): at every step, find nearest
recorded path index (monotone), command (path[i+L]-eef_now)/0.05 (rot analog);
if the lead window is stationary (<thresh) execute the recorded original action
(press semantics). Success = frame assembled within 1.5x recorded length."""
import os
os.environ["MUJOCO_GL"] = "egl"
import argparse, sys
sys.path.insert(0, "scripts"); sys.path.insert(0, ".")
import numpy as np, h5py, robosuite
import robosuite.utils.transform_utils as T
from collect_tool_hang_demos import ENV_KWARGS

ap = argparse.ArgumentParser()
ap.add_argument("--src", default="data/tool_hang_full2ins_smooth_2000.hdf5")
ap.add_argument("--n", type=int, default=30)
ap.add_argument("--L", type=int, default=4)
ap.add_argument("--move_thresh", type=float, default=0.002)
args = ap.parse_args()

env = robosuite.make("ToolHang", horizon=4000, **ENV_KWARGS)
src = h5py.File(args.src, "r")
keys = sorted(src["data"].keys(), key=lambda s: int(s.split("_")[1]))[: args.n]
succ = 0
for k in keys:
    d = src[f"data/{k}"]
    pos = np.asarray(d["obs/robot0_eef_pos"]); quat = np.asarray(d["obs/robot0_eef_quat"])
    act = np.asarray(d["actions"]); Tn = min(len(pos), len(act))
    env.reset()
    xml = env.edit_model_xml(d.attrs["model_file"]) if hasattr(env, "edit_model_xml") else d.attrs["model_file"]
    env.reset_from_xml_string(xml)
    env.sim.reset(); env.sim.set_state_from_flattened(np.asarray(d["states"])[0]); env.sim.forward()
    arm = env.robots[0].composite_controller.part_controllers["right"]
    arm.update(); arm.reset_goal()
    i = 0; asm = False
    o = env._get_observations(force_update=True)
    for step in range(int(1.5 * Tn)):
        ee = o["robot0_eef_pos"]
        locked = np.linalg.norm(pos[i] - ee) < 0.05
        if locked:
            w = slice(i, min(i + 12, Tn))
            i = i + int(np.argmin(np.linalg.norm(pos[w] - ee, axis=1)))
        j = min(i + args.L, Tn - 1)   # always lead; lock only gates index advance
        if locked and np.linalg.norm(pos[j] - pos[i]) <= args.move_thresh:
            a = act[i].astype(np.float64)          # press/stationary: recorded action
            if np.linalg.norm(pos[i] - ee) < 0.005:
                i = min(i + 1, Tn - 1)
        else:
            dp = np.clip((pos[j] - ee) / 0.05, -1, 1)
            Re = T.quat2mat(o["robot0_eef_quat"]); Rj = T.quat2mat(quat[j])
            dr = np.clip(T.quat2axisangle(T.mat2quat(Rj @ Re.T)) / 0.5, -1, 1)
            a = np.concatenate([dp, dr, act[i, 6:]])
        o, _, _, _ = env.step(a)
        if env._check_frame_assembled(): asm = True; break
        if i >= Tn - 1 and step > Tn: break
    succ += int(asm)
    ee = o["robot0_eef_pos"]
    print(f"{k}: {'OK' if asm else 'fail'} | i={i}/{Tn} ({i/Tn:.2f}) distToPath={np.linalg.norm(ee-pos[min(i,Tn-1)])*1000:.0f}mm", flush=True)
print(f"PURSUIT-CL-SR L={args.L}: {succ}/{len(keys)} = {100*succ/len(keys):.0f}%", flush=True)
