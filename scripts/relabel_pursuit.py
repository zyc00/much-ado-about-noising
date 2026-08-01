"""Hybrid pursuit relabel: make actions follow the realized trajectory.

Moving steps (lead-window displacement > --move_thresh):
  a_pos[t] = (eef_pos[t+L] - eef_pos[t]) / 0.05          (clipped to [-1,1])
  a_rot[t] = axisangle(R[t+L] @ R[t]^T) / 0.5            (clipped)
Stationary/contact steps: keep original action (press semantics preserved).
Gripper always original.

Modes:
  --verify N   : relabel N demos and REPLAY them open-loop from the stored initial
                 state (model_file + states[0]); report frame-assembled success rate
                 and final eef deviation vs recorded trajectory. Sweep --L values.
  --out FILE   : write the fully relabeled dataset.
"""
import os
os.environ["MUJOCO_GL"] = "egl"
import argparse
import sys
sys.path.insert(0, "scripts"); sys.path.insert(0, ".")
import numpy as np
import h5py
import robosuite.utils.transform_utils as T


def relabel_demo(obs_pos, obs_quat, actions, L, move_thresh, mode="disp"):
    Tn = min(len(obs_pos), len(actions))
    out = actions.copy()
    n_mov = 0
    for t in range(Tn):
        t2 = min(t + L, Tn - 1)
        d = obs_pos[t2] - obs_pos[t]
        if np.linalg.norm(d) <= move_thresh:
            continue  # stationary/contact: keep original
        n_mov += 1
        if mode == "magdir":
            # direction = realized path; magnitude = original command (keeps drive/feedforward)
            out[t, :3] = np.clip(np.linalg.norm(actions[t, :3]) * d / np.linalg.norm(d), -1, 1)
        else:
            out[t, :3] = np.clip(d / 0.05, -1, 1)
        Rt = T.quat2mat(obs_quat[t]); R2 = T.quat2mat(obs_quat[t2])
        out[t, 3:6] = np.clip(T.quat2axisangle(T.mat2quat(R2 @ Rt.T)) / 0.5, -1, 1)
    return out, n_mov / max(Tn, 1)


def replay(env, model_file, state0, actions):
    env.reset()
    xml = env.edit_model_xml(model_file) if hasattr(env, "edit_model_xml") else model_file
    env.reset_from_xml_string(xml)
    env.sim.reset()
    env.sim.set_state_from_flattened(state0)
    env.sim.forward()
    arm = env.robots[0].composite_controller.part_controllers["right"]
    arm.update(); arm.reset_goal()
    asm = False
    eefs = []
    for a in actions:
        env.step(np.asarray(a, dtype=np.float64))
        eefs.append(env._get_observations(force_update=True)["robot0_eef_pos"].copy())
        if env._check_frame_assembled():
            asm = True
            break
    return asm, np.array(eefs)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--src", default="data/tool_hang_full2ins_smooth_2000.hdf5")
    ap.add_argument("--out", default=None)
    ap.add_argument("--verify", type=int, default=0)
    ap.add_argument("--L", type=int, nargs="+", default=[6])
    ap.add_argument("--move_thresh", type=float, default=0.002)
    args = ap.parse_args()

    src = h5py.File(args.src, "r")
    keys = sorted(src["data"].keys(), key=lambda s: int(s.split("_")[1]))

    if args.verify and os.environ.get("WINVERIFY") == "1":
        verify_windows(args, src, keys)
        return
    if args.verify:
        import robosuite
        from collect_tool_hang_demos import ENV_KWARGS
        env = robosuite.make("ToolHang", horizon=4000, **ENV_KWARGS)
        for L in args.L:
            succ = succ_orig = 0; devs = []; movfrac = []
            for k in keys[: args.verify]:
                d = src[f"data/{k}"]
                pos = np.asarray(d["obs/robot0_eef_pos"])
                quat = np.asarray(d["obs/robot0_eef_quat"])
                act = np.asarray(d["actions"])
                new, mf = relabel_demo(pos, quat, act, L, args.move_thresh)
                movfrac.append(mf)
                asm, eefs = replay(env, d.attrs["model_file"], np.asarray(d["states"])[0], new)
                succ += int(asm)
                if len(eefs):
                    n = min(len(eefs), len(pos))
                    devs.append(np.linalg.norm(eefs[:n] - pos[:n], axis=1).max())
                if L == args.L[0]:
                    asm0, _ = replay(env, d.attrs["model_file"], np.asarray(d["states"])[0], act)
                    succ_orig += int(asm0)
            line = (f"RELABEL-VERIFY L={L} success={succ}/{args.verify} "
                    f"maxTrackDev_p50={np.median(devs)*1000:.1f}mm p90={np.percentile(devs,90)*1000:.1f}mm "
                    f"movfrac={np.mean(movfrac):.2f}")
            if L == args.L[0]:
                line += f" | ORIGINAL-action replay control: {succ_orig}/{args.verify}"
            print(line, flush=True)
        return

    assert args.out
    L = args.L[0]
    dst = h5py.File(args.out, "w")
    src.copy("data", dst)
    for k in keys:
        d = dst[f"data/{k}"]
        pos = np.asarray(d["obs/robot0_eef_pos"])
        quat = np.asarray(d["obs/robot0_eef_quat"])
        act = np.asarray(d["actions"])
        new, _ = relabel_demo(pos, quat, act, L, args.move_thresh, mode=os.environ.get("RELMODE", "disp"))
        d["actions"][...] = new
    dst.close()
    print(f"saved {args.out} (L={L})", flush=True)




def verify_windows(args, src, keys):
    """Short-horizon fidelity: set_state(states[t]), run 12 steps of orig vs relabeled
    actions, endpoint deviation vs recorded pos[t+12]."""
    import robosuite
    from collect_tool_hang_demos import ENV_KWARGS
    env = robosuite.make("ToolHang", horizon=4000, **ENV_KWARGS)
    W = 12
    rng = np.random.RandomState(0)
    for L in args.L:
        dev_o, dev_n = [], []
        for k in keys[: args.verify]:
            d = src[f"data/{k}"]
            pos = np.asarray(d["obs/robot0_eef_pos"])
            quat = np.asarray(d["obs/robot0_eef_quat"])
            act = np.asarray(d["actions"])
            states = np.asarray(d["states"])
            new, _ = relabel_demo(pos, quat, act, L, args.move_thresh)
            Tn = min(len(pos), len(act))
            for t in rng.choice(np.arange(2, Tn - W - 2), 8, replace=False):
                for acts, acc in [(act, dev_o), (new, dev_n)]:
                    env.reset()
                    xml = env.edit_model_xml(d.attrs["model_file"]) if hasattr(env, "edit_model_xml") else d.attrs["model_file"]
                    env.reset_from_xml_string(xml)
                    env.sim.reset(); env.sim.set_state_from_flattened(states[t]); env.sim.forward()
                    arm = env.robots[0].composite_controller.part_controllers["right"]
                    arm.update(); arm.reset_goal()
                    for a in acts[t:t + W]:
                        env.step(np.asarray(a, dtype=np.float64))
                    ee = env._get_observations(force_update=True)["robot0_eef_pos"]
                    acc.append(np.linalg.norm(ee - pos[t + W]))
        print(f"RELABEL-WIN L={L} W={W} origDev p50={np.median(dev_o)*1000:.1f}mm p90={np.percentile(dev_o,90)*1000:.1f}mm | "
              f"newDev p50={np.median(dev_n)*1000:.1f}mm p90={np.percentile(dev_n,90)*1000:.1f}mm", flush=True)


if __name__ == "__main__":
    main()
