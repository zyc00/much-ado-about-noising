"""Collect scripted tool_hang demos in robomimic-compatible HDF5 format.

Format matches robomimic official tool_hang/ph dataset:
- data attrs: env_args (JSON of env config), total (num demos)
- demo_N groups:
  - actions: (T, 7) float64 — OSC_POSE delta in world frame [-1, 1]
  - states: (T, 58) — mujoco sim state (time + qpos + qvel)
  - rewards: (T,)
  - dones: (T,)
  - obs/{key}: (T, dim) — per-key observation arrays
  - attrs: num_samples, model_file (XML)

Action format (verified against robomimic env_args):
- Controller: OSC_POSE, delta=true, world frame
- output_max=[0.05,0.05,0.05,0.5,0.5,0.5], kp=150, damping=1
- action[0:3]=delta_pos, action[3:6]=delta_axisangle, action[6]=gripper
- gripper: -1 = open, 1 = close

Usage:
    MUJOCO_GL=egl python scripts/collect_tool_hang_demos.py --n_demos 200 --output data/tool_hang_scripted.hdf5
"""

import os
os.environ["MUJOCO_GL"] = "egl"

import argparse
import json
import h5py
import numpy as np
import robosuite
import robosuite.utils.transform_utils as T
from scipy.spatial.transform import Rotation as R
from tqdm import tqdm


# Env config matching robomimic official tool_hang/ph
ENV_KWARGS = {
    "has_renderer": False,
    "has_offscreen_renderer": False,
    "ignore_done": True,
    "use_object_obs": True,
    "use_camera_obs": False,
    "control_freq": 20,
    "controller_configs": {
        "type": "BASIC",
        "body_parts": {
            "right": {
                "type": "OSC_POSE",
                "input_max": 1,
                "input_min": -1,
                "output_max": [0.05, 0.05, 0.05, 0.5, 0.5, 0.5],
                "output_min": [-0.05, -0.05, -0.05, -0.5, -0.5, -0.5],
                "kp": 150,
                "damping": 1,
                "impedance_mode": "fixed",
                "kp_limits": [0, 300],
                "damping_limits": [0, 10],
                "position_limits": None,
                "orientation_limits": None,
                "uncouple_pos_ori": True,
                "control_delta": True,
                "interpolation": None,
                "ramp_ratio": 0.2,
                "input_ref_frame": "world",
                "gripper": {"type": "GRIP"},
            }
        },
    },
    "robots": ["Panda"],
    "camera_depths": False,
    "camera_heights": 84,
    "camera_widths": 84,
    "lite_physics": False,
    "reward_shaping": False,
}

ENV_ARGS = {"env_name": "ToolHang", "env_version": "1.5.1", "type": 1, "env_kwargs": ENV_KWARGS}

# Obs keys to record (match robomimic official dataset)
OBS_KEYS_TO_RECORD = [
    "object",                    # constructed from object-state
    "robot0_eef_pos",
    "robot0_eef_quat",
    "robot0_eef_quat_site",
    "robot0_gripper_qpos",
    "robot0_gripper_qvel",
    "robot0_joint_pos",          # constructed from sin/cos
    "robot0_joint_pos_cos",
    "robot0_joint_pos_sin",
    "robot0_joint_vel",
]


def make_env(horizon=4000):
    return robosuite.make("ToolHang", horizon=horizon, **ENV_KWARGS)


def extract_obs(obs):
    """Convert raw env obs into dict with robomimic-compatible keys."""
    out = {}
    out["object"] = obs["object-state"].copy()
    out["robot0_eef_pos"] = obs["robot0_eef_pos"].copy()
    out["robot0_eef_quat"] = obs["robot0_eef_quat"].copy()
    out["robot0_eef_quat_site"] = obs["robot0_eef_quat_site"].copy()
    out["robot0_gripper_qpos"] = obs["robot0_gripper_qpos"].copy()
    out["robot0_gripper_qvel"] = obs["robot0_gripper_qvel"].copy()
    cos = obs["robot0_joint_pos_cos"]; sin = obs["robot0_joint_pos_sin"]
    out["robot0_joint_pos"] = np.arctan2(sin, cos)
    out["robot0_joint_pos_cos"] = cos.copy()
    out["robot0_joint_pos_sin"] = sin.copy()
    out["robot0_joint_vel"] = obs["robot0_joint_vel"].copy()
    return out


def run_episode(env, seed):
    """Run scripted policy. Returns (success, trajectory dict)."""
    np.random.seed(seed)
    obs_h = [env.reset()]
    sim = env.sim

    # Record initial state and model XML
    init_state = sim.get_state().flatten()
    model_xml = sim.model.get_xml()

    traj = {"actions": [], "states": [], "rewards": [], "dones": [],
            "obs": {k: [] for k in OBS_KEYS_TO_RECORD}}

    def do_step(a):
        a = np.clip(a, -1, 1).astype(np.float64)
        # Record current obs + state BEFORE the action
        traj["actions"].append(a.copy())
        traj["states"].append(sim.get_state().flatten().copy())
        for k, v in extract_obs(obs_h[0]).items():
            traj["obs"][k].append(v)
        obs_h[0], reward, done, _ = env.step(a)
        traj["rewards"].append(float(reward))
        traj["dones"].append(int(done))

    def obs():
        return obs_h[0]

    def move_ori(tp, tq, g, steps=80, pg=10, og=3, pt=0.005, ot=0.03):
        for i in range(steps):
            o = obs(); eef = o["robot0_eef_pos"]
            dp = np.clip((tp - eef) * pg, -1, 1)
            em = T.quat2mat(o["robot0_eef_quat"]); tm = T.quat2mat(tq)
            ea = T.quat2axisangle(T.mat2quat(tm @ em.T))
            do = np.clip(ea * og, -1, 1)
            do_step(np.concatenate([dp, do, [g]]))
            if np.linalg.norm(eef - tp) < pt and np.linalg.norm(ea) < ot:
                return i
        return steps

    def cfq():
        o = obs(); em = T.quat2mat(o["robot0_eef_quat"])
        fm = gs("frame_mount_site"); fi = gs("frame_intersection_site"); fh = gs("frame_hang_site")
        nw = (fm - fi) / np.linalg.norm(fm - fi)
        hw = (fh - fi) / np.linalg.norm(fh - fi)
        ne = em.T @ nw; he = em.T @ hw
        neu = ne / np.linalg.norm(ne)
        hep = he - np.dot(he, neu) * neu; heu = hep / np.linalg.norm(hep)
        te = np.cross(neu, heu)
        ed = np.column_stack([neu, heu, te])
        nt = np.array([0, 0, -1.0]); ht = np.array([0, -1, 0.0]); tt = np.cross(nt, ht)
        wt = np.column_stack([nt, ht, tt])
        return T.mat2quat(wt @ np.linalg.inv(ed))

    def gs(name):
        return sim.data.site_xpos[sim.model.site_name2id(name)].copy()

    # === Phase 1: Frame Assembly ===
    for _ in range(10):
        do_step(np.zeros(7))

    fp = obs()["frame_pos"].copy()
    fmw = T.quat2mat(obs()["frame_quat"].copy())
    fx = fmw[:, 0].copy(); fx[2] = 0; fx = fx / np.linalg.norm(fx)
    fa = np.arctan2(fx[1], fx[0])
    fmount = gs("frame_mount_site"); nd = fmount - fp; nd[2] = 0; nd = nd / np.linalg.norm(nd)
    hc = np.zeros(3)
    for i in range(4):
        hc += sim.data.geom_xpos[sim.model.geom_name2id(f"stand_wall{i}")]
    hc /= 4; sm = gs("stand_mount_site"); hc[2] = sm[2]
    pa = fa - np.pi / 2
    gq = T.mat2quat(np.column_stack([
        np.array([np.cos(pa), np.sin(pa), 0]),
        np.cross(np.array([0, 0, -1.0]), np.array([np.cos(pa), np.sin(pa), 0])),
        np.array([0, 0, -1.0]),
    ]))
    gp = fp + nd * 0.02; gp[2] = fp[2]
    ab = gp.copy(); ab[2] += 0.10
    move_ori(ab, gq, -1, steps=40, pg=12)
    dn = gp.copy(); dn[2] -= 0.005
    move_ori(dn, gq, -1, steps=50, pg=8, og=5)
    for _ in range(15):
        o = obs(); dp = np.clip((dn - o["robot0_eef_pos"]) * 3, -1, 1)
        ea = T.quat2axisangle(T.mat2quat(T.quat2mat(gq) @ T.quat2mat(o["robot0_eef_quat"]).T))
        do_step(np.concatenate([dp, np.clip(ea * 2, -1, 1), [1]]))
    lf = obs()["robot0_eef_pos"].copy(); lf[2] += 0.25
    move_ori(lf, gq, 1, steps=30, pg=15, og=3)
    if obs()["frame_pos"][2] - fp[2] < 0.05:
        return False, traj, init_state, model_xml
    tq = cfq()
    move_ori(obs()["robot0_eef_pos"].copy(), tq, 1, steps=80, pg=4, og=8, ot=0.02)
    tq2 = cfq()
    move_ori(obs()["robot0_eef_pos"].copy(), tq2, 1, steps=40, pg=4, og=10, ot=0.01)
    for _ in range(3):
        etn = gs("frame_mount_site") - obs()["robot0_eef_pos"]
        tf = hc - etn; tf[2] = sm[2] + 0.05 - etn[2]
        move_ori(tf, tq2, 1, steps=35, pg=25, og=10, pt=0.002, ot=0.015)

    # Push down with gradual 60° tilt
    tq2_mat = T.quat2mat(tq2); cur_z = tq2_mat[:, 2]; target_z = np.array([0, 0, -1.0])
    axis = np.cross(cur_z, target_z); axis = axis / np.linalg.norm(axis)
    rot = R.from_rotvec(axis * np.radians(60)).as_matrix()
    tilted_q = T.mat2quat(rot @ tq2_mat)
    n_push = 120
    for i in range(n_push):
        frac = (i + 1) / n_push
        dot = np.dot(tq2, tilted_q)
        if dot < 0:
            tilted_use = -tilted_q; dot = -dot
        else:
            tilted_use = tilted_q
        dot = np.clip(dot, -1, 1); theta = np.arccos(dot)
        if theta < 1e-6:
            wp_quat = tilted_use
        else:
            wp_quat = (np.sin((1 - frac) * theta) / np.sin(theta)) * tq2 \
                + (np.sin(frac * theta) / np.sin(theta)) * tilted_use
        wp_quat = wp_quat / np.linalg.norm(wp_quat)
        em = T.quat2mat(obs()["robot0_eef_quat"]); tm = T.quat2mat(wp_quat)
        ea = T.quat2axisangle(T.mat2quat(tm @ em.T))
        do_step(np.concatenate([[0, 0, -1], np.clip(ea * 8, -1, 1), [1]]))
        if obs()["robot0_eef_pos"][2] <= 1.0:
            break

    for _ in range(20):
        do_step(np.array([0, 0, 0, 0, 0, 0, -1]))
    for _ in range(25):
        do_step(np.array([1, 0, 0, 0, 0, 0, -1]))
    if not env._check_frame_assembled():
        return False, traj, init_state, model_xml

    # === Phase 2: Tool Pick + Hang ===
    tp = obs()["tool_pos"].copy(); target = tp.copy(); target[2] += 0.05
    for i in range(120):
        eef = obs()["robot0_eef_pos"]; dp = np.clip((target - eef) * 12, -1, 1)
        do_step(np.concatenate([dp, [0, 0, 0, -1]]))
        if np.linalg.norm(eef - target) < 0.01:
            break
    tool_mat = T.quat2mat(obs()["tool_quat"].copy())
    tool_x = tool_mat[:, 0].copy(); tool_x[2] = 0; tool_x = tool_x / np.linalg.norm(tool_x)
    ta = np.arctan2(tool_x[1], tool_x[0])
    x_eef = np.array([np.cos(ta), np.sin(ta), 0])
    y_eef = np.cross(np.array([0, 0, -1.0]), x_eef)
    tgq = T.mat2quat(np.column_stack([x_eef, y_eef, np.array([0, 0, -1.0])]))
    move_ori(obs()["robot0_eef_pos"].copy(), tgq, -1, steps=100, pg=8, og=5)
    tp2 = obs()["tool_pos"].copy(); ab2 = tp2.copy(); ab2[2] += 0.05
    move_ori(ab2, tgq, -1, steps=40, pg=15, og=5)
    tp3 = obs()["tool_pos"].copy(); dn2 = tp3.copy(); dn2[2] -= 0.005
    move_ori(dn2, tgq, -1, steps=50, pg=8, og=5)
    for _ in range(15):
        o = obs(); dp = np.clip((dn2 - o["robot0_eef_pos"]) * 3, -1, 1)
        ea = T.quat2axisangle(T.mat2quat(T.quat2mat(tgq) @ T.quat2mat(o["robot0_eef_quat"]).T))
        do_step(np.concatenate([dp, np.clip(ea * 2, -1, 1), [1]]))
    lift = obs()["robot0_eef_pos"].copy(); lift[2] += 0.20
    move_ori(lift, tgq, 1, steps=40, pg=12, og=3)
    if obs()["tool_pos"][2] - tp3[2] < 0.05:
        return False, traj, init_state, model_xml

    # Hang
    fh = gs("frame_hang_site"); th = gs("tool_hole1_center"); eth = th - obs()["robot0_eef_pos"]
    ta2 = fh - eth; ta2[2] = fh[2] + 0.10
    move_ori(ta2, tgq, 1, steps=60, pg=12, og=5)
    for _ in range(3):
        th = gs("tool_hole1_center"); etn = th - obs()["robot0_eef_pos"]
        tf = fh - etn; tf[2] = fh[2] + 0.05
        move_ori(tf, tgq, 1, steps=30, pg=20, og=8, pt=0.002, ot=0.01)
    for _ in range(2):
        th = gs("tool_hole1_center"); etn = th - obs()["robot0_eef_pos"]
        tf = fh - etn; tf[2] = fh[2] - 0.02
        move_ori(tf, tgq, 1, steps=30, pg=5, og=3)

    # Gentle down with grip → release → firmer/longer down to seat tool fully
    # on hook without destabilizing the frame
    for _ in range(20):
        do_step(np.array([0, 0, -0.2, 0, 0, 0, 1]))
    for _ in range(15):
        do_step(np.array([0, 0, 0, 0, 0, 0, -1]))
    for _ in range(40):
        do_step(np.array([0, 0, -0.2, 0, 0, 0, -1]))
    for _ in range(10):
        do_step(np.array([0, 0, 0, 0, 0, 0, -1]))

    success = env._check_success()
    return success, traj, init_state, model_xml


def save_demos(demos, output_path):
    """Save demos to HDF5 in robomimic format."""
    os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)
    with h5py.File(output_path, "w") as f:
        data = f.create_group("data")
        data.attrs["env_args"] = json.dumps(ENV_ARGS)
        data.attrs["total"] = sum(len(d["actions"]) for d in demos)

        for i, demo in enumerate(demos):
            grp = data.create_group(f"demo_{i}")
            grp.attrs["num_samples"] = len(demo["actions"])
            grp.attrs["model_file"] = demo["model_file"]
            grp.attrs["seed"] = demo["seed"]
            grp.create_dataset("actions", data=np.array(demo["actions"], dtype=np.float64))
            grp.create_dataset("states", data=np.array(demo["states"], dtype=np.float64))
            grp.create_dataset("rewards", data=np.array(demo["rewards"], dtype=np.float64))
            grp.create_dataset("dones", data=np.array(demo["dones"], dtype=np.int64))
            obs_grp = grp.create_group("obs")
            for k in OBS_KEYS_TO_RECORD:
                obs_grp.create_dataset(k, data=np.array(demo["obs"][k]))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--n_demos", type=int, default=200)
    parser.add_argument("--output", type=str, default="data/tool_hang_scripted.hdf5")
    parser.add_argument("--start_seed", type=int, default=0)
    args = parser.parse_args()

    env = make_env(horizon=4000)
    demos = []
    seed = args.start_seed
    attempts = 0
    pbar = tqdm(total=args.n_demos, desc="Collecting")
    while len(demos) < args.n_demos:
        success, traj, init_state, model_xml = run_episode(env, seed)
        attempts += 1
        if success:
            traj["model_file"] = model_xml
            traj["seed"] = seed
            demos.append(traj)
            pbar.update(1)
        seed += 1
    pbar.close()
    print(f"Collected {len(demos)} successful demos from {attempts} attempts "
          f"({100 * len(demos) / attempts:.0f}% success)")

    save_demos(demos, args.output)
    print(f"Saved to {args.output}")

    # Print summary
    n_samples = sum(len(d["actions"]) for d in demos)
    print(f"Total samples: {n_samples}, avg per demo: {n_samples / len(demos):.0f}")


if __name__ == "__main__":
    main()
