"""Collect deterministic scripted demonstrations for tool_hang task."""

import os
os.environ["MUJOCO_GL"] = "egl"

import h5py
import numpy as np
import robosuite
import robosuite.utils.transform_utils as T


def make_env(seed=0):
    env = robosuite.make(
        "ToolHang",
        robots="Panda",
        has_renderer=False,
        has_offscreen_renderer=False,
        use_camera_obs=False,
        use_object_obs=True,
        reward_shaping=False,
        horizon=700,
        initialization_noise={"magnitude": 0.3, "type": "gaussian"},
    )
    return env


def get_site_pos(env, name):
    sid = env.sim.model.site_name2id(name)
    return env.sim.data.site_xpos[sid].copy()


def get_body_pos(env, name):
    bid = env.sim.model.body_name2id(name)
    return env.sim.data.body_xpos[bid].copy()


def get_body_quat(env, name):
    bid = env.sim.model.body_name2id(name)
    return env.sim.data.body_xquat[bid].copy()


def move_to(obs, target_pos, target_quat=None, gripper_action=1.0, gain=10.0, ori_gain=3.0):
    """Compute delta action to move EEF toward target."""
    eef_pos = obs["robot0_eef_pos"]
    delta_pos = (target_pos - eef_pos) * gain
    delta_pos = np.clip(delta_pos, -1.0, 1.0)

    if target_quat is not None:
        eef_quat = obs["robot0_eef_quat"]
        # Compute orientation error as axis-angle
        eef_mat = T.quat2mat(eef_quat)
        target_mat = T.quat2mat(target_quat)
        error_mat = target_mat @ eef_mat.T
        error_quat = T.mat2quat(error_mat)
        error_aa = T.quat2axisangle(error_quat)
        delta_ori = error_aa * ori_gain
        delta_ori = np.clip(delta_ori, -1.0, 1.0)
    else:
        delta_ori = np.zeros(3)

    return np.concatenate([delta_pos, delta_ori, [gripper_action]])


def at_target(obs, target_pos, tol=0.01):
    return np.linalg.norm(obs["robot0_eef_pos"] - target_pos) < tol


def scripted_policy(env, obs, max_steps=700):
    """Execute scripted tool hang policy. Returns list of (obs, action) pairs."""
    trajectory = []

    # Get object positions
    frame_pos = obs["frame_pos"].copy()
    frame_quat = obs["frame_quat"].copy()
    tool_pos = obs["tool_pos"].copy()
    tool_quat = obs["tool_quat"].copy()
    stand_mount = get_site_pos(env, "stand_mount_site")
    frame_mount = get_site_pos(env, "frame_mount_site")
    frame_hang = get_site_pos(env, "frame_hang_site")
    frame_intersection = get_site_pos(env, "frame_intersection_site")

    # Gripper pointing down quaternion
    down_quat = np.array([1.0, 0.0, 0.0, 0.0])

    step = 0

    def do_action(action):
        nonlocal obs, step
        raw_obs = {k: obs[k] for k in obs}
        trajectory.append((raw_obs, action.copy()))
        obs, reward, done, info = env.step(action)
        step += 1
        return reward, done

    # === Phase 1: Pick up frame ===

    # 1a: Move above frame
    frame_above = frame_pos.copy()
    frame_above[2] += 0.15
    for _ in range(40):
        if step >= max_steps:
            break
        action = move_to(obs, frame_above, down_quat, gripper_action=-1.0)
        do_action(action)
        if at_target(obs, frame_above, 0.02):
            break

    # 1b: Lower to frame grasp position
    frame_grasp = frame_intersection.copy()
    frame_grasp[2] += 0.01
    for _ in range(40):
        if step >= max_steps:
            break
        action = move_to(obs, frame_grasp, down_quat, gripper_action=-1.0, gain=8.0)
        do_action(action)
        if at_target(obs, frame_grasp, 0.01):
            break

    # 1c: Close gripper
    for _ in range(15):
        if step >= max_steps:
            break
        action = move_to(obs, frame_grasp, down_quat, gripper_action=1.0)
        do_action(action)

    # 1d: Lift frame
    frame_lifted = frame_grasp.copy()
    frame_lifted[2] += 0.15
    for _ in range(30):
        if step >= max_steps:
            break
        action = move_to(obs, frame_lifted, down_quat, gripper_action=1.0)
        do_action(action)
        if at_target(obs, frame_lifted, 0.02):
            break

    # === Phase 2: Insert frame into stand ===

    # 2a: Move above stand mount
    stand_above = stand_mount.copy()
    stand_above[2] += 0.10
    for _ in range(50):
        if step >= max_steps:
            break
        action = move_to(obs, stand_above, down_quat, gripper_action=1.0)
        do_action(action)
        if at_target(obs, stand_above, 0.02):
            break

    # 2b: Insert into stand
    stand_insert = stand_mount.copy()
    stand_insert[2] -= 0.01
    for _ in range(40):
        if step >= max_steps:
            break
        action = move_to(obs, stand_insert, down_quat, gripper_action=1.0, gain=5.0)
        do_action(action)
        if at_target(obs, stand_insert, 0.01):
            break

    # 2c: Release frame
    for _ in range(15):
        if step >= max_steps:
            break
        action = move_to(obs, stand_insert, down_quat, gripper_action=-1.0)
        do_action(action)

    # 2d: Move up away from frame
    retreat = obs["robot0_eef_pos"].copy()
    retreat[2] += 0.15
    for _ in range(30):
        if step >= max_steps:
            break
        action = move_to(obs, retreat, down_quat, gripper_action=-1.0)
        do_action(action)
        if at_target(obs, retreat, 0.02):
            break

    # === Phase 3: Pick up tool ===

    # Re-read tool position (might have shifted)
    tool_pos_now = obs["tool_pos"].copy()

    # 3a: Move above tool
    tool_above = tool_pos_now.copy()
    tool_above[2] += 0.15
    for _ in range(50):
        if step >= max_steps:
            break
        action = move_to(obs, tool_above, down_quat, gripper_action=-1.0)
        do_action(action)
        if at_target(obs, tool_above, 0.02):
            break

    # 3b: Lower to tool
    tool_grasp = tool_pos_now.copy()
    tool_grasp[2] += 0.005
    for _ in range(40):
        if step >= max_steps:
            break
        action = move_to(obs, tool_grasp, down_quat, gripper_action=-1.0, gain=8.0)
        do_action(action)
        if at_target(obs, tool_grasp, 0.01):
            break

    # 3c: Close gripper on tool
    for _ in range(15):
        if step >= max_steps:
            break
        action = move_to(obs, tool_grasp, down_quat, gripper_action=1.0)
        do_action(action)

    # 3d: Lift tool
    tool_lifted = tool_grasp.copy()
    tool_lifted[2] += 0.15
    for _ in range(30):
        if step >= max_steps:
            break
        action = move_to(obs, tool_lifted, down_quat, gripper_action=1.0)
        do_action(action)
        if at_target(obs, tool_lifted, 0.02):
            break

    # === Phase 4: Hang tool on frame ===

    # Re-read frame hang site (frame should now be in stand)
    frame_hang_now = get_site_pos(env, "frame_hang_site")

    # 4a: Move above hang site
    hang_above = frame_hang_now.copy()
    hang_above[2] += 0.10
    for _ in range(50):
        if step >= max_steps:
            break
        action = move_to(obs, hang_above, down_quat, gripper_action=1.0)
        do_action(action)
        if at_target(obs, hang_above, 0.02):
            break

    # 4b: Lower tool onto hook
    hang_target = frame_hang_now.copy()
    hang_target[2] += 0.02
    for _ in range(40):
        if step >= max_steps:
            break
        action = move_to(obs, hang_target, down_quat, gripper_action=1.0, gain=5.0)
        do_action(action)
        if at_target(obs, hang_target, 0.01):
            break

    # 4c: Release tool
    for _ in range(20):
        if step >= max_steps:
            break
        action = move_to(obs, hang_target, down_quat, gripper_action=-1.0)
        do_action(action)

    # 4d: Move away
    retreat2 = obs["robot0_eef_pos"].copy()
    retreat2[2] += 0.15
    for _ in range(30):
        if step >= max_steps:
            break
        action = move_to(obs, retreat2, down_quat, gripper_action=-1.0)
        do_action(action)

    success = env._check_success()
    return trajectory, success, step


def collect_data(n_episodes=200, save_path="data/tool_hang_scripted.hdf5"):
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    env = make_env()

    successes = 0
    all_trajectories = []

    for ep in range(n_episodes * 3):  # try more episodes to get enough successes
        obs = env.reset()
        traj, success, steps = scripted_policy(env, obs)

        if success:
            successes += 1
            all_trajectories.append(traj)
            print(f"Episode {ep}: SUCCESS ({successes}/{n_episodes} collected, {steps} steps)")
            if successes >= n_episodes:
                break
        else:
            print(f"Episode {ep}: FAILED (steps={steps})")

    print(f"\nTotal: {successes}/{ep+1} successful ({100*successes/(ep+1):.1f}%)")

    if successes == 0:
        print("No successful episodes! Check scripted policy.")
        return

    # Save to HDF5 in robomimic format
    obs_keys = ["object", "robot0_eef_pos", "robot0_eef_quat", "robot0_gripper_qpos"]

    with h5py.File(save_path, "w") as f:
        grp = f.create_group("data")
        for i, traj in enumerate(all_trajectories):
            ep_grp = grp.create_group(f"demo_{i}")

            obs_dict = {k: [] for k in obs_keys}
            actions = []

            for obs, act in traj:
                for k in obs_keys:
                    if k == "object":
                        obj = np.concatenate([obs[ok] for ok in sorted(obs.keys())
                                            if ok.startswith(("base_", "frame_", "tool_"))
                                            and not ok.endswith(("robot0_eef_pos", "robot0_eef_quat"))], axis=0)
                        obs_dict[k].append(obj)
                    else:
                        obs_dict[k].append(obs[k])
                actions.append(act)

            obs_grp = ep_grp.create_group("obs")
            for k in obs_keys:
                obs_grp.create_dataset(k, data=np.array(obs_dict[k]))
            ep_grp.create_dataset("actions", data=np.array(actions))
            ep_grp.attrs["num_samples"] = len(traj)

        grp.attrs["total"] = len(all_trajectories)

    print(f"Saved {len(all_trajectories)} demos to {save_path}")


if __name__ == "__main__":
    # First test a single episode
    env = make_env()
    obs = env.reset()
    traj, success, steps = scripted_policy(env, obs)
    print(f"Test episode: success={success}, steps={steps}")

    if success:
        print("Scripted policy works! Collecting full dataset...")
        collect_data(n_episodes=200)
    else:
        print("Scripted policy failed. Need to tune waypoints.")
        # Try a few more seeds
        n_success = 0
        for seed in range(20):
            np.random.seed(seed)
            obs = env.reset()
            _, success, steps = scripted_policy(env, obs)
            if success:
                n_success += 1
            print(f"  Seed {seed}: success={success}, steps={steps}")
        print(f"  Total: {n_success}/20 successful")
