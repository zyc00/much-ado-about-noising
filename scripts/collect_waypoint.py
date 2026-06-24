"""Waypoint-tracking DART collection (real nominal trajectory, NOT subsampled).

User's design:
  1. RECORD (no noise): run the scripted expert once, record EVERY step's
     waypoint = (eef_pos, eef_quat, grip). Not subsampled — every nominal step
     is a waypoint, so contact "sustained pressing" (many steps, eef barely
     moving) is preserved as a dense waypoint sequence.
  2. TRACK (with noise): for each waypoint in order, PD-control toward it; inject
     action noise on EXECUTION (pushes off the tube); the PD naturally pulls back
     to the current waypoint, then advance to the next. RECORD the clean
     toward-waypoint action at the (perturbed) obs -> recovery coverage that, by
     construction, always points back onto the nominal tube.

First validate: with noise=0, tracking reproduces the nominal SR.

Usage:
  MUJOCO_GL=egl python scripts/collect_waypoint.py --test_sr 15 --noise 0.0
  MUJOCO_GL=egl python scripts/collect_waypoint.py --n_demos 200 --noise 0.05 \
     --output data/tool_hang_wp_200.hdf5
"""
import os
os.environ["MUJOCO_GL"] = "egl"
import argparse, json
import numpy as np
import h5py
import robosuite
import robosuite.utils.transform_utils as T
from tqdm import tqdm
import sys
sys.path.insert(0, "scripts")
from scripted_tool_hang_v2 import run_episode, ENV_KWARGS, ENV_ARGS, OBS_KEYS_TO_RECORD, extract_obs

WP_EPS = 0.01     # arrival tolerance (m)
WP_STEPCAP = 25   # max PD steps per waypoint (noise/contact safety)
PG = 12.0         # position gain
OG = 6.0          # orientation gain


def nominal_waypoints(seed):
    """Run the clean scripted expert; return per-step (pos, quat, grip) waypoints."""
    ok, traj, _, _ = run_episode(seed, record=True)
    if not ok:
        return None
    P = np.array(traj["obs"]["robot0_eef_pos"])
    Q = np.array(traj["obs"]["robot0_eef_quat"])
    G = np.array(traj["actions"])[:, 6]
    return [(P[t].copy(), Q[t].copy(), float(G[t])) for t in range(len(P))]


def track(seed, wps, noise=0.0, noise_seed=0, record=False, horizon=4000):
    env = robosuite.make("ToolHang", horizon=horizon, **ENV_KWARGS)
    np.random.seed(seed)
    o = env.reset()
    for _ in range(10):  # settle (not recorded)
        o, _, _, _ = env.step(np.zeros(7))
    rng = np.random.RandomState(noise_seed)
    sim = env.sim
    traj = {"actions": [], "states": [], "rewards": [], "dones": [],
            "obs": {k: [] for k in OBS_KEYS_TO_RECORD}}
    init_state = sim.get_state().flatten(); model_xml = sim.model.get_xml()

    def step(a):
        a = np.clip(a, -1, 1).astype(np.float64)
        if record:
            traj["actions"].append(a.copy())  # CLEAN toward-waypoint action
            traj["states"].append(sim.get_state().flatten().copy())
            for k, v in extract_obs(o[0]).items():
                traj["obs"][k].append(v)
        ax = a.copy()
        if noise > 0:
            ax[:3] = np.clip(a[:3] + rng.randn(3) * noise, -1, 1)
        o[0], r, d, _ = env.step(ax)
        if record:
            traj["rewards"].append(float(r)); traj["dones"].append(int(d))

    o = [o]
    for (wp_pos, wp_quat, wp_grip) in wps:
        for _ in range(WP_STEPCAP):
            eef = o[0]["robot0_eef_pos"]
            dp = np.clip((wp_pos - eef) * PG, -1, 1)
            em = T.quat2mat(o[0]["robot0_eef_quat"]); tm = T.quat2mat(wp_quat)
            ea = T.quat2axisangle(T.mat2quat(tm @ em.T))
            do = np.clip(ea * OG, -1, 1)
            step(np.concatenate([dp, do, [wp_grip]]))
            if np.linalg.norm(o[0]["robot0_eef_pos"] - wp_pos) < WP_EPS:
                break
    succ = env._check_success()
    env.close()
    return succ, traj, init_state, model_xml


def save_demos(demos, output_path):
    os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)
    with h5py.File(output_path, "w") as f:
        data = f.create_group("data")
        data.attrs["env_args"] = json.dumps(ENV_ARGS)
        data.attrs["total"] = sum(len(d["actions"]) for d in demos)
        for i, demo in enumerate(demos):
            grp = data.create_group(f"demo_{i}")
            grp.attrs["num_samples"] = len(demo["actions"])
            grp.attrs["model_file"] = demo["model_file"]; grp.attrs["seed"] = demo["seed"]
            grp.create_dataset("actions", data=np.array(demo["actions"], dtype=np.float64))
            grp.create_dataset("states", data=np.array(demo["states"], dtype=np.float64))
            grp.create_dataset("rewards", data=np.array(demo["rewards"], dtype=np.float64))
            grp.create_dataset("dones", data=np.array(demo["dones"], dtype=np.int64))
            og = grp.create_group("obs")
            for k in OBS_KEYS_TO_RECORD:
                og.create_dataset(k, data=np.array(demo["obs"][k]))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n_demos", type=int, default=200)
    ap.add_argument("--output", type=str, default="data/tool_hang_wp_200.hdf5")
    ap.add_argument("--noise", type=float, default=0.0)
    ap.add_argument("--test_sr", type=int, default=0)
    args = ap.parse_args()

    if args.test_sr > 0:
        succ = 0; n = 0
        for s in range(args.test_sr):
            wps = nominal_waypoints(s)
            if wps is None:
                print(f"seed {s}: nominal FAIL skip"); continue
            ok, _, _, _ = track(s, wps, noise=args.noise, noise_seed=0)
            succ += int(ok); n += 1
            print(f"seed {s}: nwp={len(wps)} track(noise={args.noise})={ok}")
        print(f"\nWAYPOINT-TRACK SR (noise={args.noise}): {succ}/{n}")
        return

    demos = []; seed = 0; attempts = 0
    pbar = tqdm(total=args.n_demos, desc="wp-collect")
    while len(demos) < args.n_demos:
        wps = nominal_waypoints(seed)
        if wps is not None:
            ok, t2, ist, xml = track(seed, wps, noise=args.noise, noise_seed=seed, record=True)
            attempts += 1
            if ok and len(t2["actions"]) > 0:
                t2["model_file"] = xml; t2["seed"] = seed
                demos.append(t2); pbar.update(1)
        seed += 1
    pbar.close()
    print(f"Collected {len(demos)} from {attempts} ({100*len(demos)/attempts:.0f}%)")
    save_demos(demos, args.output)
    nn = sum(len(d["actions"]) for d in demos)
    print(f"Saved {args.output}: {nn} samples, avg {nn/len(demos):.0f}/demo")


if __name__ == "__main__":
    main()
