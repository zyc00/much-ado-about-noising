"""Scripted waypoint oracle for robomimic Can (PickPlaceCan) + NFL behavior-noise hook.

Translation-dominant task chosen as the witness venue for the regression-wins
constructions: wrist orientation is HELD FIXED throughout (zero rotation
commands), so one-shot regression policies are not structurally handicapped the
way they are on ToolHang insertion. Phases: hover over can -> descend -> grip ->
lift -> traverse to target bin -> lower -> release -> retreat.

Noise hook identical to the WPM/MP tiers: SKEW_NOISE/_DIST(toyskew|contam|symm|
exp|twopoint|t)/_DIMS/_CLIP/_DF, SKEW_HOLD chunk-holding; eps applied to the
executed AND recorded action (on-dynamics oracle stochasticity). Acceptance rate
printed at the end = the certificate.

Usage:
  MUJOCO_GL=egl python scripts/collect_can_script.py --test_sr 30
  SKEW_NOISE=0.05 SKEW_DIST=toyskew SKEW_HOLD=8 MUJOCO_GL=egl \
    python scripts/collect_can_script.py --n_demos 200 --output data/can_skh5_200.hdf5
"""
import os
os.environ["MUJOCO_GL"] = "egl"
import argparse
import json

import h5py
import numpy as np
import robosuite

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
                "gripper": {"type": "GRIP"},
            }
        },
    },
    "single_object_mode": 2,
    "object_type": "can",
    "robots": "Panda",
}
ENV_ARGS = {"env_name": "PickPlaceCan", "env_version": "1.5.1", "type": 1, "env_kwargs": ENV_KWARGS}
OBS_KEYS_TO_RECORD = ["object", "robot0_eef_pos", "robot0_eef_quat", "robot0_gripper_qpos"]

PG = 12.0
HOVER = 0.10       # hover height above can (m)
LIFT_Z = 1.10      # absolute lift height
EPS = 0.012


def extract_obs(o):
    return {
        "object": np.asarray(o["object-state"], dtype=np.float64),
        "robot0_eef_pos": np.asarray(o["robot0_eef_pos"], dtype=np.float64),
        "robot0_eef_quat": np.asarray(o["robot0_eef_quat"], dtype=np.float64),
        "robot0_gripper_qpos": np.asarray(o["robot0_gripper_qpos"], dtype=np.float64),
    }


def run_episode(seed, record=False, horizon=600):
    env = robosuite.make("PickPlaceCan", horizon=horizon, **ENV_KWARGS)
    np.random.seed(seed)
    o = env.reset()
    sim = env.sim
    init_state = sim.get_state().flatten()
    model_xml = sim.model.get_xml()
    traj = {"actions": [], "states": [], "rewards": [], "dones": [],
            "obs": {k: [] for k in OBS_KEYS_TO_RECORD}}

    _sk = float(os.environ.get("SKEW_NOISE", "0"))
    _sd = os.environ.get("SKEW_DIST", "exp")
    _sp = float(os.environ.get("SKEW_P", "0.15"))
    _snd = int(os.environ.get("SKEW_DIMS", "3"))  # translation dims by default
    _scl = float(os.environ.get("SKEW_CLIP", "0"))
    _sdf = float(os.environ.get("SKEW_DF", "2.0"))
    _shold = int(os.environ.get("SKEW_HOLD", "1"))
    _rng = np.random.RandomState(seed + 777)
    _hst = {"eps": None, "k": 0}
    st = {"o": o, "done": False, "succ": False}

    def step(a):
        a = np.clip(np.asarray(a, dtype=np.float64), -1, 1)
        if _sk > 0:
            if _shold > 1 and _hst["eps"] is not None and _hst["k"] % _shold != 0:
                eps = _hst["eps"]
            else:
                if _sd == "t":
                    eps = _sk * _rng.standard_t(_sdf, _snd)
                elif _sd == "twopoint":
                    b = (_rng.rand(_snd) < _sp).astype(np.float64)
                    eps = _sk * (b - _sp)
                elif _sd == "toyskew":
                    b = (_rng.rand(_snd) < 0.2).astype(np.float64)
                    eps = _sk * (5.0 * b - 1.0)
                elif _sd == "contam":
                    u = _rng.rand(_snd)
                    eps = np.where(u < 0.7, 0.0, np.where(u < 0.9, 2.0 * _sk, 16.0 * _sk))
                elif _sd == "symm":
                    eps = _sk * np.sign(_rng.rand(_snd) - 0.5)
                else:
                    eps = _sk * (_rng.exponential(1.0, _snd) - 1.0)
                if _scl > 0:
                    eps = np.clip(eps, -_scl, _scl)
                _hst["eps"] = eps
            _hst["k"] += 1
            a[:_snd] = np.clip(a[:_snd] + eps, -1, 1)
        if record:
            traj["actions"].append(a.copy())
            traj["states"].append(sim.get_state().flatten().copy())
            for k, v in extract_obs(st["o"]).items():
                traj["obs"][k].append(v)
        st["o"], r, d, _ = env.step(a)
        if record:
            traj["rewards"].append(float(r))
            traj["dones"].append(int(d))
        if env._check_success():
            st["succ"] = True

    def goto(target, grip, max_steps=120, tol=EPS):
        for _ in range(max_steps):
            eef = st["o"]["robot0_eef_pos"]
            dp = np.clip((np.asarray(target) - eef) * PG, -1, 1)
            step(np.concatenate([dp, [0.0, 0.0, 0.0], [grip]]))
            if np.linalg.norm(st["o"]["robot0_eef_pos"] - np.asarray(target)) < tol:
                return True
        return False

    def can_pos():
        # object-state layout: can pos is the first 3 dims for single_object_mode=2
        return np.asarray(st["o"]["object-state"][:3], dtype=np.float64)

    try:
        cp = can_pos()
        goto([cp[0], cp[1], cp[2] + HOVER], -1)
        cp = can_pos()
        goto([cp[0], cp[1], cp[2] + 0.005], -1, tol=0.008)
        for _ in range(8):
            step([0, 0, 0, 0, 0, 0, 1])          # close
        goto([st["o"]["robot0_eef_pos"][0], st["o"]["robot0_eef_pos"][1], LIFT_Z], 1)
        tb = env.target_bin_placements[env.object_id]  # (x, y, z) of target bin
        goto([tb[0], tb[1], LIFT_Z], 1)
        goto([tb[0], tb[1], tb[2] + 0.10], 1)
        for _ in range(8):
            step([0, 0, 0, 0, 0, 0, -1])         # release
        goto([tb[0], tb[1], LIFT_Z], -1, max_steps=40)
        for _ in range(10):                        # settle so success registers
            step([0, 0, 0, 0, 0, 0, -1])
    except Exception as e:
        env.close()
        return False, traj, init_state, model_xml
    succ = bool(st["succ"] or env._check_success())
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
            grp.attrs["model_file"] = demo["model_file"]
            grp.attrs["seed"] = demo["seed"]
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
    ap.add_argument("--output", type=str, default="data/can_script_200.hdf5")
    ap.add_argument("--test_sr", type=int, default=0)
    ap.add_argument("--start_seed", type=int, default=600000)
    args = ap.parse_args()

    if args.test_sr > 0:
        succ = 0
        for s in range(args.test_sr):
            ok, _, _, _ = run_episode(args.start_seed + s, record=False)
            succ += int(ok)
            print(f"seed {args.start_seed + s}: {ok}")
        print(f"\nCAN-SCRIPT SR: {succ}/{args.test_sr}")
        return

    from tqdm import tqdm
    demos = []
    seed = args.start_seed
    attempts = 0
    pbar = tqdm(total=args.n_demos, desc="can-collect")
    while len(demos) < args.n_demos:
        ok, traj, ist, xml = run_episode(seed, record=True)
        attempts += 1
        if ok and len(traj["actions"]) > 0:
            traj["model_file"] = xml
            traj["seed"] = seed
            demos.append(traj)
            pbar.update(1)
        seed += 1
    pbar.close()
    print(f"can-script collected {len(demos)} from {attempts} attempts "
          f"({100 * len(demos) / attempts:.0f}%), seeds {args.start_seed}..{seed - 1}")
    save_demos(demos, args.output)
    nn = sum(len(d["actions"]) for d in demos)
    print(f"Saved {args.output}: {nn} samples, avg {nn / len(demos):.0f}/demo")


if __name__ == "__main__":
    main()
