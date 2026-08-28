"""Scripted oracle for BigCubeLift: pick a cube sized to BARELY fit the gripper.

Witness task for the regression-wins NFL constructions (user-specified):
difficulty = lateral grasp precision (cube width ~ gripper opening), NO wrist
rotation at any point. A lateral action bias (skewed noise estimand) directly
produces grasp failure, so loss-estimand differences couple maximally to SR
while the task stays regression-friendly (pure translation).

BigCubeLift subclasses robosuite Lift with a configurable cube half-size
(CUBE_HALF env var, default 0.033 m -> cube width 66 mm vs Panda max opening
~80 mm; ~7 mm clearance per side at perfect centering).

Noise hook identical to WPM/MP tiers (SKEW_* env family, SKEW_HOLD chunking),
applied to executed AND recorded actions. Acceptance printed = certificate.

Usage:
  MUJOCO_GL=egl python scripts/collect_cube_script.py --test_sr 30
  SKEW_NOISE=0.05 SKEW_DIST=toyskew SKEW_HOLD=8 MUJOCO_GL=egl \
    python scripts/collect_cube_script.py --n_demos 200 --output data/cube_skh5_200.hdf5
"""
import os
os.environ["MUJOCO_GL"] = "egl"
import argparse
import json

import h5py
import numpy as np
import robosuite
from robosuite.environments.manipulation.lift import Lift

CUBE_HALF = float(os.environ.get("CUBE_HALF", "0.033"))


class BigCubeLift(Lift):
    """Lift with a large, gripper-width cube (uniform size, no randomization)."""

    def _load_model(self):
        super()._load_model()
        # Resize the cube geoms/sites in the generated model directly.
        cube = self.cube
        for g in cube.get_obj().findall(".//geom"):
            g.set("size", f"{CUBE_HALF} {CUBE_HALF} {CUBE_HALF}")


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
    "robots": "Panda",
}
ENV_ARGS = {"env_name": "BigCubeLift", "env_version": "1.5.1", "type": 1,
            "env_kwargs": {**ENV_KWARGS, "cube_half": CUBE_HALF}}
OBS_KEYS_TO_RECORD = ["object", "robot0_eef_pos", "robot0_eef_quat", "robot0_gripper_qpos"]

PG = 12.0
HOVER = 0.08
LIFT_Z_REL = 0.15
EPS = 0.010


def extract_obs(o):
    return {
        "object": np.asarray(o["object-state"], dtype=np.float64),
        "robot0_eef_pos": np.asarray(o["robot0_eef_pos"], dtype=np.float64),
        "robot0_eef_quat": np.asarray(o["robot0_eef_quat"], dtype=np.float64),
        "robot0_gripper_qpos": np.asarray(o["robot0_gripper_qpos"], dtype=np.float64),
    }


def run_episode(seed, record=False, horizon=400):
    env = robosuite.make("BigCubeLift", horizon=horizon, **ENV_KWARGS)
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
    _snd = int(os.environ.get("SKEW_DIMS", "3"))
    _scl = float(os.environ.get("SKEW_CLIP", "0"))
    _sdf = float(os.environ.get("SKEW_DF", "2.0"))
    _shold = int(os.environ.get("SKEW_HOLD", "1"))
    # SKEW_PHASE=all (default): inject everywhere. SKEW_PHASE=reach: inject only
    # during the free-space reach/align phases; the pick (descend+grasp+lift) is
    # clean, so the pre-grasp entry state stays matched to the clean cell (the
    # align goto still exits at its tolerance) and the precision endgame labels
    # are uncorrupted.
    _sphase = os.environ.get("SKEW_PHASE", "all")
    _rng = np.random.RandomState(seed + 777)
    _hst = {"eps": None, "k": 0}
    _gate = {"on": True}
    st = {"o": o, "succ": False}

    def step(a):
        a = np.clip(np.asarray(a, dtype=np.float64), -1, 1)
        if _sk > 0 and _gate["on"]:
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

    def goto(target, grip, max_steps=100, tol=EPS):
        for _ in range(max_steps):
            eef = st["o"]["robot0_eef_pos"]
            dp = np.clip((np.asarray(target) - eef) * PG, -1, 1)
            step(np.concatenate([dp, [0.0, 0.0, 0.0], [grip]]))
            if np.linalg.norm(st["o"]["robot0_eef_pos"] - np.asarray(target)) < tol:
                return True
        return False

    def cube_pos():
        return np.asarray(st["o"]["object-state"][:3], dtype=np.float64)

    try:
        cp = cube_pos()
        goto([cp[0], cp[1], cp[2] + HOVER], -1)
        cp = cube_pos()
        # tight tolerance on the lateral alignment before descending
        goto([cp[0], cp[1], cp[2] + HOVER], -1, tol=0.004)
        if _sphase == "reach":
            _gate["on"] = False  # pick phase (descend+grasp+lift) stays clean
        goto([cp[0], cp[1], cp[2] + 0.002], -1, tol=0.006)
        for _ in range(10):
            step([0, 0, 0, 0, 0, 0, 1])
        base = st["o"]["robot0_eef_pos"]
        goto([base[0], base[1], base[2] + LIFT_Z_REL], 1, max_steps=60)
        for _ in range(10):
            step([0, 0, 0, 0, 0, 0, 1])
    except Exception:
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
    ap.add_argument("--output", type=str, default="data/cube_script_200.hdf5")
    ap.add_argument("--test_sr", type=int, default=0)
    ap.add_argument("--start_seed", type=int, default=700000)
    args = ap.parse_args()

    if args.test_sr > 0:
        succ = 0
        for s in range(args.test_sr):
            ok, _, _, _ = run_episode(args.start_seed + s, record=False)
            succ += int(ok)
            print(f"seed {args.start_seed + s}: {ok}")
        print(f"\nCUBE-SCRIPT SR (CUBE_HALF={CUBE_HALF}): {succ}/{args.test_sr}")
        return

    from tqdm import tqdm
    demos = []
    seed = args.start_seed
    attempts = 0
    pbar = tqdm(total=args.n_demos, desc="cube-collect")
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
    print(f"cube-script collected {len(demos)} from {attempts} attempts "
          f"({100 * len(demos) / attempts:.0f}%), seeds {args.start_seed}..{seed - 1}, "
          f"CUBE_HALF={CUBE_HALF}")
    save_demos(demos, args.output)
    nn = sum(len(d["actions"]) for d in demos)
    print(f"Saved {args.output}: {nn} samples, avg {nn / len(demos):.0f}/demo")


if __name__ == "__main__":
    main()
