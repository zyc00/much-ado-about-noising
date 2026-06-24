"""Collect INSERTION-ONLY demos by resetting to each source demo's align-done
state and recording the insertion script segment (align -> 60deg tilt push ->
release). Isolates the insertion sub-skill.

Reset fidelity: set_state_from_flattened + sim.forward() + controller sync
(arm.update(); arm.reset_goal()) to avoid the OSC stale-goal transient; then a
few grip-hold settle steps so the grasp contact force balances the frame weight.

Usage:
  MUJOCO_GL=egl python scripts/collect_insertion_only.py \
     --src data/tool_hang_clean_20000.hdf5 --n_demos 20000 \
     --output data/tool_hang_insertion_20000.hdf5
"""
import os
os.environ["MUJOCO_GL"] = "egl"
import argparse, json
import numpy as np
import h5py
import robosuite
import robosuite.utils.transform_utils as T
from scipy.spatial.transform import Rotation as R
from tqdm import tqdm
import sys
sys.path.insert(0, "scripts")
from collect_tool_hang_demos import ENV_KWARGS, ENV_ARGS, OBS_KEYS_TO_RECORD, extract_obs

SETTLE = 5  # grip-hold steps after reset so grasp contact settles


def align_done_frame(d):
    a = d["actions"][:]; ez = d["obs/robot0_eef_pos"][:, 2]; g = a[:, 6]
    cl = [t for t in range(1, len(g)) if g[t-1] < 0 and g[t] >= 0]
    op = [t for t in range(1, len(g)) if g[t-1] >= 0 and g[t] < 0]
    if not cl or not op:
        return None
    c1, o1 = cl[0], op[0]
    return c1 + int(np.argmax(ez[c1:o1]))


def run_one(env, src_state, src_seed):
    """Reset to align-done state, sync controller, settle, run insertion script.
    Returns (success, traj) where traj records the insertion segment."""
    np.random.seed(src_seed)
    env.reset()
    sim = env.sim
    sim.set_state_from_flattened(src_state)
    sim.forward()
    arm = env.robots[0].composite_controller.part_controllers["right"]
    arm.update(); arm.reset_goal()

    traj = {"actions": [], "states": [], "rewards": [], "dones": [],
            "obs": {k: [] for k in OBS_KEYS_TO_RECORD}}
    o = [env._get_observations(force_update=True)]

    def obs():
        return o[0]

    def gs(n):
        return sim.data.site_xpos[sim.model.site_name2id(n)].copy()

    def step(a, rec=True):
        a = np.clip(a, -1, 1).astype(np.float64)
        if rec:
            traj["actions"].append(a.copy())
            traj["states"].append(sim.get_state().flatten().copy())
            for k, v in extract_obs(o[0]).items():
                traj["obs"][k].append(v)
        o[0], r, dn, _ = env.step(a)
        if rec:
            traj["rewards"].append(float(r)); traj["dones"].append(int(dn))

    # settle (recorded: these grip-hold steps are part of the insertion demo start)
    for _ in range(SETTLE):
        step(np.array([0, 0, 0, 0, 0, 0, 1.0]))

    def cfq():
        em = T.quat2mat(obs()["robot0_eef_quat"])
        fm = gs("frame_mount_site"); fi = gs("frame_intersection_site"); fh = gs("frame_hang_site")
        nw = (fm - fi) / np.linalg.norm(fm - fi); hw = (fh - fi) / np.linalg.norm(fh - fi)
        ne = em.T @ nw; he = em.T @ hw; neu = ne / np.linalg.norm(ne)
        hep = he - np.dot(he, neu) * neu; heu = hep / np.linalg.norm(hep)
        ed = np.column_stack([neu, heu, np.cross(neu, heu)])
        nt = np.array([0, 0, -1.]); ht = np.array([0, -1, 0.])
        return T.mat2quat(np.column_stack([nt, ht, np.cross(nt, ht)]) @ np.linalg.inv(ed))

    hc = np.mean([sim.data.geom_xpos[sim.model.geom_name2id(f"stand_wall{i}")] for i in range(4)], axis=0)
    sm = gs("stand_mount_site"); hc[2] = sm[2]
    tq2 = cfq()
    # align above hole
    for _ in range(60):
        eef = obs()["robot0_eef_pos"]; etn = gs("frame_mount_site") - eef
        tf = hc - etn; tf[2] = sm[2] + 0.05 - etn[2]; dp = np.clip((tf - eef) * 15, -1, 1)
        ea = T.quat2axisangle(T.mat2quat(T.quat2mat(tq2) @ T.quat2mat(obs()["robot0_eef_quat"]).T))
        step(np.concatenate([dp, np.clip(ea * 8, -1, 1), [1]]))
        if np.linalg.norm((gs("frame_mount_site") - hc)[:2]) < 0.002:
            break
    # 60deg tilt push
    m = T.quat2mat(tq2); ax = np.cross(m[:, 2], [0, 0, -1.]); ax /= np.linalg.norm(ax)
    tqt = T.mat2quat(R.from_rotvec(ax * np.radians(60)).as_matrix() @ m)
    for i in range(120):
        fr = (i + 1) / 120; dot = np.clip(abs(np.dot(tq2, tqt)), -1, 1); th = np.arccos(dot)
        tu = -tqt if np.dot(tq2, tqt) < 0 else tqt
        wp = tu if th < 1e-6 else (np.sin((1-fr)*th)/np.sin(th))*tq2 + (np.sin(fr*th)/np.sin(th))*tu
        wp /= np.linalg.norm(wp)
        ea = T.quat2axisangle(T.mat2quat(T.quat2mat(wp) @ T.quat2mat(obs()["robot0_eef_quat"]).T))
        step(np.concatenate([[0, 0, -1], np.clip(ea * 8, -1, 1), [1]]))
        if obs()["robot0_eef_pos"][2] <= 1.0:
            break
    # release
    for _ in range(20):
        step(np.array([0, 0, 0, 0, 0, 0, -1]))
    return env._check_frame_assembled(), traj


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--src", default="data/tool_hang_clean_20000.hdf5")
    ap.add_argument("--n_demos", type=int, default=20000)
    ap.add_argument("--output", default="data/tool_hang_insertion_20000.hdf5")
    ap.add_argument("--reset_frame", default="align_done")  # align_done | c1 (expert grasp)
    args = ap.parse_args()

    sf = h5py.File(args.src, "r")
    src_keys = list(sf["data"].keys())
    env = robosuite.make("ToolHang", horizon=4000, **ENV_KWARGS)

    demos = []
    attempts = 0
    pbar = tqdm(total=args.n_demos, desc="insertion-only")
    ki = 0
    while len(demos) < args.n_demos and ki < len(src_keys):
        d = sf["data/" + src_keys[ki]]; ki += 1
        af = align_done_frame(d)
        if af is None:
            continue
        if args.reset_frame in ("c1", "c1lift"):  # expert grasp (+lift) frame
            g = d["actions"][:, 6]
            c1 = next((t for t in range(1, len(g)) if g[t-1] < 0 and g[t] >= 0), None)
            if c1 is None:
                continue
            af = c1 + 20 if args.reset_frame == "c1lift" else c1  # +20 == grasp specialist endpoint
        st = d["states"][af]; sd = int(d.attrs.get("seed", 0))
        model_xml = d.attrs["model_file"]
        attempts += 1
        ok, traj = run_one(env, st, sd)
        if ok and len(traj["actions"]) > 0:
            traj["model_file"] = model_xml; traj["seed"] = sd
            demos.append(traj); pbar.update(1)
    pbar.close()
    sf.close()
    print(f"Collected {len(demos)} insertion demos from {attempts} attempts ({100*len(demos)/max(attempts,1):.0f}%)")

    os.makedirs(os.path.dirname(args.output) or ".", exist_ok=True)
    with h5py.File(args.output, "w") as f:
        data = f.create_group("data")
        data.attrs["env_args"] = json.dumps(ENV_ARGS)
        data.attrs["total"] = sum(len(x["actions"]) for x in demos)
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
    n = sum(len(x["actions"]) for x in demos)
    print(f"Saved {args.output}: {n} samples, avg {n/max(len(demos),1):.0f}/demo")


if __name__ == "__main__":
    main()
