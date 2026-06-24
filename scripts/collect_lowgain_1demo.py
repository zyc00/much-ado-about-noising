"""Collect ONE low-gain (smooth, non-bang-bang) scripted demo for seed 0.

Same task logic / waypoints as the high-gain scripted policy, but every
pos/ori action component is capped to +/-ACAP so actions never saturate to the
bang-bang extreme. Loop budgets are scaled up so the (slower) motion still
completes. Saves a 1-demo hdf5 in robomimic format + the init state.

Usage:
  MUJOCO_GL=egl python scripts/collect_lowgain_1demo.py --acap 0.3 --out data/tool_hang_lowgain_1demo.hdf5
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

import sys
sys.path.insert(0, "scripts")
from collect_tool_hang_demos import ENV_KWARGS, ENV_ARGS, OBS_KEYS_TO_RECORD, extract_obs


def collect(seed, acap, out):
    env = robosuite.make("ToolHang", horizon=8000, **ENV_KWARGS)
    np.random.seed(seed)
    obs_h = [env.reset()]
    sim = env.sim
    init_state = sim.get_state().flatten()
    model_xml = sim.model.get_xml()
    traj = {"actions": [], "states": [], "rewards": [], "dones": [],
            "obs": {k: [] for k in OBS_KEYS_TO_RECORD}}

    # scale factor for step budgets (slower motion needs more steps)
    sc = max(1.0, 1.0 / (acap / 1.0))  # e.g. acap=0.3 -> ~3.3x

    def do_step(a):
        a = np.array(a, dtype=np.float64)
        # cap pos (0:3) and ori (3:6) to +/-acap; gripper (6) untouched
        a[:6] = np.clip(a[:6], -acap, acap)
        a = np.clip(a, -1, 1)
        traj["actions"].append(a.copy())
        traj["states"].append(sim.get_state().flatten().copy())
        for k, v in extract_obs(obs_h[0]).items():
            traj["obs"][k].append(v)
        obs_h[0], r, d, _ = env.step(a)
        traj["rewards"].append(float(r)); traj["dones"].append(int(d))

    def obs():
        return obs_h[0]

    def move_ori(tp, tq, g, steps=80, pg=10, og=3, pt=0.005, ot=0.03):
        steps = int(steps * sc)
        for i in range(steps):
            o = obs(); eef = o["robot0_eef_pos"]
            dp = (tp - eef) * pg
            em = T.quat2mat(o["robot0_eef_quat"]); tm = T.quat2mat(tq)
            ea = T.quat2axisangle(T.mat2quat(tm @ em.T))
            do = ea * og
            do_step(np.concatenate([dp, do, [g]]))
            if np.linalg.norm(eef - tp) < pt and np.linalg.norm(ea) < ot:
                return i
        return steps

    def cfq():
        o = obs(); em = T.quat2mat(o["robot0_eef_quat"])
        fm = gs("frame_mount_site"); fi = gs("frame_intersection_site"); fh = gs("frame_hang_site")
        nw = (fm - fi) / np.linalg.norm(fm - fi); hw = (fh - fi) / np.linalg.norm(fh - fi)
        ne = em.T @ nw; he = em.T @ hw
        neu = ne / np.linalg.norm(ne); hep = he - np.dot(he, neu) * neu; heu = hep / np.linalg.norm(hep)
        te = np.cross(neu, heu); ed = np.column_stack([neu, heu, te])
        nt = np.array([0, 0, -1.0]); ht = np.array([0, -1, 0.0]); tt = np.cross(nt, ht)
        wt = np.column_stack([nt, ht, tt])
        return T.mat2quat(wt @ np.linalg.inv(ed))

    def gs(name):
        return sim.data.site_xpos[sim.model.site_name2id(name)].copy()

    def push_const(pos3, g, n):
        # constant-velocity translation segment (zero ori delta), run n*sc steps
        v = np.concatenate([np.array(pos3, dtype=np.float64), np.zeros(3)])
        for _ in range(int(n * sc)):
            do_step(np.concatenate([v, [g]]))

    # === Phase 1 ===
    for _ in range(10):
        do_step(np.zeros(7))
    fp = obs()["frame_pos"].copy()
    fmw = T.quat2mat(obs()["frame_quat"].copy())
    fx = fmw[:, 0].copy(); fx[2] = 0; fx = fx / np.linalg.norm(fx)
    fa = np.arctan2(fx[1], fx[0]); fmount = gs("frame_mount_site")
    nd = fmount - fp; nd[2] = 0; nd = nd / np.linalg.norm(nd)
    hc = np.zeros(3)
    for i in range(4):
        hc += sim.data.geom_xpos[sim.model.geom_name2id(f"stand_wall{i}")]
    hc /= 4; sm = gs("stand_mount_site"); hc[2] = sm[2]
    pa = fa - np.pi / 2
    gq = T.mat2quat(np.column_stack([np.array([np.cos(pa), np.sin(pa), 0]),
        np.cross(np.array([0, 0, -1.0]), np.array([np.cos(pa), np.sin(pa), 0])),
        np.array([0, 0, -1.0])]))
    gp = fp + nd * 0.02; gp[2] = fp[2]
    ab = gp.copy(); ab[2] += 0.10; move_ori(ab, gq, -1, steps=40, pg=12)
    dn = gp.copy(); dn[2] -= 0.005; move_ori(dn, gq, -1, steps=50, pg=8, og=5)
    for _ in range(int(15 * sc)):
        o = obs(); dp = (dn - o["robot0_eef_pos"]) * 3
        ea = T.quat2axisangle(T.mat2quat(T.quat2mat(gq) @ T.quat2mat(o["robot0_eef_quat"]).T))
        do_step(np.concatenate([dp, ea * 2, [1]]))
    lf = obs()["robot0_eef_pos"].copy(); lf[2] += 0.25; move_ori(lf, gq, 1, steps=30, pg=15, og=3)
    if obs()["frame_pos"][2] - fp[2] < 0.05:
        return False, None
    tq = cfq(); move_ori(obs()["robot0_eef_pos"].copy(), tq, 1, steps=80, pg=4, og=8, ot=0.02)
    tq2 = cfq(); move_ori(obs()["robot0_eef_pos"].copy(), tq2, 1, steps=40, pg=4, og=10, ot=0.01)
    for _ in range(3):
        etn = gs("frame_mount_site") - obs()["robot0_eef_pos"]
        tf = hc - etn; tf[2] = sm[2] + 0.05 - etn[2]
        move_ori(tf, tq2, 1, steps=35, pg=20, og=10, pt=0.002, ot=0.015)
    # tilt push down (capped)
    tq2_mat = T.quat2mat(tq2); cz = tq2_mat[:, 2]; ax = np.cross(cz, np.array([0, 0, -1.0])); ax /= np.linalg.norm(ax)
    tilted = T.mat2quat(R.from_rotvec(ax * np.radians(60)).as_matrix() @ tq2_mat)
    n_push = int(120 * sc)
    for i in range(n_push):
        frac = (i + 1) / n_push
        dot = np.dot(tq2, tilted); tu = -tilted if dot < 0 else tilted; dot = abs(dot)
        th_ = np.arccos(np.clip(dot, -1, 1))
        wpq = tu if th_ < 1e-6 else (np.sin((1 - frac) * th_) / np.sin(th_)) * tq2 + (np.sin(frac * th_) / np.sin(th_)) * tu
        wpq = wpq / np.linalg.norm(wpq)
        em = T.quat2mat(obs()["robot0_eef_quat"]); ea = T.quat2axisangle(T.mat2quat(T.quat2mat(wpq) @ em.T))
        do_step(np.concatenate([[0, 0, -acap], ea * 8, [1]]))
        if obs()["robot0_eef_pos"][2] <= 1.0:
            break
    push_const([0, 0, 0], -1, 20)          # release in place
    push_const([acap, 0, 0], -1, 40)       # retreat +x (capped)
    if not env._check_frame_assembled():
        return False, None

    # === Phase 2 ===
    tp = obs()["tool_pos"].copy(); target = tp.copy(); target[2] += 0.05
    for i in range(int(120 * sc)):
        eef = obs()["robot0_eef_pos"]; dp = (target - eef) * 12
        do_step(np.concatenate([dp, [0, 0, 0, -1]]))
        if np.linalg.norm(eef - target) < 0.01:
            break
    tool_mat = T.quat2mat(obs()["tool_quat"].copy()); tx = tool_mat[:, 0].copy(); tx[2] = 0; tx /= np.linalg.norm(tx)
    ta = np.arctan2(tx[1], tx[0]); xe = np.array([np.cos(ta), np.sin(ta), 0]); ye = np.cross(np.array([0, 0, -1.0]), xe)
    tgq = T.mat2quat(np.column_stack([xe, ye, np.array([0, 0, -1.0])]))
    move_ori(obs()["robot0_eef_pos"].copy(), tgq, -1, steps=100, pg=8, og=5)
    tp2 = obs()["tool_pos"].copy(); ab2 = tp2.copy(); ab2[2] += 0.05; move_ori(ab2, tgq, -1, steps=40, pg=15, og=5)
    tp3 = obs()["tool_pos"].copy(); dn2 = tp3.copy(); dn2[2] -= 0.005; move_ori(dn2, tgq, -1, steps=50, pg=8, og=5)
    for _ in range(int(15 * sc)):
        o = obs(); dp = (dn2 - o["robot0_eef_pos"]) * 3
        ea = T.quat2axisangle(T.mat2quat(T.quat2mat(tgq) @ T.quat2mat(o["robot0_eef_quat"]).T))
        do_step(np.concatenate([dp, ea * 2, [1]]))
    lift = obs()["robot0_eef_pos"].copy(); lift[2] += 0.20; move_ori(lift, tgq, 1, steps=40, pg=12, og=3)
    if obs()["tool_pos"][2] - tp3[2] < 0.05:
        return False, None
    fh = gs("frame_hang_site"); th = gs("tool_hole1_center"); eth = th - obs()["robot0_eef_pos"]
    ta2 = fh - eth; ta2[2] = fh[2] + 0.10; move_ori(ta2, tgq, 1, steps=60, pg=12, og=5)
    for _ in range(3):
        th = gs("tool_hole1_center"); etn = th - obs()["robot0_eef_pos"]
        tf = fh - etn; tf[2] = fh[2] + 0.05; move_ori(tf, tgq, 1, steps=30, pg=20, og=8, pt=0.002, ot=0.01)
    for _ in range(2):
        th = gs("tool_hole1_center"); etn = th - obs()["robot0_eef_pos"]
        tf = fh - etn; tf[2] = fh[2] - 0.02; move_ori(tf, tgq, 1, steps=30, pg=5, og=3)
    push_const([0, 0, -acap], 1, 20)       # seat with grip
    push_const([0, 0, 0], -1, 15)          # release
    push_const([0, 0, -acap], -1, 40)      # push down to seat fully
    push_const([0, 0, 0], -1, 10)

    success = env._check_success()
    if not success:
        return False, None

    return True, (traj, init_state, model_xml)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--acap", type=float, default=0.3)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", default="data/tool_hang_lowgain_1demo.hdf5")
    args = ap.parse_args()

    ok, res = collect(args.seed, args.acap, args.out)
    if not ok:
        print(f"FAILED to collect (seed {args.seed}, acap {args.acap})")
        return
    traj, init_state, model_xml = res
    n = len(traj["actions"])
    A = np.array(traj["actions"])
    print(f"collected seed {args.seed} acap {args.acap}: {n} steps")
    print(f"  pos |a|>0.95 saturated: {np.mean(np.abs(A[:,:3])>0.95)*100:.1f}%   |pos| mean={np.abs(A[:,:3]).mean():.3f}")
    print(f"  pos mid[0.2,0.8]: {np.mean((np.abs(A[:,:3])>0.2)&(np.abs(A[:,:3])<0.8))*100:.1f}%")

    with h5py.File(args.out, "w") as f:
        data = f.create_group("data")
        data.attrs["env_args"] = json.dumps(ENV_ARGS)
        data.attrs["total"] = n
        g = data.create_group("demo_0")
        g.attrs["num_samples"] = n; g.attrs["seed"] = args.seed; g.attrs["model_file"] = model_xml
        g.create_dataset("actions", data=A.astype(np.float64))
        g.create_dataset("states", data=np.array(traj["states"], dtype=np.float64))
        g.create_dataset("rewards", data=np.array(traj["rewards"], dtype=np.float64))
        g.create_dataset("dones", data=np.array(traj["dones"], dtype=np.int64))
        og = g.create_group("obs")
        for k in OBS_KEYS_TO_RECORD:
            og.create_dataset(k, data=np.array(traj["obs"][k]))
    np.save(args.out.replace(".hdf5", "_init.npy"), init_state)
    print(f"  saved {args.out}")


if __name__ == "__main__":
    main()
