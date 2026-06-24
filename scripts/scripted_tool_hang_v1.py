"""Scripted tool hang controller - 100% success rate on 50 seeds.

Key tricks:
- Phase 1 insertion: gradually tilt EEF 60° toward vertical while pushing down.
  This leaves the arm in a joint config that avoids singularity for the next phase.
- Release frame in place, then pure +x action (no OSC) to retreat without
  dragging the frame out of the hole.
- Tool hang: continue pushing down before fully releasing, so the tool slides
  fully onto the hook instead of jamming on the rim.

Usage:
    MUJOCO_GL=egl python scripts/scripted_tool_hang_v1.py
"""

import os
os.environ["MUJOCO_GL"] = "egl"

import numpy as np
import robosuite
import robosuite.utils.transform_utils as T
from scipy.spatial.transform import Rotation as R


# Controller config matching robomimic official tool_hang/ph dataset
# (world frame, kp=150, damping=1, OSC_POSE delta)
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
    "reward_shaping": False,
}


def run_episode(seed, horizon=4000):
    """Run full scripted tool hang episode. Returns success bool."""
    env = robosuite.make("ToolHang", horizon=horizon, **ENV_KWARGS)
    np.random.seed(seed)
    obs_h = [env.reset()]
    sim = env.sim

    def do_step(a):
        obs_h[0], _, _, _ = env.step(a)

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

    def compute_frame_target_quat():
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

    # ==================== Phase 1: Frame Assembly ====================
    for _ in range(10):
        do_step(np.zeros(7))

    fp = obs()["frame_pos"].copy()
    fmw = T.quat2mat(obs()["frame_quat"].copy())
    fx = fmw[:, 0].copy(); fx[2] = 0; fx = fx / np.linalg.norm(fx)
    fa = np.arctan2(fx[1], fx[0])
    fmount = gs("frame_mount_site")
    nd = fmount - fp; nd[2] = 0; nd = nd / np.linalg.norm(nd)

    hc = np.zeros(3)
    for i in range(4):
        hc += sim.data.geom_xpos[sim.model.geom_name2id(f"stand_wall{i}")]
    hc /= 4
    sm = gs("stand_mount_site"); hc[2] = sm[2]

    pa = fa - np.pi / 2
    gq = T.mat2quat(np.column_stack([
        np.array([np.cos(pa), np.sin(pa), 0]),
        np.cross(np.array([0, 0, -1.0]), np.array([np.cos(pa), np.sin(pa), 0])),
        np.array([0, 0, -1.0]),
    ]))
    gp = fp + nd * 0.02; gp[2] = fp[2]

    # Grasp frame
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
        return False

    # Rotate (two passes for slip correction)
    tq = compute_frame_target_quat()
    move_ori(obs()["robot0_eef_pos"].copy(), tq, 1, steps=80, pg=4, og=8, ot=0.02)
    tq2 = compute_frame_target_quat()
    move_ori(obs()["robot0_eef_pos"].copy(), tq2, 1, steps=40, pg=4, og=10, ot=0.01)

    # Fine align above hole
    for _ in range(3):
        etn = gs("frame_mount_site") - obs()["robot0_eef_pos"]
        tf = hc - etn; tf[2] = sm[2] + 0.05 - etn[2]
        move_ori(tf, tq2, 1, steps=35, pg=25, og=10, pt=0.002, ot=0.015)

    # Push down with gradual 60° tilt of EEF toward vertical
    # This leaves the arm in a joint config that avoids singularity for picking the tool
    tilt_deg = 60
    tq2_mat = T.quat2mat(tq2)
    cur_z = tq2_mat[:, 2]
    target_z = np.array([0, 0, -1.0])
    axis = np.cross(cur_z, target_z)
    axis = axis / np.linalg.norm(axis)
    rot = R.from_rotvec(axis * np.radians(tilt_deg)).as_matrix()
    tilted_q = T.mat2quat(rot @ tq2_mat)

    n_push = 120
    for i in range(n_push):
        frac = (i + 1) / n_push
        dot = np.dot(tq2, tilted_q)
        if dot < 0:
            tilted_use = -tilted_q; dot = -dot
        else:
            tilted_use = tilted_q
        dot = np.clip(dot, -1, 1)
        theta = np.arccos(dot)
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

    # Release in place + pure +x retreat (no OSC, just raw action)
    for _ in range(20):
        do_step(np.array([0, 0, 0, 0, 0, 0, -1]))
    for _ in range(25):
        do_step(np.array([1, 0, 0, 0, 0, 0, -1]))

    if not env._check_frame_assembled():
        return False

    # ==================== Phase 2: Tool Pick + Hang ====================

    # Move to tool +5cm (no rotation)
    tp = obs()["tool_pos"].copy(); target = tp.copy(); target[2] += 0.05
    for i in range(120):
        eef = obs()["robot0_eef_pos"]; dp = np.clip((target - eef) * 12, -1, 1)
        do_step(np.concatenate([dp, [0, 0, 0, -1]]))
        if np.linalg.norm(eef - target) < 0.01:
            break

    # Rotate to tool-aligned orientation
    tool_mat = T.quat2mat(obs()["tool_quat"].copy())
    tool_x = tool_mat[:, 0].copy(); tool_x[2] = 0
    tool_x = tool_x / np.linalg.norm(tool_x)
    ta = np.arctan2(tool_x[1], tool_x[0])
    x_eef = np.array([np.cos(ta), np.sin(ta), 0])
    y_eef = np.cross(np.array([0, 0, -1.0]), x_eef)
    tgq = T.mat2quat(np.column_stack([x_eef, y_eef, np.array([0, 0, -1.0])]))
    move_ori(obs()["robot0_eef_pos"].copy(), tgq, -1, steps=100, pg=8, og=5)

    # Re-align above tool
    tp2 = obs()["tool_pos"].copy(); ab2 = tp2.copy(); ab2[2] += 0.05
    move_ori(ab2, tgq, -1, steps=40, pg=15, og=5)

    # Lower to tool
    tp3 = obs()["tool_pos"].copy(); dn2 = tp3.copy(); dn2[2] -= 0.005
    move_ori(dn2, tgq, -1, steps=50, pg=8, og=5)

    # Close gripper
    for _ in range(15):
        o = obs(); dp = np.clip((dn2 - o["robot0_eef_pos"]) * 3, -1, 1)
        ea = T.quat2axisangle(T.mat2quat(T.quat2mat(tgq) @ T.quat2mat(o["robot0_eef_quat"]).T))
        do_step(np.concatenate([dp, np.clip(ea * 2, -1, 1), [1]]))

    # Lift tool
    lift = obs()["robot0_eef_pos"].copy(); lift[2] += 0.20
    move_ori(lift, tgq, 1, steps=40, pg=12, og=3)

    if obs()["tool_pos"][2] - tp3[2] < 0.05:
        return False

    # Hang tool on frame
    fh = gs("frame_hang_site")
    th = gs("tool_hole1_center")
    eth = th - obs()["robot0_eef_pos"]

    import os as _os
    if _os.environ.get("PROBE") == "1":
        fi = gs("frame_intersection_site")
        hd = fh - fi; hd[2] = 0; hd = hd / np.linalg.norm(hd)
        tx = T.quat2mat(obs()["tool_quat"])[:, 0].copy()
        ah = np.degrees(np.arctan2(hd[1], hd[0]))
        at = np.degrees(np.arctan2(tx[1], tx[0]))
        print(f"  PROBE seed: hook_ang={ah:.1f} tool_x_ang={at:.1f} "
              f"delta={(at-ah+180)%360-180:.1f} ring_off={np.round(th-fh,3)}")

    # Above hang site
    ta2 = fh - eth; ta2[2] = fh[2] + 0.10
    move_ori(ta2, tgq, 1, steps=60, pg=12, og=5)

    # Fine align
    for _ in range(3):
        th = gs("tool_hole1_center"); etn = th - obs()["robot0_eef_pos"]
        tf = fh - etn; tf[2] = fh[2] + 0.05
        move_ori(tf, tgq, 1, steps=30, pg=20, og=8, pt=0.002, ot=0.01)

    # Lower onto hook
    for _ in range(2):
        th = gs("tool_hole1_center"); etn = th - obs()["robot0_eef_pos"]
        tf = fh - etn; tf[2] = fh[2] - 0.02
        move_ori(tf, tgq, 1, steps=30, pg=5, og=3)

    # Release: gentle down with grip to seat tool on hook, then release, then
    # firmer/longer down with open gripper to slide tool fully onto hook. Force
    # kept small enough not to destabilize the frame.
    for _ in range(20):
        do_step(np.array([0, 0, -0.2, 0, 0, 0, 1]))
    for _ in range(15):
        do_step(np.array([0, 0, 0, 0, 0, 0, -1]))
    for _ in range(40):
        do_step(np.array([0, 0, -0.2, 0, 0, 0, -1]))
    for _ in range(10):
        do_step(np.array([0, 0, 0, 0, 0, 0, -1]))

    return env._check_success()


def test(n_seeds=50):
    s = 0
    fails = []
    for seed in range(n_seeds):
        ok = run_episode(seed)
        s += 1 if ok else 0
        if not ok:
            fails.append(seed)
        if seed % 10 == 9:
            print(f"Seeds 0-{seed}: {s}/{seed+1}")
    print(f"\nTotal: {s}/{n_seeds} ({100*s/n_seeds:.0f}%)")
    if fails:
        print(f"Failed: {fails}")


if __name__ == "__main__":
    test(50)
