import os, numpy as np, mujoco, robosuite
xml = os.path.join(os.path.dirname(robosuite.__file__), "models/assets/robots/gr1/robot.xml")
m = mujoco.MjModel.from_xml_path(xml); d = mujoco.MjData(m)
JN = {"right": ["r_shoulder_pitch","r_shoulder_roll","r_shoulder_yaw","r_elbow_pitch","r_wrist_yaw","r_wrist_roll","r_wrist_pitch"],
      "left":  ["l_shoulder_pitch","l_shoulder_roll","l_shoulder_yaw","l_elbow_pitch","l_wrist_yaw","l_wrist_roll","l_wrist_pitch"]}
QADR = {arm: [m.jnt_qposadr[mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_JOINT, n)] for n in names] for arm, names in JN.items()}
WAIST = [m.jnt_qposadr[mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_JOINT, n)] for n in ["torso_waist_yaw","torso_waist_pitch","torso_waist_roll"]]
SITE = {"right": mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_SITE, "r_wrist_site"), "left": mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_SITE, "l_wrist_site")}
def fk(q_left, q_right, waist=(0, 0, 0)):
    d.qpos[:] = 0
    for a, q in (("left", q_left), ("right", q_right)):
        for adr, v in zip(QADR[a], q): d.qpos[adr] = v
    for adr, v in zip(WAIST, waist): d.qpos[adr] = v
    mujoco.mj_forward(m, d)
    return d.site_xpos[SITE["left"]].copy(), d.site_xpos[SITE["right"]].copy()
def ee_traj(st_arm14, waist3=None):
    """st_arm14: (n,14) left7+right7 joint angles -> (n,3) right wrist site, (n,3) left"""
    L, R = [], []
    for i in range(len(st_arm14)):
        w = waist3[i] if waist3 is not None else (0, 0, 0); l, r = fk(st_arm14[i, :7], st_arm14[i, 7:], w); L.append(l); R.append(r)
    return np.array(L), np.array(R)
