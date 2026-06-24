"""Reusable scripted insertion from the CURRENT env state (no reset).
Runs: compute-frame-quat -> align above hole -> 60deg tilt push -> release.
Returns whether the frame ended up assembled. Used as the ground-truth
'back-half' takeover when evaluating a front-half (pick->handoff) policy.
"""
import numpy as np
import robosuite.utils.transform_utils as T
from scipy.spatial.transform import Rotation as R


def run_insertion(env):
    sim = env.sim
    o = [env._get_observations(force_update=True)]

    def obs():
        return o[0]

    def gs(n):
        return sim.data.site_xpos[sim.model.site_name2id(n)].copy()

    def step(a):
        o[0], _, _, _ = env.step(np.clip(a, -1, 1).astype(np.float64))

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
    for _ in range(60):
        eef = obs()["robot0_eef_pos"]; etn = gs("frame_mount_site") - eef
        tf = hc - etn; tf[2] = sm[2] + 0.05 - etn[2]; dp = np.clip((tf - eef) * 15, -1, 1)
        ea = T.quat2axisangle(T.mat2quat(T.quat2mat(tq2) @ T.quat2mat(obs()["robot0_eef_quat"]).T))
        step(np.concatenate([dp, np.clip(ea * 8, -1, 1), [1]]))
        if np.linalg.norm((gs("frame_mount_site") - hc)[:2]) < 0.002:
            break
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
    for _ in range(20):
        step(np.array([0, 0, 0, 0, 0, 0, -1]))
    return env._check_frame_assembled()
