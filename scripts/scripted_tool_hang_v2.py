"""Scripted tool hang controller v2 - OBS-TRIGGERED transitions (Markovian).

v1 used fixed step-count dwells for gripper close/open, retreat, and tool
seating. Those create non-Markovian points: the obs is static for N steps while
the action is constant, then the phase switches purely because the counter hit
N -> identical obs maps to two different actions -> regression learns the mean
and undershoots the transition.

v2 replaces every fixed-count constant-action loop with an obs-triggered
`while` so the transition fires exactly when an observable threshold is crossed
(gripper_qpos settles, eef_z settles, eef_x passes a target). Crossing the
threshold changes the discriminating obs dimension, so each obs maps to a single
action -> Markovian -> learnable by plain regression.

The proportional `move_ori` moves are already obs-conditioned and unchanged.
The leading settle (frame free-falls on reset) is still emitted but the
collection script drops those leading near-zero actions from the recording.

Usage:
    MUJOCO_GL=egl python scripts/scripted_tool_hang_v2.py
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
                "kp": float(__import__("os").environ.get("KP_OVERRIDE", "150")),
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

# For robomimic-format recording (record=True)
ENV_ARGS = {"env_name": "ToolHang", "env_version": "1.5.1", "type": 1,
            "env_kwargs": {**ENV_KWARGS, "camera_depths": False,
                           "camera_heights": 84, "camera_widths": 84,
                           "lite_physics": False}}
OBS_KEYS_TO_RECORD = [
    "object", "robot0_eef_pos", "robot0_eef_quat", "robot0_eef_quat_site",
    "robot0_gripper_qpos", "robot0_gripper_qvel", "robot0_joint_pos",
    "robot0_joint_pos_cos", "robot0_joint_pos_sin", "robot0_joint_vel",
]


def extract_obs(o):
    out = {
        "object": o["object-state"].copy(),
        "robot0_eef_pos": o["robot0_eef_pos"].copy(),
        "robot0_eef_quat": o["robot0_eef_quat"].copy(),
        "robot0_eef_quat_site": o["robot0_eef_quat_site"].copy(),
        "robot0_gripper_qpos": o["robot0_gripper_qpos"].copy(),
        "robot0_gripper_qvel": o["robot0_gripper_qvel"].copy(),
        "robot0_joint_pos": np.arctan2(o["robot0_joint_pos_sin"], o["robot0_joint_pos_cos"]),
        "robot0_joint_pos_cos": o["robot0_joint_pos_cos"].copy(),
        "robot0_joint_pos_sin": o["robot0_joint_pos_sin"].copy(),
        "robot0_joint_vel": o["robot0_joint_vel"].copy(),
    }
    return out


def run_episode(seed, horizon=4000, render=False, verbose=False, record=False, noise_seed=None):
    """Run full scripted tool hang episode. Returns success bool; if render,
    (success, frames); if record, (success, traj, init_state, model_xml). The
    leading settle steps are NOT recorded (frame free-falls on reset).
    noise_seed: separate RNG seed for DART disturbances (so the same placement
    seed can be rolled out with different noise -> recovery coverage of one tube)."""
    env_kw = dict(ENV_KWARGS)
    if render:
        env_kw["use_camera_obs"] = True
        env_kw["has_offscreen_renderer"] = True
        env_kw["camera_names"] = ["sideview"]
        env_kw["camera_heights"] = 320
        env_kw["camera_widths"] = 320
    env = robosuite.make("ToolHang", horizon=horizon, **env_kw)
    np.random.seed(seed)
    obs_h = [env.reset()]
    sim = env.sim
    frames = []
    init_state = sim.get_state().flatten()
    model_xml = sim.model.get_xml()
    traj = {"actions": [], "states": [], "rewards": [], "dones": [],
            "obs": {k: [] for k in OBS_KEYS_TO_RECORD}}

    import os as _os
    _smooth_cap = float(_os.environ.get("SMOOTH_CAP", "1.0"))  # cap |pos action| for smooth (official-like) trajs
    _dart_pos = float(_os.environ.get("DART_POS", "0.0"))   # DART: scale of pos-action disturbance
    _dart_rot = float(_os.environ.get("DART_ROT", "0.0"))   # DART: scale of rot-action disturbance
    _dart_dist = _os.environ.get("DART_DIST", "gaussian")   # "gaussian" | "cauchy" | "student_t" | "phased"
    _dart_df = float(_os.environ.get("DART_DF", "2.0"))     # Student-t degrees of freedom (heavy tail)
    _dart_gain = float(_os.environ.get("DART_GAIN", "5.0")) # global gain on per-phase human MAD (phased mode)
    _dart_p = float(_os.environ.get("DART_P", "1.0"))       # phased mode: per-step probability of injecting noise (temporal sparsity)
    _dart_rng = np.random.RandomState(noise_seed if noise_seed is not None else seed)
    # Per-phase noise matched to measured HUMAN demo structure (MAD, Student-t df):
    #   align phases (P2,P5): larger + heavy-tailed; insert/hang (P3,P6): small + near-Gaussian
    # grasp phases (P1 pick-frame, P4 pick-tool) use SMALLER noise — we don't want
    # to waste effort on grasp failures; the recovery focus is insertion (via
    # MISALIGN) and the align/hang phases.
    _gn = float(_os.environ.get("GRASP_NOISE_SCALE", "0.4"))
    PHASE_NOISE = {
        "P1_pickframe":  (0.0145 * _gn, 2.0),
        "P2_align_frame":(0.0111, 2.0),
        "P3_insert":     (0.0053, 4.6),
        "P4_pick_tool":  (0.0114 * _gn, 5.1),
        "P5_align_tool": (0.0083, 4.1),
        "P6_hang":       (0.0053, 5.8),
    }
    _phase = ["P1_pickframe"]  # mutable current-phase marker, set by controller sections
    def _dart_noise(n, scale):
        if _dart_dist == "cauchy":
            return _dart_rng.standard_cauchy(n) * scale
        if _dart_dist == "student_t":
            return _dart_rng.standard_t(_dart_df, n) * scale
        return _dart_rng.randn(n) * scale
    def _phase_noise(n):
        mad, df = PHASE_NOISE[_phase[0]]
        return _dart_rng.standard_t(df, n) * (mad * _dart_gain)
    def _sc(n):  # scale step budgets up when capping (slower per-step -> more steps to converge)
        return int(n / max(_smooth_cap, 0.15)) if _smooth_cap < 1.0 else n

    _dart_act = float(_os.environ.get("DART_ACT", "0.0"))  # if >0: also add action-target noise (dual: state+action noise, human-like)
    _impulse_mag = float(_os.environ.get("IMPULSE", "0.0"))  # if >0: one-shot per-phase position impulse
    _imp_state = {"done_in": set(), "cnt": 0, "last_phase": None}
    def do_step(a, rec=True):
        a = np.clip(a, -1, 1).astype(np.float64)
        if record and rec:
            a_rec = a.copy()
            if _dart_act > 0:  # DUAL noise: record corrective action + human-like action noise
                mad, df = PHASE_NOISE[_phase[0]]
                a_rec[:3] = np.clip(a[:3] + _dart_rng.standard_t(df, 3) * (mad * _dart_act), -1, 1)
            traj["actions"].append(a_rec)
            traj["states"].append(sim.get_state().flatten().copy())
            for k, v in extract_obs(obs_h[0]).items():
                traj["obs"][k].append(v)
        a_exec = a.copy()
        if _impulse_mag > 0:
            # one-shot impulse per phase: a few steps after entering a new phase,
            # inject a single large random lateral push -> off-tube -> expert recovers
            ph = _phase[0]
            if ph != _imp_state["last_phase"]:
                _imp_state["last_phase"] = ph; _imp_state["cnt"] = 0
            _imp_state["cnt"] += 1
            if ph not in _imp_state["done_in"] and _imp_state["cnt"] == 3:
                u = _dart_rng.randn(3); u = u / (np.linalg.norm(u) + 1e-9)
                a_exec[:3] = np.clip(a[:3] + u * _impulse_mag, -1, 1)
                _imp_state["done_in"].add(ph)
        elif _dart_dist == "phased":  # per-phase human-matched noise (pos dims), temporally SPARSE
            if _dart_rng.rand() < _dart_p:
                a_exec[:3] = np.clip(a[:3] + _phase_noise(3), -1, 1)
        elif _dart_pos > 0 or _dart_rot > 0:  # DART: execute action + disturbance so state wanders off-tube
            _gate = True
            if _os.environ.get("DART_TRANSIT", "0") == "1":
                _gate = _transit_flag[0] or np.linalg.norm(a[:3]) > 0.3   # transit context or fast command
            _pb = float(_os.environ.get("DART_PB", "1.0"))  # burst mode: sparse-in-time large kicks
            if _gate and _pb < 1.0 and _dart_rng.rand() >= _pb:
                _gate = False
            if _gate:
                a_exec[:3] = np.clip(a[:3] + _dart_noise(3, _dart_pos), -1, 1)
                a_exec[3:6] = np.clip(a[3:6] + _dart_noise(3, _dart_rot), -1, 1)
        obs_h[0], reward, done, _ = env.step(a_exec)
        if record and rec:
            traj["rewards"].append(float(reward))
            traj["dones"].append(int(done))
        if render:
            frames.append(obs_h[0]["sideview_image"][::-1])

    def obs():
        return obs_h[0]

    def gqsum():
        return float(np.abs(obs()["robot0_gripper_qpos"]).sum())

    def ori_act(oq, og=2):
        o = obs()
        ea = T.quat2axisangle(T.mat2quat(T.quat2mat(oq) @ T.quat2mat(o["robot0_eef_quat"]).T))
        return np.clip(ea * og, -1, 1)

    _waypoint = _os.environ.get("WAYPOINT", "0") == "1"
    _brake = _os.environ.get("BRAKE", "0") == "1"
    _vtraj = _os.environ.get("VTRAJ", "0") == "1"   # virtual straight-line carrot transit
    _vt_lead = float(_os.environ.get("VT_LEAD", "0.02"))  # carrot lead (m)
    _mj = _os.environ.get("MJTRAJ", "0") == "1"     # min-jerk time-parameterized reference (MP-style)
    _mj_vmax = float(_os.environ.get("MJ_VMAX", "0.006"))  # m per control step cruise
    _exc_amp = float(_os.environ.get("EXC_AMP", "0"))   # planned excursion amplitude (m); 0 = pure zero-point
    _exc_n = int(_os.environ.get("EXC_N", "2"))         # excursions per transit
    _exc_len = int(_os.environ.get("EXC_LEN", "26"))    # steps per excursion (out and back)
    _exc_asym = float(_os.environ.get("EXC_ASYM", "0.3"))  # fraction of window spent outbound (fast-out slow-back)
    _mj_dec = _os.environ.get("MJ_DECOUPLE", "0") == "1"  # rotate-in-place first, then translate (kills rot-coupling bow)
    _mj_ff = _os.environ.get("MJ_FF", "0") == "1"       # model-aware feedforward: action = KFF*dref (tangent) + KFB*(ref-eef)
    _mj_kff = float(_os.environ.get("MJ_KFF", "100"))   # feedforward gain: a = KFF * ref-step (cmd/realized ~5x, 0.05 scaling)
    _mj_kfb = float(_os.environ.get("MJ_KFB", "2"))     # small lateral correction gain (vs pg=10 pure feedback)
    _mp_js = _os.environ.get("MP_JS", "0") == "1"       # joint-space motion planning (robot model + IK), plan->FK ref->track
    _exc_rng = np.random.RandomState(int(_os.environ.get("EXC_SEED", "0")) + 7919)
    _ramp = float(_os.environ.get("DECEL_RAMP", "0"))  # >0: cap_eff=min(cap, ramp*dist) near target
    def _brake_stop(g, eps=4e-4, max_steps=25):
        """zero-pos-action hold until eef velocity dies (momentum bleed before next target)."""
        prev = None
        for i in range(max_steps):
            o = obs(); cur = o["robot0_eef_pos"].copy()
            do_step(np.array([0, 0, 0, 0, 0, 0, g]))
            if prev is not None and np.linalg.norm(cur - prev) < eps:
                return i
            prev = cur
        return max_steps
    _wp_spacing = float(_os.environ.get("WP_SPACING", "0.015"))  # 1.5cm
    _wp_eps = float(_os.environ.get("WP_EPS", "0.01"))           # 1cm arrival
    _wp_hopcap = int(_os.environ.get("WP_HOPCAP", "20"))         # per-hop safety valve (steps)
    global WP_STATS
    WP_STATS = {"hops": 0, "valve": 0}
    _transit_flag = [False]  # set by waypoint-mode non-final hops; used by DART gate

    def move_ori(tp, tq, g, steps=80, pg=10, og=3, pt=0.005, ot=0.03):
        steps = _sc(steps)
        _wp_min = 0.03  # only decompose moves longer than 3cm; short/rotation moves stay original
        if _mp_js and np.linalg.norm(tp - obs()["robot0_eef_pos"]) > 0.03:
            refs = _mp_plan(tp, T.quat2mat(tq))
            if refs is not None:
                _mp_track(refs, g, og=og)
                # terminal precision: short reactive convergence to pt/ot
                for _i2 in range(20):
                    o = obs(); eef = o["robot0_eef_pos"]
                    em = T.quat2mat(o["robot0_eef_quat"]); tm = T.quat2mat(tq)
                    ea = T.quat2axisangle(T.mat2quat(tm @ em.T))
                    if np.linalg.norm(eef - tp) < pt and np.linalg.norm(ea) < ot:
                        return 0
                    dp = np.clip((tp - eef) * pg, -0.15, 0.15)
                    do_step(np.concatenate([dp, np.clip(ea * og, -1, 1), [g]]))
                return 0
        if not _waypoint or np.linalg.norm(tp - obs()["robot0_eef_pos"]) < _wp_min:
            _vt_start = obs()["robot0_eef_pos"].copy()
            _vt_dist = np.linalg.norm(tp - _vt_start)
            _vt_dir = (tp - _vt_start) / (_vt_dist + 1e-9)
            _vt_use = (_vtraj or _mj) and _vt_dist > 0.03   # transit only; short precision moves untouched
            _mj_T = max(int(np.ceil(_vt_dist / _mj_vmax * 1.5)), 8)  # min-jerk duration (steps)
            _wmax = float(_os.environ.get("MJ_WMAX", "0"))
            if _wmax > 0:
                # rotation-budgeted duration: cap angular speed at MJ_WMAX
                # rad/step by extending the segment clock (SLERP follows it)
                _em0 = T.quat2mat(obs()["robot0_eef_quat"]); _tm0 = T.quat2mat(tq)
                _rot_d = float(np.linalg.norm(T.quat2axisangle(T.mat2quat(_tm0 @ _em0.T))))
                _mj_T = max(_mj_T, int(np.ceil(_rot_d / _wmax * 1.5)))
            _mj_i = 0
            _mj_slerp = _os.environ.get("MJ_SLERP", "0") == "1"
            _q_start = obs()["robot0_eef_quat"].copy()
            _exc_sched = None
            _ref_prev = _vt_start.copy()
            for i in range(steps):
                o = obs(); eef = o["robot0_eef_pos"]
                if _mj and _vt_use:
                    if _mj_dec and _mj_i == 0:
                        em_ = T.quat2mat(o["robot0_eef_quat"]); tm_ = T.quat2mat(tq)
                        ea_ = T.quat2axisangle(T.mat2quat(tm_ @ em_.T))
                        if np.linalg.norm(ea_) > 0.15:
                            # rotation not settled: hold position at start, rotate only
                            dp = np.clip((_vt_start - eef) * pg, -1, 1)
                            do_ = np.clip(ea_ * og, -1, 1)
                            do_step(np.concatenate([dp, do_, [g]]))
                            continue
                    # min-jerk reference s(t): 10t^3-15t^4+6t^5, time-indexed (no arrival waits)
                    if _exc_sched is None and _exc_amp > 0:
                        # schedule excursions: (start_step, unit normal) pairs on this transit
                        _exc_sched = []
                        for _e in range(_exc_n):
                            if _mj_T < 2 * _exc_len + 40: break  # transit too short for safe excursions
                            c = int(_mj_T * (0.22 + 0.33 * _e / max(_exc_n - 1, 1)))
                            nv = _exc_rng.randn(3); nv -= (nv @ _vt_dir) * _vt_dir
                            nn_ = np.linalg.norm(nv)
                            if nn_ > 1e-6: _exc_sched.append((c, nv / nn_))
                    if _exc_sched is None: _exc_sched = []
                    _mj_i = min(_mj_i + 1, _mj_T)
                    tt_ = _mj_i / _mj_T
                    sref = 10*tt_**3 - 15*tt_**4 + 6*tt_**5
                    ref = _vt_start + _vt_dir * (_vt_dist * sref)
                    if _exc_amp > 0 and _exc_sched:
                        for (c0, nv) in _exc_sched:
                            ph_ = (_mj_i - c0) / _exc_len
                            if 0.0 <= ph_ <= 1.0:
                                # asymmetric bump: fast out (half-cosine), slow back -> inbound labels dominate
                                if ph_ < _exc_asym:
                                    bump_ = 0.5 * (1 - np.cos(np.pi * ph_ / _exc_asym))
                                else:
                                    bump_ = 0.5 * (1 + np.cos(np.pi * (ph_ - _exc_asym) / (1 - _exc_asym)))
                                ref = ref + nv * (_exc_amp * bump_)
                    if _mj_ff:
                        # model-aware MP: tangent feedforward from the plan, weak lateral servo
                        dp = np.clip(_mj_kff * (ref - _ref_prev) + _mj_kfb * (ref - eef), -1, 1)
                        _ref_prev = ref.copy()
                    else:
                        lead = _vt_dir * min(_vt_lead, _vt_dist * max(0.0, min(1.0, (10*3*tt_**2 - 15*4*tt_**3 + 6*5*tt_**4) / _mj_T)) * 3 + 0.004)
                        dp = np.clip((ref + lead - eef) * pg, -1, 1)
                elif _vt_use:
                    # carrot on the straight line, leading the arm's own progress (closed-loop)
                    prog = float((eef - _vt_start) @ _vt_dir)
                    lead_eff = min(_vt_lead, 0.005 + max(prog, 0.0))
                    carrot = _vt_start + _vt_dir * min(_vt_dist, prog + lead_eff)
                    dp = np.clip((carrot - eef) * pg, -1, 1)
                else:
                    cap_eff = _smooth_cap if _ramp <= 0 else min(_smooth_cap, max(0.06, _ramp * np.linalg.norm(tp - eef)))
                    dp = np.clip((tp - eef) * pg, -cap_eff, cap_eff)
                if _mj and _vt_use and _mj_slerp:
                    # SE(3) reference: orientation follows the same min-jerk
                    # schedule (slerp q_start -> tq), no in-place twist
                    tt_ = _mj_i / _mj_T
                    s_ = 10*tt_**3 - 15*tt_**4 + 6*tt_**5
                    d_ = float(np.dot(_q_start, tq))
                    tq_u = -tq if d_ < 0 else tq; d_ = abs(np.clip(d_, -1, 1))
                    th_ = np.arccos(d_)
                    if th_ < 1e-6:
                        q_ref = tq_u
                    else:
                        q_ref = (np.sin((1 - s_) * th_) / np.sin(th_)) * _q_start \
                            + (np.sin(s_ * th_) / np.sin(th_)) * tq_u
                    q_ref = q_ref / np.linalg.norm(q_ref)
                else:
                    q_ref = tq
                em = T.quat2mat(o["robot0_eef_quat"]); tm = T.quat2mat(q_ref)
                ea = T.quat2axisangle(T.mat2quat(tm @ em.T))
                do = np.clip(ea * og, -1, 1)
                do_step(np.concatenate([dp, do, [g]]))
                tmf = T.quat2mat(tq)
                eaf = T.quat2axisangle(T.mat2quat(tmf @ em.T))
                if np.linalg.norm(eef - tp) < pt and np.linalg.norm(eaf) < ot:
                    if _brake: _brake_stop(g)
                    return i
            if _brake: _brake_stop(g)
            return steps
        # --- waypoint mode: decompose current->tp into ~_wp_spacing waypoints,
        # track each (under noise the controller is forced back onto the line),
        # final waypoint uses the original tight pt/ot convergence. ---
        start = obs()["robot0_eef_pos"].copy()
        dist = np.linalg.norm(tp - start)
        nwp = max(1, int(np.ceil(dist / _wp_spacing)))
        if _os.environ.get("WP_REANCHOR", "0") == "1":
            nwp = nwp * 3 + 10  # loop bound only; reanchor decides 'final' by remaining distance
        budget = max(steps, nwp * _wp_hopcap)
        used = 0
        _tpg = float(_os.environ.get("TRANSIT_PG", "0"))  # weak lateral authority in transit (0=off)
        _reanchor = _os.environ.get("WP_REANCHOR", "0") == "1"  # re-plan chain from CURRENT pos toward the key point each hop
        for j in range(1, nwp + 1):
            if _reanchor:
                eef_now = obs()["robot0_eef_pos"]
                rem = tp - eef_now
                nleft = max(1, int(np.ceil(np.linalg.norm(rem) / _wp_spacing)))
                wp = eef_now + rem / nleft
                final = nleft == 1
            else:
                wp = start + (tp - start) * (j / nwp)
                final = (j == nwp)
            cap = (budget - used) if final else _wp_hopcap
            if not final: WP_STATS["hops"] += 1
            _transit_flag[0] = (not final) and (j < nwp - 2)  # end margin: no noise near precision approach
            for _ in range(max(1, cap)):
                o = obs(); eef = o["robot0_eef_pos"]
                pg_eff = pg if (final or _tpg <= 0) else _tpg
                dp = np.clip((wp - eef) * pg_eff, -_smooth_cap, _smooth_cap)
                em = T.quat2mat(o["robot0_eef_quat"]); tm = T.quat2mat(tq)
                ea = T.quat2axisangle(T.mat2quat(tm @ em.T))
                do = np.clip(ea * og, -1, 1)
                do_step(np.concatenate([dp, do, [g]]))
                used += 1
                if final:
                    if np.linalg.norm(eef - tp) < pt and np.linalg.norm(ea) < ot:
                        return used
                else:
                    if np.linalg.norm(eef - wp) < _wp_eps:
                        break
                    if _ == cap - 1: WP_STATS["valve"] += 1
                if used >= budget:
                    return used
        return used

    # ---- obs-triggered primitives (Markovian transitions) ----
    def grip_until_settled(g, target, tq, gain=3, og=2, eps=8e-4, max_steps=40):
        """Hold position at `target`/orientation `tq`, set gripper `g`, until the
        gripper_qpos stops changing (clamped on object / fully open). Returns the
        moment it settles -> next phase starts immediately (no dwell)."""
        prev = None
        for i in range(max_steps):
            o = obs(); qs = float(np.abs(o["robot0_gripper_qpos"]).sum())
            dp = np.clip((target - o["robot0_eef_pos"]) * gain, -1, 1)
            do_step(np.concatenate([dp, ori_act(tq, og), [g]]))
            if prev is not None and abs(qs - prev) < eps and i >= 3:
                return i
            prev = qs
        return max_steps

    def push_z_until_settled(zact, g, tq=None, eps=5e-4, max_steps=80, extra=0):
        max_steps = _sc(max_steps)
        """Push along z with constant `zact` until eef_z stops decreasing
        (contact/seat), then push `extra` more steps to seat firmly. Orientation
        held if tq given. obs-triggered on eef_z."""
        prev = None
        for i in range(max_steps):
            o = obs(); z = o["robot0_eef_pos"][2]
            do = ori_act(tq, 3) if tq is not None else np.zeros(3)
            do_step(np.concatenate([[0, 0, zact], do, [g]]))
            if prev is not None and abs(z - prev) < eps and i >= 3:
                for _ in range(extra):
                    do_step(np.concatenate([[0, 0, zact], np.zeros(3), [g]]))
                return i
            prev = z
        return max_steps

    def retreat_x_until(dx, g, max_steps=60):
        max_steps = _sc(max_steps)
        """Pure +x retreat until eef_x has advanced by `dx`. obs-triggered."""
        start_x = obs()["robot0_eef_pos"][0]
        for i in range(max_steps):
            if obs()["robot0_eef_pos"][0] >= start_x + dx:
                return i
            do_step(np.array([1, 0, 0, 0, 0, 0, g]))
        return max_steps

    def hold_until_static(g, eps=3e-3, max_steps=30):
        """Terminal settle: hold [0,0,0,g] until the full obs stops changing."""
        prev = None
        for i in range(max_steps):
            o = obs()
            cur = np.concatenate([o["object-state"], o["robot0_eef_pos"]])
            do_step(np.array([0, 0, 0, 0, 0, 0, g]))
            if prev is not None and np.linalg.norm(cur - prev) < eps and i >= 3:
                return i
            prev = cur
        return max_steps

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

    # ==================== Phase 0: settle (NOT recorded) ====
    for _ in range(10):
        do_step(np.zeros(7), rec=False)

    # ==================== Phase 1: Frame Assembly ====================
    hc = np.zeros(3)
    for i in range(4):
        hc += sim.data.geom_xpos[sim.model.geom_name2id(f"stand_wall{i}")]
    hc /= 4
    sm = gs("stand_mount_site"); hc[2] = sm[2]


    # ---------- joint-space motion planning (robot model + IK) ----------
    _rj = env.robots[0]
    _qidx = np.array(_rj._ref_joint_pos_indexes)
    _didx = np.array(getattr(_rj, "_ref_joint_vel_indexes", _rj._ref_joint_pos_indexes))
    _site = "gripper0_right_grip_site"
    _jlim = env.sim.model.jnt_range[:len(_qidx)].copy()

    def _fk(q):
        """kinematic-only FK on scratch state; restores sim exactly."""
        sim = env.sim
        q0 = sim.data.qpos.copy(); v0 = sim.data.qvel.copy()
        sim.data.qpos[_qidx] = q; sim.forward()
        pos = sim.data.get_site_xpos(_site).copy()
        mat = sim.data.get_site_xmat(_site).copy()
        sim.data.qpos[:] = q0; sim.data.qvel[:] = v0; sim.forward()
        return pos, mat

    def _mp_ik(p_star, R_star, iters=80):
        """damped-least-squares IK from the CURRENT joint config (robot model)."""
        sim = env.sim
        q0 = sim.data.qpos.copy(); v0 = sim.data.qvel.copy()
        q = sim.data.qpos[_qidx].copy()
        ok_ik = False
        for _ in range(iters):
            sim.data.qpos[_qidx] = q; sim.forward()
            pc = sim.data.get_site_xpos(_site).copy()
            Rc = sim.data.get_site_xmat(_site).copy()
            ep = p_star - pc
            eR = T.quat2axisangle(T.mat2quat(R_star @ Rc.T))
            if np.linalg.norm(ep) < 3e-4 and np.linalg.norm(eR) < 3e-3:
                ok_ik = True; break
            Jp = sim.data.get_site_jacp(_site).reshape(3, -1)[:, _didx]
            Jr = sim.data.get_site_jacr(_site).reshape(3, -1)[:, _didx]
            J = np.vstack([Jp, Jr])
            e = np.concatenate([ep, 0.5 * eR])
            dq = J.T @ np.linalg.solve(J @ J.T + 1e-4 * np.eye(6), e)
            q = np.clip(q + np.clip(dq, -0.2, 0.2), _jlim[:, 0], _jlim[:, 1])
        sim.data.qpos[:] = q0; sim.data.qvel[:] = v0; sim.forward()
        return (q, ok_ik)

    def _ik_step(q_from, p_t, R_t, iters=8):
        """incremental DLS-IK: refine q_from toward (p_t, R_t) on scratch state."""
        sim = env.sim
        q0 = sim.data.qpos.copy(); v0 = sim.data.qvel.copy()
        q = q_from.copy()
        for _ in range(iters):
            sim.data.qpos[_qidx] = q; sim.forward()
            pc = sim.data.get_site_xpos(_site).copy()
            Rc = sim.data.get_site_xmat(_site).copy()
            ep = p_t - pc
            eR = T.quat2axisangle(T.mat2quat(R_t @ Rc.T))
            if np.linalg.norm(ep) < 2e-4 and np.linalg.norm(eR) < 2e-3:
                break
            Jp = sim.data.get_site_jacp(_site).reshape(3, -1)[:, _didx]
            Jr = sim.data.get_site_jacr(_site).reshape(3, -1)[:, _didx]
            J = np.vstack([Jp, Jr])
            e = np.concatenate([ep, 0.5 * eR])
            dq = J.T @ np.linalg.solve(J @ J.T + 1e-4 * np.eye(6), e)
            q = np.clip(q + np.clip(dq, -0.15, 0.15), _jlim[:, 0], _jlim[:, 1])
        pc, Rc = _fk(q)
        err = np.linalg.norm(pc - p_t)
        sim.data.qpos[:] = q0; sim.data.qvel[:] = v0; sim.forward()
        return q, pc, Rc, err

    def _mp_plan(p_star, R_star, vmax=None):
        """Cartesian path planning with per-knot IK resolution (robot model):
        pos = min-jerk straight line, ori = slerp on the same profile; each knot
        resolved to joints incrementally -> kinematically verified FK path."""
        vmax = vmax or _mj_vmax
        o = obs()
        p0 = o["robot0_eef_pos"].copy()
        R0 = T.quat2mat(o["robot0_eef_quat"])
        q = env.sim.data.qpos[_qidx].copy()
        dist = np.linalg.norm(p_star - p0)
        ang = np.linalg.norm(T.quat2axisangle(T.mat2quat(R_star @ R0.T)))
        Tn = int(max(np.ceil(dist / vmax * 1.4), np.ceil(ang / 0.03), 8))
        q0m = T.mat2quat(R0); q1m = T.mat2quat(R_star)
        if np.dot(q0m, q1m) < 0: q1m = -q1m
        th = np.arccos(np.clip(np.dot(q0m, q1m), -1, 1))
        refs = []
        for t in range(1, Tn + 1):
            tt = t / Tn
            ss = 10 * tt**3 - 15 * tt**4 + 6 * tt**5
            p_t = p0 + (p_star - p0) * ss
            if th < 1e-6:
                R_t = R_star
            else:
                qm = (np.sin((1 - ss) * th) * q0m + np.sin(ss * th) * q1m) / np.sin(th)
                R_t = T.quat2mat(qm / np.linalg.norm(qm))
            q, pc, Rc, err = _ik_step(q, p_t, R_t)
            if err > 0.01:
                return None  # kinematically infeasible knot -> caller falls back
            refs.append((pc, Rc))
        return refs

    def _mp_track(refs, g, og=4):
        """track the planned FK reference through OSC: tangent feedforward +
        weak trim; ori tracks the planned FK orientation profile."""
        p_prev = obs()["robot0_eef_pos"].copy()
        for (p_ref, R_ref) in refs:
            o = obs(); eef = o["robot0_eef_pos"]
            dp = np.clip(_mj_kff * (p_ref - p_prev) + _mj_kfb * (p_ref - eef), -1, 1)
            p_prev = p_ref
            em = T.quat2mat(o["robot0_eef_quat"])
            ea = T.quat2axisangle(T.mat2quat(R_ref @ em.T))
            do_step(np.concatenate([dp, np.clip(ea * og, -1, 1), [g]]))

    def frame_held():
        # frame still in gripper: frame body close to eef
        return np.linalg.norm(obs()["frame_pos"] - obs()["robot0_eef_pos"]) < 0.20

    def grasp_frame():
        """Open gripper, go to current frame pose, grasp, lift. Returns True if lifted."""
        _phase[0] = "P1_pickframe"
        fp = obs()["frame_pos"].copy()
        fmw = T.quat2mat(obs()["frame_quat"].copy())
        fx = fmw[:, 0].copy(); fx[2] = 0; fx = fx / np.linalg.norm(fx)
        fa = np.arctan2(fx[1], fx[0])
        fmount = gs("frame_mount_site")
        nd = fmount - fp; nd[2] = 0; nd = nd / np.linalg.norm(nd)
        pa = fa - np.pi / 2
        gql = T.mat2quat(np.column_stack([
            np.array([np.cos(pa), np.sin(pa), 0]),
            np.cross(np.array([0, 0, -1.0]), np.array([np.cos(pa), np.sin(pa), 0])),
            np.array([0, 0, -1.0]),
        ]))
        gp = fp + nd * 0.02; gp[2] = fp[2]
        ab = gp.copy(); ab[2] += 0.10
        move_ori(ab, gql, -1, steps=40, pg=12)
        dn = gp.copy(); dn[2] -= 0.005
        move_ori(dn, gql, -1, steps=50, pg=8, og=5)
        grip_until_settled(1, dn, gql, gain=3, og=2)
        lf = obs()["robot0_eef_pos"].copy(); lf[2] += 0.25
        move_ori(lf, gql, 1, steps=30, pg=15, og=3)
        return (obs()["frame_pos"][2] - fp[2] > 0.05) and frame_held()

    # grasp with retry (re-grasp if frame slips out / not lifted)
    _grasp_retries = int(_os.environ.get("GRASP_RETRIES", "4"))
    grasped = False
    for gi in range(_grasp_retries):
        if grasp_frame():
            grasped = True; break
        if verbose:
            print(f"  [seed {seed}] grasp attempt {gi} FAILED (frame not lifted/held) -> regrasp")
        # open gripper so next attempt can re-approach cleanly
        do_step(np.array([0, 0, 0.3, 0, 0, 0, -1]))
    if not grasped:
        if record: return False, traj, init_state, model_xml
        return (False, frames) if render else False

    # Move to above the hole AND align orientation simultaneously, in one
    # continuous loop (no in-place rotation -> no dwell/drift). Each step
    # recomputes the alignment target (self-corrects for slip) and drives the
    # frame_mount_site toward the hole center with the eef actively held on
    # target (high pg) so it does not drift. Terminates obs-triggered when the
    # mount is precisely above the hole and the orientation is aligned.
    def approach_align_hole(tq, pg=12, og=6, zoff=0.05, xy_tol=0.004,
                            o_tol=0.025, max_steps=120):
        _aah_start = None; _aah_dist = 0.0; _aah_dir = None
        max_steps = _sc(max_steps)
        """Move the frame_mount_site above the hole AND rotate the eef to the
        FIXED target quat `tq` simultaneously (continuous, no in-place hold).
        Orientation target is fixed (stable); only the position target adapts
        as the frame settles into `tq`. obs-triggered termination."""
        if _mp_js:
            used = 0
            for _replan in range(4):
                o = obs(); eef = o["robot0_eef_pos"]
                etn = gs("frame_mount_site") - eef
                goal = hc - etn; goal[2] = sm[2] + zoff - etn[2]
                if np.linalg.norm(goal - eef) > 2e-3:
                    refs = _mp_plan(goal, T.quat2mat(tq))
                    if refs is None: break
                    _mp_track(refs, 1, og=og); used += len(refs)
                xy = np.linalg.norm((gs("frame_mount_site") - hc)[:2])
                o = obs(); em = T.quat2mat(o["robot0_eef_quat"]); tm = T.quat2mat(tq)
                ea = T.quat2axisangle(T.mat2quat(tm @ em.T))
                if xy < xy_tol and np.linalg.norm(ea) < o_tol:
                    return used
            return used
        if _mj_ff and _os.environ.get("MJ_ALIGN", "1") == "1":
            # ---- plan -> track -> event-triggered replan (MP architecture) ----
            budget = int(max_steps * 2.0); used = 0
            # stage 1: rotation planned as its own segment (decoupled, position held)
            hold = obs()["robot0_eef_pos"].copy()
            while used < budget // 2:
                o = obs(); em = T.quat2mat(o["robot0_eef_quat"]); tm = T.quat2mat(tq)
                ea = T.quat2axisangle(T.mat2quat(tm @ em.T))
                if np.linalg.norm(ea) < 0.06: break
                dp = np.clip((hold - o["robot0_eef_pos"]) * 2.0, -1, 1)
                do_step(np.concatenate([dp, np.clip(ea * og, -1, 1), [1]])); used += 1
            # stage 2: plan a min-jerk straight segment to the goal, track it
            # tangentially; replan (short segment) if slip moved the goal.
            for _replan in range(4):
                o = obs(); eef = o["robot0_eef_pos"]
                etn = gs("frame_mount_site") - eef
                goal = hc - etn; goal[2] = sm[2] + zoff - etn[2]
                seg_d = np.linalg.norm(goal - eef)
                if seg_d < 1e-4: break
                seg_dir = (goal - eef) / seg_d
                seg_T = max(int(np.ceil(seg_d / (_mj_vmax * 1.4) * 1.5)), 8)
                ref_prev = eef.copy()
                for j in range(1, seg_T + 1):
                    if used >= budget: break
                    o = obs(); eef = o["robot0_eef_pos"]
                    ttj = j / seg_T
                    ref = (eef * 0) + (goal - seg_dir * seg_d) + seg_dir * seg_d * (10*ttj**3 - 15*ttj**4 + 6*ttj**5)
                    em = T.quat2mat(o["robot0_eef_quat"]); tm = T.quat2mat(tq)
                    ea = T.quat2axisangle(T.mat2quat(tm @ em.T))
                    dp = np.clip(_mj_kff * (ref - ref_prev) + _mj_kfb * (ref - eef), -1, 1)
                    ref_prev = ref.copy()
                    do_step(np.concatenate([dp, np.clip(ea * og, -1, 1), [1]])); used += 1
                xy = np.linalg.norm((gs("frame_mount_site") - hc)[:2])
                o = obs(); em = T.quat2mat(o["robot0_eef_quat"]); tm = T.quat2mat(tq)
                ea = T.quat2axisangle(T.mat2quat(tm @ em.T))
                if xy < xy_tol and np.linalg.norm(ea) < o_tol:
                    return used
                if used >= budget: break
            return used
        for i in range(max_steps):
            o = obs(); eef = o["robot0_eef_pos"]
            etn = gs("frame_mount_site") - eef
            tf = hc - etn; tf[2] = sm[2] + zoff - etn[2]
            dp = np.clip((tf - eef) * pg, -_smooth_cap, _smooth_cap)  # smooth free-space approach
            em = T.quat2mat(o["robot0_eef_quat"]); tm = T.quat2mat(tq)
            ea = T.quat2axisangle(T.mat2quat(tm @ em.T))
            do_step(np.concatenate([dp, np.clip(ea * og, -1, 1), [1]]))
            xy = np.linalg.norm((gs("frame_mount_site") - hc)[:2])
            if i >= 5 and xy < xy_tol and np.linalg.norm(ea) < o_tol:
                return i
        return max_steps

    # pass 1: continuous move-to-hole + align toward initial target
    tq = compute_frame_target_quat()
    def align_and_push(misalign=None, tilt_dir=None, skew_deg=20.0, shallow=False):
        """One align+tilt-push attempt. `misalign` (xy vector, m) offsets the eef
        before pushing; `tilt_dir` (angle rad) rotates the insertion tilt axis so
        the frame goes in skewed from a RANDOM direction -> diverse jam/recovery.
        `shallow`: only push ~1/3 of the way down (gentle skewed touch, not a deep
        hard jam) before bailing -> milder recovery, less stand disturbance.
        Returns tq2 used."""
        _phase[0] = "P2_align_frame"
        tqa = compute_frame_target_quat()
        approach_align_hole(tqa, pg=12, og=6, zoff=0.06)
        tq2 = compute_frame_target_quat()
        approach_align_hole(tq2, pg=15, og=8, zoff=0.05, xy_tol=0.002, o_tol=0.012, max_steps=60)
        if misalign is not None:
            # deliberately offset laterally before pushing -> skewed insertion
            tgt = obs()["robot0_eef_pos"].copy(); tgt[:2] += misalign
            move_ori(tgt, tq2, 1, steps=15, pg=10, og=4, pt=0.003)
        # Push down with gradual 60 deg tilt of EEF toward vertical
        _phase[0] = "P3_insert"
        tq2_mat = T.quat2mat(tq2)
        if tilt_dir is not None:
            # tilt toward an EXPLICIT horizontal direction (clean, controllable),
            # by a SMALL angle (skew_tilt_deg) so the skew is gentle and doesn't
            # ram/knock the frame hard. The old cross(frame_z,down) axis degenerated
            # when frame is vertical (norm~0.07) -> all skews looked the same.
            axis = np.array([np.cos(tilt_dir), np.sin(tilt_dir), 0.0])
            tdeg = skew_deg
        else:
            axis = np.cross(tq2_mat[:, 2], np.array([0, 0, -1.0]))
            axis = axis / (np.linalg.norm(axis) + 1e-9)
            tdeg = 60
        tilted_q = T.mat2quat(R.from_rotvec(axis * np.radians(tdeg)).as_matrix() @ tq2_mat)
        z_start = obs()["robot0_eef_pos"][2]
        z_stop = z_start - (z_start - 1.0) / 3.0 if shallow else 1.0  # shallow: 1/3 travel
        n_push = _sc(120)
        for i in range(n_push):
            frac = (i + 1) / n_push
            dot = np.dot(tq2, tilted_q)
            tilted_use = -tilted_q if dot < 0 else tilted_q; dot = abs(dot)
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
            if obs()["robot0_eef_pos"][2] <= z_stop:
                break
        return tq2

    def insert_ok():
        # GEOMETRIC held-judge (no release needed): frame_mount over hole center
        # (xy aligned), frame vertical, AND mount dropped to near hole depth.
        # Tight thresholds so a skewed/jammed frame is not mistaken for seated.
        fm = gs("frame_mount_site"); fi = gs("frame_intersection_site"); sm = gs("stand_mount_site")
        xy = np.linalg.norm((fm - hc)[:2])
        v = fm - fi; v = v / (np.linalg.norm(v) + 1e-9)
        vert = np.degrees(np.arccos(abs(v[2])))
        depth_ok = (fm[2] - sm[2]) < 0.06   # mount actually descended into the hole
        return (xy < 0.012) and (vert < 3.0) and depth_ok

    # Insertion with RECOVERY: if the frame doesn't seat (jammed/misaligned),
    # lift out (pull up, gripper still holding) -> re-align -> retry. This makes
    # the expert demonstrate "insert failed -> pull out -> realign -> reinsert",
    # giving real insertion-recovery coverage (which open-loop DART lacks).
    _max_attempts = int(_os.environ.get("INSERT_RETRIES", "4"))
    # one-shot deliberate misalignment on the FIRST insertion attempt (m), to
    # produce human-like skewed-insertion -> pull-out -> re-align -> re-insert
    # recovery in the data. Retries align cleanly (no misalign) so they succeed.
    _misalign_mag = float(_os.environ.get("MISALIGN", "0.0"))
    tq2 = None
    for attempt in range(_max_attempts):
        if not frame_held():  # frame fully dropped (rare) -> re-grasp from the floor
            if verbose:
                print(f"  [seed {seed}] attempt {attempt}: frame dropped -> regrasp")
            do_step(np.array([0, 0, 0.3, 0, 0, 0, -1]))
            for gi in range(_grasp_retries):
                if grasp_frame():
                    break
                do_step(np.array([0, 0, 0.3, 0, 0, 0, -1]))
        # On the first attempt, with probability MISALIGN_PROB (~0.4), deliberately
        # skew the insertion (random direction, tilt sampled 0-20deg, shallow 1/3
        # push) so ~40% of demos contain a jam -> hold -> re-align -> re-insert
        # recovery; the rest insert cleanly first try.
        mrng = np.random.RandomState((seed + 1) * 2654435761 % (2**32))
        _mis_prob = float(_os.environ.get("MISALIGN_PROB", "0.4"))
        if attempt == 0 and _misalign_mag > 0 and mrng.rand() < _mis_prob:
            ang = mrng.rand() * 2 * np.pi
            mis = _misalign_mag * np.array([np.cos(ang), np.sin(ang)])
            tdir = mrng.rand() * 2 * np.pi          # random skew direction
            tdeg = mrng.rand() * 20.0               # tilt magnitude sampled 0-20deg
            tq2 = align_and_push(misalign=mis, tilt_dir=tdir, skew_deg=tdeg, shallow=True)
        else:
            tq2 = align_and_push()
        # GEOMETRIC judge while STILL GRIPPING (no release). If seated -> done.
        if insert_ok():
            break
        if attempt < _max_attempts - 1:
            if verbose:
                fm = gs("frame_mount_site"); xy = np.linalg.norm((fm - hc)[:2])
                print(f"  [seed {seed}] attempt {attempt} jammed (xy={xy*100:.1f}cm) "
                      f"-> LIFT (hold frame) & re-align & re-insert")
            # RECOVERY (human-like): keep gripping, lift straight up out of the
            # hole, then the next loop re-aligns & re-inserts WITHOUT releasing.
            up = obs()["robot0_eef_pos"].copy(); up[2] += 0.12
            move_ori(up, tq2, 1, steps=40, pg=12, og=4)

    # only NOW release the frame (it is seated), then retreat
    grip_until_settled(-1, obs()["robot0_eef_pos"].copy(), tq2, gain=0, og=3)
    retreat_x_until(0.12, -1)

    if verbose:
        print(f"  [seed {seed}] frame_assembled={env._check_frame_assembled()} "
              f"frame_z={obs()['frame_pos'][2]:.3f}")
    if not env._check_frame_assembled():
        if record: return False, traj, init_state, model_xml
        return (False, frames) if render else False

    if _os.environ.get("STOP_AFTER_INSERT", "0") == "1":
        # init->insertion subtask only: insertion verified seated, stop here
        if record: return True, traj, init_state, model_xml
        return (True, frames) if render else True

    # ==================== Phase 2: Tool Pick + Hang ====================

    def grasp_tool():
        """Approach, align, grasp, lift the tool. Returns (lifted_ok, tgq, tp3)."""
        _phase[0] = "P4_pick_tool"
        tp = obs()["tool_pos"].copy(); target = tp.copy(); target[2] += 0.05
        for i in range(_sc(120)):
            eef = obs()["robot0_eef_pos"]; dp = np.clip((target - eef) * 12, -1, 1)
            do_step(np.concatenate([dp, [0, 0, 0, -1]]))
            if np.linalg.norm(eef - target) < 0.01:
                break
        tool_mat = T.quat2mat(obs()["tool_quat"].copy())
        tool_x = tool_mat[:, 0].copy(); tool_x[2] = 0; tool_x = tool_x / np.linalg.norm(tool_x)
        ta = np.arctan2(tool_x[1], tool_x[0])
        x_eef = np.array([np.cos(ta), np.sin(ta), 0]); y_eef = np.cross(np.array([0, 0, -1.0]), x_eef)
        tgq = T.mat2quat(np.column_stack([x_eef, y_eef, np.array([0, 0, -1.0])]))
        move_ori(obs()["robot0_eef_pos"].copy(), tgq, -1, steps=100, pg=8, og=5)
        tp2 = obs()["tool_pos"].copy(); ab2 = tp2.copy(); ab2[2] += 0.05
        move_ori(ab2, tgq, -1, steps=40, pg=15, og=5)
        tp3 = obs()["tool_pos"].copy(); dn2 = tp3.copy(); dn2[2] -= 0.005
        move_ori(dn2, tgq, -1, steps=50, pg=8, og=5)
        grip_until_settled(1, dn2, tgq, gain=3, og=2)
        _phase[0] = "P5_align_tool"
        lift = obs()["robot0_eef_pos"].copy(); lift[2] += 0.20
        move_ori(lift, tgq, 1, steps=40, pg=12, og=3)
        return (obs()["tool_pos"][2] - tp3[2] > 0.05), tgq, tp3

    # grasp tool with retry (re-grasp if not lifted)
    tgq = None; tp3 = None; tool_ok = False
    for ti in range(_grasp_retries):
        tool_ok, tgq, tp3 = grasp_tool()
        if tool_ok:
            break
        if verbose:
            print(f"  [seed {seed}] tool grasp attempt {ti} FAILED -> regrasp")
        do_step(np.array([0, 0, 0.3, 0, 0, 0, -1]))
    if not tool_ok:
        if record: return False, traj, init_state, model_xml
        return (False, frames) if render else False

    # Hang tool on frame (v1's mechanism: precise centering, lower below the
    # hook, press down). Plus a FIXED yaw rotation of the tool so the ring clears
    # the hook even when the frame seats a few degrees rotated.
    import os as _os
    _deg = float(_os.environ.get("TOOL_YAW", "-12"))
    tgq = T.mat2quat(R.from_rotvec(np.array([0, 0, 1.0]) * np.radians(_deg)).as_matrix()
                     @ T.quat2mat(tgq))

    _phase[0] = "P6_hang"
    fh = gs("frame_hang_site")
    th = gs("tool_hole1_center")
    eth = th - obs()["robot0_eef_pos"]

    # Above hang site
    ta2 = fh - eth; ta2[2] = fh[2] + 0.10
    move_ori(ta2, tgq, 1, steps=60, pg=12, og=5)

    # Fine align (tight: precisely center the ring over the hook)
    for _ in range(3):
        th = gs("tool_hole1_center"); etn = th - obs()["robot0_eef_pos"]
        tf = fh - etn; tf[2] = fh[2] + 0.05
        move_ori(tf, tgq, 1, steps=30, pg=20, og=8, pt=0.002, ot=0.01)

    # Lower onto hook (below the hang site, gentle gain)
    for _ in range(2):
        th = gs("tool_hole1_center"); etn = th - obs()["robot0_eef_pos"]
        tf = fh - etn; tf[2] = fh[2] - 0.02
        move_ori(tf, tgq, 1, steps=30, pg=5, og=3)

    # Seat tool: gentle down+grip until eef_z settles on hook, open until gripper
    # settles, firmer down+open until eef_z settles fully, terminal static hold.
    # All obs-triggered (was 20+15+40+10 fixed steps).
    if verbose:
        th = gs("tool_hole1_center"); fh = gs("frame_hang_site")
        fi = gs("frame_intersection_site")
        hookdir = (fh - fi) / (np.linalg.norm(fh - fi) + 1e-9)
        tqd = obs()["tool_quat"]
        toolx = T.quat2mat(tqd)[:, 0]
        print(f"  [seed {seed}] before seat: tool_hole={np.round(th,3)} "
              f"hang_site={np.round(fh,3)} eef_z={obs()['robot0_eef_pos'][2]:.3f}")
        print(f"  [seed {seed}] hook_dir={np.round(hookdir,3)} tool_x={np.round(toolx,3)} "
              f"frame_quat={np.round(obs()['frame_quat'],3)}")
        _m = T.quat2mat(obs()['frame_quat'])
        print(f"  CANON_FRAME seed {seed}: frame_pos={np.round(obs()['frame_pos'],4)} "
              f"z_ang={np.degrees(np.arctan2(_m[1,0],_m[0,0])):.2f}")
        # dump hook + ring geometry to design insertion
        for nm in ["frame_hook", "frame_hook_frame"]:
            try:
                gp = sim.data.geom_xpos[sim.model.geom_name2id(nm)]
                print(f"  [seed {seed}] geom {nm} pos={np.round(gp,3)}")
            except Exception:
                pass
        for sn in ["tool_hole1_center", "tool_hole1", "tool_hole2_center"]:
            try:
                sp = gs(sn)
                print(f"  [seed {seed}] site {sn}={np.round(sp,3)}")
            except Exception:
                pass
    # Terminal seating: a fixed down-press sequence (settle-detection is wrong
    # here -- we WANT to keep pushing the tool onto the hook after eef_z stalls,
    # not stop). Being terminal, the fixed steps cost little Markovian-ness: the
    # tool is already hung by the down-open phase, nothing different follows.
    for _ in range(20):
        do_step(np.array([0, 0, -0.2, 0, 0, 0, 1]))
    for _ in range(15):
        do_step(np.array([0, 0, 0, 0, 0, 0, -1]))
    for _ in range(40):
        do_step(np.array([0, 0, -0.2, 0, 0, 0, -1]))
    for _ in range(10):
        do_step(np.array([0, 0, 0, 0, 0, 0, -1]))
    if verbose:
        th = gs("tool_hole1_center"); fh = gs("frame_hang_site")
        print(f"  [seed {seed}] after seat: tool_hole={np.round(th,3)} "
              f"hang_site={np.round(fh,3)} success={env._check_success()}")

    succ = env._check_success()
    if record:
        return succ, traj, init_state, model_xml
    if render:
        return succ, frames
    return succ


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
    import sys
    n = int(sys.argv[1]) if len(sys.argv) > 1 else 50
    test(n)
