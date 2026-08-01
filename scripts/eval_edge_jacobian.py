"""P1/P2 measurement at DYNAMICALLY-REACHED tube-edge states (GPT-report predictions).

Anchors: seed-replay dataset demos (valid on collection-matched cluster). At probe
steps, snapshot sim, kick the recorded action with pos-noise (DART-style, dose rho
x random unit dir), env.step ONCE -> dynamically consistent edge state with a
deployment-consistent obs window [o_t, o']. Measure per model:
  P1  R = -n_out^T pi(s')_pos   (inward margin, raw action units) + cosine
  P2  boundary DIRECTIONAL Jacobian via obs finite difference along +-n_out (1mm,
      both frames): ||J_n||, dR/dn = -n_out^T J_n ; tangent J_t as structured ctrl
Tube coords: 300 demos progress-resampled to 120 knots -> center(p), sigma(p);
d = ||eef' - center|| / sigma, n_out = (eef' - center)/||.||.
Restore snapshot after each branch; replay continues with true actions.
Env: MODELS="tag:ckpt:loss,...", DS, NORMDS, DEMOS, PROBES, NUMSTEPS optional.
Output: numbers-only rows per model x d-band + R-vs-d slope.
"""
import os, sys
os.environ["MUJOCO_GL"] = "egl"
sys.path.insert(0, "scripts"); sys.path.insert(0, ".")
import numpy as np, torch, h5py
import robosuite
from hydra import initialize_config_dir, compose
from omegaconf import OmegaConf
from mip.datasets.robomimic_dataset import make_dataset
from mip.agent import TrainingAgent
from scripted_tool_hang_v2 import ENV_KWARGS
from collect_tool_hang_demos import extract_obs

OK = ["object", "robot0_eef_pos", "robot0_eef_quat", "robot0_gripper_qpos"]
POS = slice(44, 47)
H = 16
DS = os.environ.get("DS", "data/tool_hang_full2ins_2000.hdf5")
N_DEMOS = int(os.environ.get("DEMOS", "12"))
N_PROBES = int(os.environ.get("PROBES", "10"))
# (rho, n_kick_steps): sustained same-direction kicks walk dynamically deeper
KICKS = [(0.4, 1), (1.0, 1), (1.0, 2), (1.0, 4), (1.0, 8)]
NDIRS = 3
FD_H = 0.001  # 1mm finite-difference step
PKNOTS = 120

def cat_obs(od):
    return np.concatenate([np.asarray(od[k]).ravel() for k in OK]).astype(np.float32)

# ---------- tube stats (progress-resampled) ----------
hf = h5py.File(DS, "r")
keys = sorted(hf["data"].keys(), key=lambda k: int(k.split("_")[-1]))
grid = np.linspace(0, 1, PKNOTS)
tube = []
for k in keys[:300]:
    p = np.asarray(hf[f"data/{k}/obs/robot0_eef_pos"])
    t = np.linspace(0, 1, len(p))
    tube.append(np.stack([np.interp(grid, t, p[:, i]) for i in range(3)], 1))
tube = np.stack(tube)                      # (300, P, 3)
center = tube.mean(0)                      # (P, 3)
sigma = tube.std(0).mean(1) + 1e-9         # (P,)  isotropic scalar per knot
print(f"TUBE knots={PKNOTS} sigma_p50={np.median(sigma)*1000:.2f}mm "
      f"sigma_p10/p90={np.percentile(sigma,10)*1000:.2f}/{np.percentile(sigma,90)*1000:.2f}mm")

# ---------- models ----------
cfgdir = os.path.abspath("examples/configs")
agents = []
for spec in os.environ["MODELS"].replace("|", ":").split(","):
    tag, ckpt, loss = spec.split(":")[:3]
    ov = [f"optimization.num_steps={os.environ['NUMSTEPS']}"] if os.environ.get("NUMSTEPS") else []
    with initialize_config_dir(version_base=None, config_dir=cfgdir):
        cfg = compose(config_name="main", overrides=["task=tool_hang_ph_state_delta_legacy",
            "+task.dataset_path=" + os.path.abspath(os.environ.get("NORMDS", DS)),
            "network=chiunet", f"optimization.loss_type={loss}",
            "optimization.auto_resume=false", "log.wandb_mode=disabled"] + ov)
    OmegaConf.set_struct(cfg, False); cfg.task.obs_dim = 53
    ds = make_dataset(cfg.task)
    dev = cfg.optimization.device
    start = cfg.task.obs_steps - 1; AS = cfg.task.act_steps
    no = ds.normalizer["obs"]["state"]; na = ds.normalizer["action"]
    ag = TrainingAgent(cfg); ag.load(ckpt, load_optimizer=False); ag.eval()
    g = torch.Generator(device="cpu").manual_seed(0)
    act0 = torch.randn((1, H, 10), generator=g).to(dev)
    def make_actfn(ag=ag, no=no, na=na, ds=ds, dev=dev, act0=act0, start=start, AS=AS):
        def act_of(win):  # win: list of 2 raw obs vecs -> first pos action (raw)
            x = torch.tensor(no.normalize(np.stack(win)[None]), device=dev, dtype=torch.float32)
            with torch.no_grad():
                an = ag.sample(act_0=act0.clone(), obs=x, use_ema=True)
            return ds.undo_transform_action(na.unnormalize(an.detach().cpu().numpy())[:, start:start + AS])[0][0, :3]
        return act_of
    agents.append((tag, make_actfn()))
    print(f"MODEL {tag} loaded ({ckpt})", flush=True)

# ---------- replay + probe ----------
env = robosuite.make("ToolHang", horizon=4000, **ENV_KWARGS)
rng = np.random.RandomState(7)
rows = []  # (tag, d, R, cos, Jn, dRdn, Jt, dose)
used_demos = 0
for k in keys:
    if used_demos >= N_DEMOS: break
    dgrp = hf[f"data/{k}"]
    seed = int(dgrp.attrs.get("seed", -1))
    if seed < 0: continue
    acts = np.asarray(dgrp["actions"])
    rec_eef = np.asarray(dgrp["obs/robot0_eef_pos"])
    T = len(acts)
    np.random.seed(seed)
    ob = env.reset()
    # fidelity guard at ~step 40
    ok_replay = True
    obs_hist = [cat_obs(extract_obs(ob))]
    probe_at = set(np.linspace(12, T - 15, N_PROBES).astype(int))
    for t in range(T):
        if t == 40:
            drift = np.linalg.norm(env._get_observations()["robot0_eef_pos"] - rec_eef[t])
            if drift > 0.005:
                ok_replay = False; break
        if t in probe_at and len(obs_hist) >= 1:
            o_now = cat_obs(extract_obs(env._get_observations()))
            st = env.sim.get_state()
            win_base = [obs_hist[-1], o_now]
            a_base = {tag: act_of(win_base) for tag, act_of in agents}
            branches = [(0.0, 0, np.zeros(3))] + [(rho, ns, (lambda v: v / np.linalg.norm(v))(rng.randn(3)))
                                                  for (rho, ns) in KICKS for _ in range(NDIRS)]
            for bi, (rho, nsteps, u) in enumerate(branches):
                if rho > 0:
                    o_prev2 = o_now
                    for kk in range(nsteps):
                        a_k = acts[min(t + kk, T - 1)].copy()
                        a_k[:3] = np.clip(a_k[:3] + rho * u, -1, 1)
                        o_prev2 = o_new if kk > 0 else o_now
                        ob2, *_ = env.step(a_k)
                        o_new = cat_obs(extract_obs(ob2))
                    win_prev = o_prev2
                else:
                    o_new = o_now
                    win_prev = obs_hist[-1]
                p_idx = min(int(round((t + (nsteps if rho > 0 else 0)) / max(T - 1, 1) * (PKNOTS - 1))), PKNOTS - 1)
                eefp = o_new[POS]
                dv = eefp - center[p_idx]
                d = np.linalg.norm(dv) / sigma[p_idx]
                nout = dv / (np.linalg.norm(dv) + 1e-12)
                tang = np.cross(nout, np.array([0., 0, 1.]))
                if np.linalg.norm(tang) < 1e-6: tang = np.cross(nout, np.array([1., 0, 0]))
                tang /= np.linalg.norm(tang)
                win = [win_prev, o_new]
                win_static = [o_new, o_new]   # zero-velocity hover variant (old-probe analogue)
                # synthetic variant: displace ONLY eef dims of the clean anchor window by the
                # same achieved offset dv; object dims stay stale (reproduces the old probe)
                dv_full = o_new[POS] - o_now[POS]
                win_syn = [obs_hist[-1].copy(), o_now.copy()]
                for f_ in win_syn: f_[POS] = f_[POS] + dv_full
                for tag, act_of in agents:
                    a0 = act_of(win)
                    R = -float(nout @ a0); cosv = R / (np.linalg.norm(a0) + 1e-9)
                    a0s = act_of(win_static)
                    Rs = -float(nout @ a0s); coss = Rs / (np.linalg.norm(a0s) + 1e-9)
                    a0y = act_of(win_syn)
                    Ry = -float(nout @ a0y); cosy = Ry / (np.linalg.norm(a0y) + 1e-9)
                    Rdiff = -float(nout @ (a0 - a_base[tag]))     # differential, dynamic states
                    Rydiff = -float(nout @ (a0y - a_base[tag]))   # differential, synthetic (response_field3 protocol)
                    def disp(w, vec, h):
                        out = [w[0].copy(), w[1].copy()]
                        for f in out: f[POS] = f[POS] + h * vec
                        return out
                    Jn = (act_of(disp(win, nout, FD_H)) - act_of(disp(win, nout, -FD_H))) / (2 * FD_H)
                    Jt = (act_of(disp(win, tang, FD_H)) - act_of(disp(win, tang, -FD_H))) / (2 * FD_H)
                    rows.append((tag, d, R, cosv, np.linalg.norm(Jn), -float(nout @ Jn),
                                 np.linalg.norm(Jt), rho * max(nsteps, 1), Rs, coss, Ry, cosy,
                                 used_demos, t, bi, Rdiff, Rydiff))
                if rho > 0:
                    env.sim.set_state(st); env.sim.forward()
        ob, *_ = env.step(acts[t])
        obs_hist.append(cat_obs(extract_obs(ob)))
        if len(obs_hist) > 3: obs_hist.pop(0)
    if ok_replay:
        used_demos += 1
        print(f"DEMO {k} seed={seed} probed", flush=True)
    else:
        print(f"DEMO {k} seed={seed} SKIP drift", flush=True)

hf.close()
np.savez("logs/edgejac_rows.npz",
         tags=np.array([r[0] for r in rows]),
         vals=np.array([[float(x) for x in r[1:]] for r in rows]))
print("ROWS saved logs/edgejac_rows.npz", len(rows))
# ---------- aggregate ----------
import collections
rows_np = collections.defaultdict(list)
for r in rows: rows_np[r[0]].append(r[1:])
BANDS = [(0, 1), (1, 2), (2, 4), (4, 8), (8, 16), (16, 99)]
for tag, rr in rows_np.items():
    A = np.array(rr)  # d R cos Jn dRdn Jt dose
    for lo, hi in BANDS:
        m = (A[:, 0] >= lo) & (A[:, 0] < hi)
        if m.sum() < 4:
            print(f"EDGEJAC {tag} band=[{lo},{hi}) n={int(m.sum())} (skip)"); continue
        B = A[m]
        print(f"EDGEJAC {tag} band=[{lo},{hi}) n={int(m.sum())} "
              f"R_p50={np.median(B[:,1]):+.4f} R_mean={B[:,1].mean():+.4f} cos_p50={np.median(B[:,2]):+.3f} "
              f"Rs_p50={np.median(B[:,7]):+.4f} coss_p50={np.median(B[:,8]):+.3f} "
              f"Rsyn_p50={np.median(B[:,9]):+.4f} Rsyn_mean={B[:,9].mean():+.4f} cosyn_p50={np.median(B[:,10]):+.3f} "
              f"Jn_p50={np.median(B[:,3]):.2f} dRdn_p50={np.median(B[:,4]):+.2f} Jt_p50={np.median(B[:,5]):.2f}")
    kick = A[A[:, 6] > 0]
    if len(kick) > 10:
        sl = np.polyfit(kick[:, 0], kick[:, 1], 1)
        print(f"EDGEJAC {tag} R-vs-d slope={sl[0]:+.4f} intercept={sl[1]:+.4f} n={len(kick)}")
print("EDGEJAC-DONE")
