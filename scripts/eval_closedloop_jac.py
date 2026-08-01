"""CLOSED-LOOP Jacobian: at matched kicked states s, execute each policy k steps,
measure d'=d(F(s,pi(s))) -> per-band one-step drift and lambda_cl = slope(d' on d).
Baseline arm: replaying the EXPERT recorded action at the same off-tube state (pure transport).

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
CL_STEPS = [1, 3]  # closed-loop probe depths
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
            return ds.undo_transform_action(na.unnormalize(an.detach().cpu().numpy())[:, start:start + AS])[0][0]
        return act_of
    agents.append((tag, make_actfn()))
    print(f"MODEL {tag} loaded ({ckpt})", flush=True)


# ---------- replay + closed-loop probe ----------
env = robosuite.make("ToolHang", horizon=4000, **ENV_KWARGS)
rng = np.random.RandomState(7)
rows = []  # (tag, d0, d1, ncl, dose)
used_demos = 0

def tube_d(eefp, tidx, T):
    # TIME-FREE lateral distance: nearest knot within a +-12-knot window around
    # the nominal progress (robust to schedule lag after kicks)
    p_idx = min(int(round(tidx / max(T - 1, 1) * (PKNOTS - 1))), PKNOTS - 1)
    lo, hi = max(0, p_idx - 12), min(PKNOTS, p_idx + 13)
    dists = np.linalg.norm(eefp[None] - center[lo:hi], axis=1) / sigma[lo:hi]
    return float(dists.min())

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
    ok_replay = True
    obs_hist = [cat_obs(extract_obs(ob))]
    probe_at = set(np.linspace(12, T - 25, N_PROBES).astype(int))
    for t in range(T):
        if t == 40:
            if np.linalg.norm(env._get_observations()["robot0_eef_pos"] - rec_eef[t]) > 0.005:
                ok_replay = False; break
        if t in probe_at:
            o_now = cat_obs(extract_obs(env._get_observations()))
            st0 = env.sim.get_state()
            for rho, nsteps in KICKS:
                for _dir in range(NDIRS):
                    u = rng.randn(3); u /= np.linalg.norm(u)
                    # kick to the probe state
                    o_prev = o_now; o_new = o_now
                    for kk in range(nsteps):
                        a_k = acts[min(t + kk, T - 1)].copy()
                        a_k[:3] = np.clip(a_k[:3] + rho * u, -1, 1)
                        o_prev = o_new
                        ob2, *_ = env.step(a_k)
                        o_new = cat_obs(extract_obs(ob2))
                    st_kick = env.sim.get_state()
                    d0 = tube_d(o_new[POS], t + nsteps, T)
                    win0 = [o_prev, o_new]
                    # arms: each model closed-loop + expert-transport baseline
                    for tag, act_of in agents + [("EXPT", None)]:
                        env.sim.set_state(st_kick); env.sim.forward()
                        wp_, wn_ = win0[0], win0[1]
                        for m in range(max(CL_STEPS)):
                            if act_of is None:
                                a7 = acts[min(t + nsteps + m, T - 1)]
                            else:
                                a7 = act_of([wp_, wn_])
                            ob3, *_ = env.step(np.asarray(a7, dtype=np.float64))
                            wp_ = wn_; wn_ = cat_obs(extract_obs(ob3))
                            if (m + 1) in CL_STEPS:
                                d1 = tube_d(wn_[POS], t + nsteps + m + 1, T)
                                rows.append((tag, d0, d1, m + 1, rho * nsteps))
                    env.sim.set_state(st0); env.sim.forward()
            # ensure replay continues from clean state
        ob, *_ = env.step(acts[t])
        obs_hist.append(cat_obs(extract_obs(ob)))
        if len(obs_hist) > 3: obs_hist.pop(0)
    if ok_replay:
        used_demos += 1
        print(f"DEMO {k} seed={seed} done", flush=True)
    else:
        print(f"DEMO {k} SKIP drift", flush=True)

hf.close()
import collections
BANDS = [(1, 2), (2, 4), (4, 8), (8, 16), (16, 99)]
by = collections.defaultdict(list)
for tag, d0, d1, ncl, dose in rows:
    by[(tag, ncl)].append((d0, d1))
for (tag, ncl), v in sorted(by.items()):
    A = np.array(v)
    for lo, hi in BANDS:
        m = (A[:, 0] >= lo) & (A[:, 0] < hi)
        if m.sum() < 10: continue
        B = A[m]; dd = B[:, 1] - B[:, 0]
        print(f"CLJAC {tag} k={ncl} band=[{lo},{hi}) n={m.sum()} dd_p50={np.median(dd):+.3f} dd_mean={dd.mean():+.3f} frac_shrink={100*(dd<0).mean():.0f}%")
    m2 = (A[:, 0] >= 1) & (A[:, 0] < 16)
    if m2.sum() > 30:
        sl = np.polyfit(A[m2, 0], A[m2, 1], 1)
        print(f"CLJAC {tag} k={ncl} lambda_cl(1-16sig)={sl[0]:+.3f} intercept={sl[1]:+.3f} n={int(m2.sum())}")
print("CLJAC-DONE")
