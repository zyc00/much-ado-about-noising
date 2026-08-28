"""Closed-loop verification of the rank mechanism (dose-response battery).

At probe points of seeded demo replays: snapshot, apply a position kick
(rho x unit dir for n steps), then hand control to the policy for T_CONT
steps executed in act_steps chunks. Track tube deviation d_t (units of the
local demo std sigma) at each step. Restore and continue the replay.

Readout per (policy, dose): the mean d_t trajectory after the kick, and
terminal classification at T_CONT:
  returned   d_end < 2
  persisted  2 <= d_end <= d_kick + 1
  diverged   d_end > d_kick + 1
Panel-A behavior = persisted/diverged dominant; panel-B = returned dominant.

Env: MODELS="tag:ckpt:loss,...", DS (replay+tube dataset), NORMDS
(normalizer dataset = training set of the checkpoints), DEMOS, PROBES.
"""
import os
import sys

os.environ["MUJOCO_GL"] = "egl"
sys.path.insert(0, "scripts")
sys.path.insert(0, ".")

import h5py
import numpy as np
import robosuite
import torch
from hydra import compose, initialize_config_dir
from omegaconf import OmegaConf

from collect_tool_hang_demos import extract_obs
from mip.agent import TrainingAgent
from mip.datasets.robomimic_dataset import make_dataset
from scripted_tool_hang_v2 import ENV_KWARGS

OK = ["object", "robot0_eef_pos", "robot0_eef_quat", "robot0_gripper_qpos"]
POS = slice(44, 47)
H = 16
DS = os.environ.get("DS", "data/tool_hang_full2ins_2000.hdf5")
NORMDS = os.environ.get("NORMDS", "data/tool_hang_full2ins_mp200_relabel.hdf5")
N_DEMOS = int(os.environ.get("DEMOS", "8"))
N_PROBES = int(os.environ.get("PROBES", "4"))
T_CONT = int(os.environ.get("TCONT", "520"))
CHUNK = 8
KICKS = [(1.0, 1), (1.0, 2), (1.0, 4)]
if os.environ.get("KICK_SET") == "none":
    KICKS = []
elif os.environ.get("KICK_SET"):
    KICKS = [(float(p.split(":")[0]), int(p.split(":")[1]))
             for p in os.environ["KICK_SET"].split(",")]
NDIRS = 2
PKNOTS = 120


def cat_obs(od):
    return np.concatenate(
        [np.asarray(od[k]).ravel() for k in OK]).astype(np.float32)


# tube statistics from the replay dataset
hf = h5py.File(DS, "r")
keys = sorted(hf["data"].keys(), key=lambda k: int(k.split("_")[-1]))
grid = np.linspace(0, 1, PKNOTS)
tube = []
for k in keys[:300]:
    p = np.asarray(hf[f"data/{k}/obs/robot0_eef_pos"])
    t = np.linspace(0, 1, len(p))
    tube.append(np.stack([np.interp(grid, t, p[:, i]) for i in range(3)], 1))
tube = np.stack(tube)
center = tube.mean(0)
sigma = tube.std(0).mean(1) + 1e-9
print(f"TUBE sigma_p50={np.median(sigma) * 1000:.2f}mm", flush=True)

# models
cfgdir = os.path.abspath("examples/configs")
agents = []
for spec in os.environ["MODELS"].split(","):
    tag, ckpt, loss = spec.split(":")[:3]
    with initialize_config_dir(version_base=None, config_dir=cfgdir):
        cfg = compose(config_name="main", overrides=[
            "task=tool_hang_ph_state_delta_legacy",
            "+task.dataset_path=" + os.path.abspath(NORMDS),
            "network=chiunet", f"optimization.loss_type={loss}",
            "optimization.auto_resume=false", "log.wandb_mode=disabled"])
    OmegaConf.set_struct(cfg, False)
    cfg.task.obs_dim = 53
    ds = make_dataset(cfg.task)
    dev = cfg.optimization.device
    start = cfg.task.obs_steps - 1
    no = ds.normalizer["obs"]["state"]
    na = ds.normalizer["action"]
    ag = TrainingAgent(cfg)
    ag.load(ckpt, load_optimizer=False)
    ag.eval()
    g = torch.Generator(device="cpu").manual_seed(0)

    def make_chunkfn(ag=ag, no=no, na=na, ds=ds, dev=dev, g=g, start=start):
        def chunk_of(win):
            # fresh latent per chunk, matching eval_twofactor.py
            act0 = torch.randn((1, H, 10), generator=g).to(dev)
            x = torch.tensor(no.normalize(np.stack(win)[None]), device=dev,
                             dtype=torch.float32)
            with torch.no_grad():
                an = ag.sample(act_0=act0, obs=x, use_ema=True)
            return ds.undo_transform_action(
                na.unnormalize(an.detach().cpu().numpy())
                [:, start:start + CHUNK])[0]
        return chunk_of

    agents.append((tag, make_chunkfn()))
    print(f"MODEL {tag} loaded", flush=True)


ALL_EEF = []  # filled after tube stats: per-demo recorded eef paths


def dev_at(eef, tprog, T, rec_eef=None):
    # support distance (mm): min over ALL demos' recorded eef paths within a
    # progress window around the current fraction of episode progress
    frac = min(tprog / max(T - 1, 1), 1.0)
    best = 1e9
    for arr in ALL_EEF:
        c = int(frac * (len(arr) - 1))
        lo = max(0, min(c - 40, len(arr) - 1))
        hi = min(len(arr), max(c + 100, lo + 1))
        d = np.sqrt(((arr[lo:hi] - eef) ** 2).sum(1)).min()
        best = min(best, d)
    return float(best * 1000.0)


for k in keys[:200]:
    ALL_EEF.append(np.asarray(hf[f"data/{k}/obs/robot0_eef_pos"]))


TRAJ_DUMP = []
env = robosuite.make("ToolHang", horizon=6000, **ENV_KWARGS)


def reset_controller():
    try:
        arm = env.robots[0].composite_controller.part_controllers["right"]
        arm.update()
        arm.reset_goal()
    except Exception:
        try:
            env.robots[0].controller.update()
            env.robots[0].controller.reset_goal()
        except Exception:
            pass
rng = np.random.RandomState(11)
rows = []
used = 0
for k in keys:
    if used >= N_DEMOS:
        break
    dgrp = hf[f"data/{k}"]
    seed = int(dgrp.attrs.get("seed", -1))
    if seed < 0:
        continue
    acts = np.asarray(dgrp["actions"])
    rec_eef = np.asarray(dgrp["obs/robot0_eef_pos"])
    T = len(acts)
    np.random.seed(seed)
    ob = env.reset()
    ok_replay = True
    obs_hist = [cat_obs(extract_obs(ob))]
    if os.environ.get("PROBE_TS"):
        probe_at = set(int(x) for x in os.environ["PROBE_TS"].split(","))
    else:
        probe_at = set(np.linspace(20, max(T - T_CONT - 10, 21),
                                   N_PROBES).astype(int))
    for t in range(T):
        if t == 40:
            drift = np.linalg.norm(
                env._get_observations()["robot0_eef_pos"] - rec_eef[t])
            if drift > 0.005:
                ok_replay = False
                break
        if t in probe_at and len(obs_hist) >= 2:
            o_now = cat_obs(extract_obs(env._get_observations()))
            o_prevframe = obs_hist[-2]
            st = env.sim.get_state()
            branches = [(0.0, 0, np.zeros(3))]
            for rho, ns in KICKS:
                for _ in range(NDIRS):
                    u = rng.randn(3)
                    branches.append((rho, ns, u / np.linalg.norm(u)))
            for rho, nsteps, u in branches:
                # kick phase
                o_prev, o_cur = obs_hist[-1], o_now
                tt = t
                for kk in range(nsteps):
                    a_k = acts[min(tt, T - 1)].copy()
                    a_k[:3] = np.clip(a_k[:3] + rho * u, -1, 1)
                    ob2, *_ = env.step(a_k)
                    o_prev, o_cur = o_cur, cat_obs(extract_obs(ob2))
                    tt += 1
                env.sim.set_state(st)
                env.sim.forward()
                reset_controller()
                oracle = ("oracle", None)
                for tag, chunk_of in [oracle] + agents:
                    # redo kick identically for this policy
                    o_prev2, o_cur2 = o_prevframe, o_now
                    tt2 = t
                    for kk in range(nsteps):
                        a_k = acts[min(tt2, T - 1)].copy()
                        a_k[:3] = np.clip(a_k[:3] + rho * u, -1, 1)
                        ob2, *_ = env.step(a_k)
                        o_prev2, o_cur2 = o_cur2, cat_obs(extract_obs(ob2))
                        tt2 += 1
                    d0 = dev_at(o_cur2[POS], tt2, T, rec_eef)
                    dser = [d0]
                    eef_track = [o_cur2[POS].copy()]
                    amag_track = []
                    sd = 0
                    succ = False
                    while sd < T_CONT and not succ:
                        if chunk_of is None:
                            chunk = acts[min(tt2, T - 1):
                                         min(tt2, T - 1) + CHUNK]
                        else:
                            chunk = chunk_of([o_prev2, o_cur2])
                        if sd == 0 and rho == 0:
                            print(f"DIAG {tag} chunk shape "
                                  f"{np.asarray(chunk).shape} |a0| "
                                  f"{np.linalg.norm(np.asarray(chunk)[0]):.3f}"
                                  f" rec |a| "
                                  f"{np.linalg.norm(acts[min(tt2, T - 1)]):.3f}",
                                  flush=True)
                        for a in chunk:
                            amag_track.append(float(np.linalg.norm(a)))
                            ob2, *_ = env.step(a)
                            o_prev2, o_cur2 = o_cur2, cat_obs(
                                extract_obs(ob2))
                            tt2 += 1
                            sd += 1
                            dser.append(dev_at(o_cur2[POS], tt2, T, rec_eef))
                            eef_track.append(o_cur2[POS].copy())
                            if sd >= T_CONT:
                                break
                        if env._check_success():
                            succ = True
                    if os.environ.get("DUMP_TRAJ"):
                        TRAJ_DUMP.append((tag, rho * max(nsteps, 1), used,
                                          t, np.array(eef_track),
                                          np.array(amag_track)))
                    dser = np.array(dser)
                    d_end = float(np.mean(dser[-4:]))
                    dmax = float(dser.max())
                    cls = "success" if succ else "failure"
                    tr = [float(dser[min(i, len(dser) - 1)])
                          for i in (0, 8, 16, 32)]
                    rows.append((tag, rho * max(nsteps, 1), dser[0], d_end,
                                 cls, tr))
                    print(f"ROW {tag} dose={rho * max(nsteps, 1):.2f} "
                          f"d(0/8/16/32)={tr[0]:.1f}/{tr[1]:.1f}/"
                          f"{tr[2]:.1f}/{tr[3]:.1f} {cls}", flush=True)
                    env.sim.set_state(st)
                    env.sim.forward()
                    reset_controller()
        ob, *_ = env.step(acts[t])
        obs_hist.append(cat_obs(extract_obs(ob)))
        if len(obs_hist) > 3:
            obs_hist.pop(0)
    if ok_replay:
        used += 1
        print(f"DEMO {k} done", flush=True)

hf.close()

# aggregate
print("\nAGGREGATE (policy x dose): n, d0->dend mean, returned%, diverged%")
from collections import defaultdict
g = defaultdict(list)
for tag, dose, d0, dend, cls, tr in rows:
    g[(tag, dose)].append((d0, dend, cls, tr))
for (tag, dose), v in sorted(g.items()):
    n = len(v)
    tr = np.mean(np.array([x[3] for x in v]), axis=0)
    print(f"AGG {tag:10s} dose {dose:4.2f}  n{n:3d}  "
          f"d(0/8/16/32)mm {tr[0]:5.1f}/{tr[1]:5.1f}/{tr[2]:5.1f}/"
          f"{tr[3]:5.1f}")
if os.environ.get("DUMP_TRAJ"):
    np.savez(os.environ["DUMP_TRAJ"],
             tags=np.array([r[0] for r in TRAJ_DUMP]),
             doses=np.array([r[1] for r in TRAJ_DUMP]),
             demos=np.array([r[2] for r in TRAJ_DUMP]),
             probes=np.array([r[3] for r in TRAJ_DUMP]),
             **{f"traj_{i}": r[4] for i, r in enumerate(TRAJ_DUMP)},
             **{f"amag_{i}": r[5] for i, r in enumerate(TRAJ_DUMP)})
    print(f"TRAJ_DUMP saved {len(TRAJ_DUMP)} trajectories")
print("BATTERY-DONE")
