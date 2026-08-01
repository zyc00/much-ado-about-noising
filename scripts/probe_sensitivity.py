"""Action-sensitivity scores at anchor states along HUMAN tool-hang demos.

From each anchor (demo d, step t): restore sim state, perturb only a_t by
+/- delta along K random unit action directions, replay the expert actions
for H steps, and estimate S(x_t) = ||d phi(x_{t+H}) / d a_t||_F by finite
difference, phi = object obs vector + eef pos/quat. Two perturbation seeds,
two (delta, H) settings. Reports Spearman(seed0, seed1) per setting,
Spearman(setting A, setting B), Spearman(S, normalized time), and saves
anchors + S + HIGH/LOW median split to analysis/chunk/anchor_sens.npz.
Prints SENS lines.
"""
import os

os.environ["MUJOCO_GL"] = "egl"
import sys

sys.path.insert(0, "scripts")
sys.path.insert(0, ".")
import h5py
import numpy as np
import robomimic.utils.env_utils as EnvUtils
import robomimic.utils.file_utils as FileUtils
import robomimic.utils.obs_utils as ObsUtils
from scipy.stats import spearmanr

ObsUtils.initialize_obs_utils_with_obs_specs({"obs": {
    "low_dim": ["object", "robot0_eef_pos", "robot0_eef_quat",
                "robot0_gripper_qpos"], "rgb": []}})

D = os.environ.get("SENS_DATA", "data/tool_hang_human_lowdim_up.hdf5")
SETTINGS = [(0.01, 10), (0.03, 20)]
KDIR, NDEMO, NANCH = 4, 25, 8

env_meta = FileUtils.get_env_metadata_from_dataset(dataset_path=D)
env = EnvUtils.create_env_from_metadata(env_meta=env_meta, render=False,
                                        render_offscreen=False,
                                        use_image_obs=False)

h = h5py.File(D, "r")
names = sorted(h["data"].keys(), key=lambda d: int(d.split("_")[1]))[:NDEMO]


def phi():
    ob = env.get_observation()
    return np.concatenate([np.asarray(ob["object"]).ravel(),
                           np.asarray(ob["robot0_eef_pos"]).ravel(),
                           np.asarray(ob["robot0_eef_quat"]).ravel()])


def replay(state, acts):
    env.reset()
    env.reset_to({"states": state})
    for a in acts:
        env.step(a)
    return phi()


anchors = []  # (demo_idx, t, tnorm, states_t, A)
for di, dn in enumerate(names):
    St = np.asarray(h[f"data/{dn}/states"])
    A = np.asarray(h[f"data/{dn}/actions"]).astype(np.float64)
    T = len(A)
    for f in np.linspace(0.05, 0.85, NANCH):
        t = int(f * T)
        anchors.append((di, t, t / T, St[t], A))
h.close()
print(f"SENS {len(anchors)} anchors from {NDEMO} demos", flush=True)

S_all = np.zeros((len(anchors), len(SETTINGS), 2))
for ai, (di, t, tn, st, A) in enumerate(anchors):
    for si, (delta, Hh) in enumerate(SETTINGS):
        Hh = min(Hh, len(A) - t - 1)
        seq = A[t:t + Hh]
        for seed in (0, 1):
            rng = np.random.default_rng(1000 * seed + ai)
            g2 = []
            for _ in range(KDIR):
                u = rng.standard_normal(seq.shape[1])
                u /= np.linalg.norm(u)
                sp = seq.copy()
                sp[0] = np.clip(sp[0] + delta * u, -1, 1)
                fp = replay(st, sp)
                sm = seq.copy()
                sm[0] = np.clip(sm[0] - delta * u, -1, 1)
                fm = replay(st, sm)
                g2.append(((fp - fm) / (2 * delta)) ** 2)
            S_all[ai, si, seed] = np.sqrt(
                seq.shape[1] * np.mean([g.sum() for g in g2]))
    if ai % 20 == 0:
        print(f"SENS progress {ai}/{len(anchors)}", flush=True)

tn = np.array([a[2] for a in anchors])
for si, (delta, Hh) in enumerate(SETTINGS):
    r_seed = spearmanr(S_all[:, si, 0], S_all[:, si, 1]).statistic
    print(f"SENS setting delta={delta} H={Hh}: S p10/50/90 "
          f"{np.percentile(S_all[:, si].mean(1), [10, 50, 90]).round(3)} "
          f"spearman(seed0,seed1) {r_seed:.3f}", flush=True)
r_set = spearmanr(S_all[:, 0].mean(1), S_all[:, 1].mean(1)).statistic
S = S_all.mean(axis=(1, 2))
r_time = spearmanr(S, tn).statistic
print(f"SENS spearman(settingA,settingB) {r_set:.3f}", flush=True)
print(f"SENS spearman(S, tnorm) {r_time:.3f}", flush=True)
med = np.median(S)
grp = (S >= med).astype(int)  # 1=HIGH
os.makedirs("analysis/chunk", exist_ok=True)
np.savez("analysis/chunk/anchor_sens.npz",
         demo=np.array([a[0] for a in anchors]),
         t=np.array([a[1] for a in anchors]), tnorm=tn,
         S_all=S_all, S=S, group=grp, median=med)
print(f"SENS saved analysis/chunk/anchor_sens.npz | HIGH {grp.sum()} "
      f"LOW {(1 - grp).sum()} median {med:.3f}", flush=True)
print("SENS done", flush=True)
