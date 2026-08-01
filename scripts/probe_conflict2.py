"""Conflict-style channel-dominance test, v2 — hardened rerun.

Fixes the recorded weaknesses of probe_conflict.py:
  1. BOTH swap directions per pair (H_ab = obj A + prop B; H_ba = obj B +
     prop A). Object-win for H_ab is proximity to f(A); for H_ba to f(B).
     A direction-consistent dominance estimate removes hybrid-construction
     asymmetry.
  2. PAIR-DISTANCE stratification: near pairs (obs distance below median)
     give hybrids closer to the support; far pairs are stronger conflicts.
     If dominance ordering holds for near pairs, it is not an artifact of
     absurd off-manifold composites.
  3. PHASE stratification by trajectory progress of the object frame
     (early = align, progress < 0.5; late = insert/hang).
  4. SUB-CHANNEL conflict: eef_pos-only swap (dims 44:47 from B, all else
     A). lam_pos = relative proximity to f(A); low lam_pos = the policy's
     output responds to the eef_pos channel — behavioral counterpart of the
     Jacobian eef-pos energy-share result.
  5. n = 800 sampled pairs (~400 informative), fixed base seed + a second
     seed for a stability check on the headline rate.
Prints CONF2 lines. Env: CF_ARMS, CF_N.
"""
import os
import sys

os.environ["MUJOCO_GL"] = "egl"
sys.path.insert(0, "scripts")
sys.path.insert(0, ".")
import h5py
import numpy as np
import torch
from hydra import compose, initialize_config_dir
from omegaconf import OmegaConf

from mip.agent import TrainingAgent
from mip.datasets.robomimic_dataset import make_dataset
from mip.samplers import get_sampler

ARMS = [tuple(a.split(":")) for a in os.environ["CF_ARMS"].split(",")]
NPAIR = int(os.environ.get("CF_N", "800"))
HELD = "data/tool_hang_full2ins_mp_20k.hdf5"
OK = ["object", "robot0_eef_pos", "robot0_eef_quat", "robot0_gripper_qpos"]

h = h5py.File(HELD, "r")
names = sorted(h["data"].keys(), key=lambda d: int(d.split("_")[1]))[2000:2600]
rng = np.random.RandomState(0)
Wl, Ph = [], []
for dn in names:
    o = h[f"data/{dn}/obs"]
    S = np.concatenate([np.asarray(o[k]) for k in OK], 1).astype(np.float32)
    if len(S) < 20:
        continue
    for i in rng.choice(np.arange(1, len(S) - 9), 3, replace=False):
        Wl.append(np.stack([S[i - 1], S[i]]))
        Ph.append(i / len(S))
h.close()
Wl, Ph = np.stack(Wl), np.asarray(Ph)
print(f"pool {len(Wl)} windows", flush=True)


def report(name, tag, win_ab, win_ba, lam, mask):
    m = mask
    if m.sum() < 20:
        return
    w = 0.5 * (win_ab[m].mean() + win_ba[m].mean())
    print(f"CONF2 {name} {tag} n={int(m.sum())} object_win {w:.3f} "
          f"(ab {win_ab[m].mean():.3f} ba {win_ba[m].mean():.3f}) "
          f"lam_mean {lam[m].mean():.3f}", flush=True)


for seed in (0, 1):
    r2 = np.random.RandomState(100 + seed)
    ia = r2.choice(len(Wl), NPAIR)
    ib = r2.choice(len(Wl), NPAIR)
    A, B = Wl[ia], Wl[ib]
    phA = Ph[ia]
    # obs-space pair distance (raw, z-scored per dim over the pool)
    mu, sd = Wl.mean((0, 1)), Wl.std((0, 1)) + 1e-6
    dpair = (np.abs((A - B) / sd)).mean(axis=(1, 2))
    Hab, Hba, Hpos = A.copy(), B.copy(), A.copy()
    Hab[:, :, 44:53] = B[:, :, 44:53]
    Hba[:, :, 44:53] = A[:, :, 44:53]
    Hpos[:, :, 44:47] = B[:, :, 44:47]

    for name, loss, ck, dset in ARMS:
        with initialize_config_dir(version_base=None,
                                   config_dir=os.path.abspath(
                                       "examples/configs")):
            cfg = compose(config_name="main", overrides=[
                "task=tool_hang_ph_state_delta_legacy",
                "+task.dataset_path=" + os.path.abspath(dset),
                "network=chiunet", f"optimization.loss_type={loss}",
                "optimization.auto_resume=false", "log.wandb_mode=disabled"])
        OmegaConf.set_struct(cfg, False)
        cfg.task.obs_dim = 53
        cfg.task.horizon = int(2 ** np.ceil(np.log2(cfg.task.horizon)))
        H = int(cfg.task.horizon)
        ds = make_dataset(cfg.task)
        ag = TrainingAgent(cfg)
        ag.load(ck, load_optimizer=False)
        ag.eval()
        sampler = get_sampler(loss)
        no, na = ds.normalizer["obs"]["state"], ds.normalizer["action"]
        dev = cfg.optimization.device

        def act(WS):
            out = []
            for i in range(0, len(WS), 256):
                xb = torch.tensor(
                    np.stack([no.normalize(w) for w in WS[i:i + 256]]),
                    device=dev, dtype=torch.float32)
                with torch.no_grad():
                    a0 = torch.zeros((len(xb), H, 10), device=dev)
                    an = sampler(cfg.optimization, ag.flow_map_ema,
                                 ag.encoder_ema, a0, {"state": xb})
                out.append(np.asarray(ds.undo_transform_action(
                    na.unnormalize(an.cpu().numpy())[:, 1:9])))
            return np.concatenate(out)

        pA, pB = act(A), act(B)
        pHab, pHba, pHpos = act(Hab), act(Hba), act(Hpos)
        dAB = np.abs(pA - pB).mean(axis=(1, 2))
        info = dAB > np.median(dAB)
        dA_ab = np.abs(pHab - pA).mean(axis=(1, 2))
        dB_ab = np.abs(pHab - pB).mean(axis=(1, 2))
        dA_ba = np.abs(pHba - pA).mean(axis=(1, 2))
        dB_ba = np.abs(pHba - pB).mean(axis=(1, 2))
        win_ab = (dA_ab < dB_ab).astype(float)      # obj wins: follows A
        win_ba = (dB_ba < dA_ba).astype(float)      # obj wins: follows B
        lam = dB_ab / (dA_ab + dB_ab + 1e-12)
        # eef_pos-only swap: lam_pos near 1 = ignores eef_pos change
        dA_p = np.abs(pHpos - pA).mean(axis=(1, 2))
        dB_p = np.abs(pHpos - pB).mean(axis=(1, 2))
        lam_pos = dB_p / (dA_p + dB_p + 1e-12)

        near = dpair < np.median(dpair[info])
        report(name, f"s{seed}:all", win_ab, win_ba, lam, info)
        report(name, f"s{seed}:near", win_ab, win_ba, lam, info & near)
        report(name, f"s{seed}:far", win_ab, win_ba, lam, info & ~near)
        report(name, f"s{seed}:early", win_ab, win_ba, lam, info & (phA < 0.5))
        report(name, f"s{seed}:late", win_ab, win_ba, lam, info & (phA >= 0.5))
        print(f"CONF2 {name} s{seed}:pos lam_pos_mean "
              f"{lam_pos[info].mean():.3f} p10 "
              f"{np.percentile(lam_pos[info], 10):.3f}", flush=True)
print("CONF2 done", flush=True)
