"""HT failure drift-onset localization + rollout-state fitting analysis.
1. Demo rotation profile per progress bin (which bins are wrist-rotation).
2. Success-band envelope: per-phase p95 tube distance of HT successes.
3. Sensitive onset per failure = first >=10-step sustained exceedance of
   the envelope; report vs crude (d>2) onset; is it in the rotation bins?
4. Commanded-vs-reference errors at rollout states (pos/rot, attenuation)
   pre-onset vs phase-matched success states. Prints HD lines."""
import glob
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

D2 = "data/tool_hang_full2ins_mp_200.hdf5"
OK = ["object", "robot0_eef_pos", "robot0_eef_quat", "robot0_gripper_qpos"]
NB = 20

with initialize_config_dir(version_base=None,
                           config_dir=os.path.abspath("examples/configs")):
    cfg = compose(config_name="main", overrides=[
        "task=tool_hang_ph_state_delta_legacy",
        "+task.dataset_path=" + os.path.abspath(D2), "network=chiunet",
        "optimization.loss_type=regression_hetero_t",
        "optimization.auto_resume=false", "log.wandb_mode=disabled"])
OmegaConf.set_struct(cfg, False)
cfg.task.obs_dim = 53
cfg.task.horizon = int(2 ** np.ceil(np.log2(cfg.task.horizon)))
Hn = int(cfg.task.horizon)
ds = make_dataset(cfg.task)
ag = TrainingAgent(cfg)
ag.load("logs/mp200_ht_s1000/models/snap_300000.pt", load_optimizer=False)
ag.eval()
sampler = get_sampler("regression_hetero_t")
no, na = ds.normalizer["obs"]["state"], ds.normalizer["action"]
dev = cfg.optimization.device
AS = int(cfg.task.act_steps)
start = int(cfg.task.obs_steps) - 1

h = h5py.File(D2, "r")
names = sorted(h["data"].keys(), key=lambda d: int(d.split("_")[1]))
PW, PA, PP = [], [], []
rotbin = np.zeros(NB)
cnt = np.zeros(NB)
for dn in names:
    o = h[f"data/{dn}/obs"]
    S = np.concatenate([np.asarray(o[k]) for k in OK], 1).astype(np.float32)
    A = np.asarray(h[f"data/{dn}/actions"], dtype=np.float32)
    L = len(S)
    for i in range(1, L - Hn):
        PW.append(no.normalize(np.stack([S[i - 1], S[i]])).reshape(-1))
        PA.append(A[i - 1 + start:i - 1 + start + AS])
        PP.append(i / L)
        b = min(int(NB * i / L), NB - 1)
        rotbin[b] += np.abs(A[i, 3:6]).sum()
        cnt[b] += 1
h.close()
PW, PA, PP = np.stack(PW), np.stack(PA), np.asarray(PP)
tP = torch.tensor(PW)
rprof = rotbin / np.maximum(cnt, 1)
rth = np.percentile(rprof, 70)
rotbins = set(np.where(rprof >= rth)[0].tolist())
print("HD rotprofile " + " ".join(f"{v:.3f}" for v in rprof), flush=True)
print(f"HD rotation-heavy bins (top30%): {sorted(rotbins)}", flush=True)


def dser_phase(obs):
    D_, P_ = [], []
    for i in range(1, len(obs)):
        q = torch.tensor(no.normalize(np.stack([obs[i - 1], obs[i]])
                                      ).reshape(-1))
        d = torch.norm(tP - q, dim=1)
        j = int(d.argmin())
        D_.append(float(d[j]))
        P_.append(float(PP[j]))
    return np.asarray(D_), np.asarray(P_), None


def cmd_errors(obs, idxs):
    W = np.stack([np.stack([obs[i - 1], obs[i]]) for i in idxs])
    xb = torch.tensor(np.stack([no.normalize(w) for w in W]), device=dev,
                      dtype=torch.float32)
    with torch.no_grad():
        a0 = torch.zeros((len(xb), Hn, 10), device=dev)
        an = sampler(cfg.optimization, ag.flow_map_ema, ag.encoder_ema, a0,
                     {"state": xb})
    pe = np.asarray(ds.undo_transform_action(
        na.unnormalize(an.cpu().numpy())[:, start:start + AS]))
    out = []
    for k, i in enumerate(idxs):
        q = torch.tensor(no.normalize(np.stack([obs[i - 1], obs[i]])
                                      ).reshape(-1))
        j = int(torch.norm(tP - q, dim=1).argmin())
        ref = PA[j]
        pl = np.abs(pe[k, :, 0:3] - ref[:, 0:3]).mean()
        rl = np.abs(pe[k, :, 3:6] - ref[:, 3:6]).mean()
        att = np.abs(pe[k, :, 3:6]).sum() / (np.abs(ref[:, 3:6]).sum()
                                             + 1e-6)
        out.append((pl, rl, att))
    return np.asarray(out)


succ_d = {b: [] for b in range(NB)}
succ_err = {b: [] for b in range(NB)}
fails = []
for f in sorted(glob.glob("logs/rd_ht/ep_*.npz")):
    z = np.load(f)
    obs, asm, seed = z["obs"], int(z["asm"]), int(z["seed"])
    D_, P_, _ = dser_phase(obs)
    if asm:
        idxs = list(range(1, len(obs) - 1, 6))
        E = cmd_errors(obs, idxs)
        for k, i in enumerate(idxs):
            b = min(int(NB * P_[i - 1]), NB - 1)
            succ_d[b].append(D_[i - 1])
            succ_err[b].append(E[k])
    else:
        fails.append((seed, obs, D_, P_))

env95 = {b: (np.percentile(succ_d[b], 95) if len(succ_d[b]) > 3 else 3.0)
         for b in range(NB)}
print("HD env95 " + " ".join(f"{env95[b]:.2f}" for b in range(NB)),
      flush=True)

pre_rows, on_rows = [], []
for seed, obs, D_, P_ in fails:
    exceed = np.array([D_[i] > env95[min(int(NB * P_[i]), NB - 1)]
                       for i in range(len(D_))])
    onset = None
    for i in range(len(exceed) - 10):
        if exceed[i:i + 10].all():
            onset = i
            break
    if onset is None:
        onset = int(np.argmax(D_))
    crude = np.where(D_ > 2.0)[0]
    crude = int(crude[0]) if len(crude) else -1
    b_on = min(int(NB * P_[onset]), NB - 1)
    print(f"HD fail {seed} onset step {onset} phase {P_[onset]:.2f} "
          f"bin {b_on} inrot {b_on in rotbins} d {D_[onset]:.2f} | crude "
          f"step {crude} phase {P_[crude] if crude >= 0 else -1:.2f}",
          flush=True)
    idxs = list(range(max(1, onset - 30), onset + 2, 2))
    E = cmd_errors(obs, [i + 1 for i in idxs])
    pre_rows.append(E)
    on_rows.append(b_on)
PRE = np.concatenate(pre_rows)
SB = np.concatenate([np.asarray(succ_err[b]) for b in set(on_rows)
                     if len(succ_err[b])])
print(f"HD preonset cmd-err pos {np.median(PRE[:, 0]):.5f} rot "
      f"{np.median(PRE[:, 1]):.5f} atten {np.median(PRE[:, 2]):.2f} "
      f"(n={len(PRE)})", flush=True)
print(f"HD succ-matched  pos {np.median(SB[:, 0]):.5f} rot "
      f"{np.median(SB[:, 1]):.5f} atten {np.median(SB[:, 2]):.2f} "
      f"(n={len(SB)})", flush=True)
print("HD done", flush=True)
