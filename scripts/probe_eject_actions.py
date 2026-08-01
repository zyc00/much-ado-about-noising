"""Phase-resolved + counterfactual action comparison around MSE failures.

Uses the captured rollouts (analysis/failvids/<tag>_trajs.npz: P eef pos,
D tube distance, W raw obs windows, O outcome). Defines state sets:
  insert   : eef z < Z_INS (the precision phase), all episodes
  preonset : the 40 steps before the first d>=2 crossing (failures only)
  posteject: steps after the first d>=4 crossing (failures only)
Evaluates ALL four policies on each set and reports, per set:
  |da| between (L2-200,MIP-200) and (L2-2k,MIP-2k), split into
  pos(0:3) / rot6d(3:9) / grip(9);
  SIGNED mean per position dim for (MIP - L2) — the direction of the
  correction MIP would have applied at MSE's own states;
  |a - a_NNdemo| per policy (who is closer to the expert action).
Prints EJACT lines.
"""
import os
import sys

os.environ["MUJOCO_GL"] = "egl"
sys.path.insert(0, "scripts")
sys.path.insert(0, ".")
import numpy as np
import torch
from hydra import compose, initialize_config_dir
from omegaconf import OmegaConf
from scipy.spatial import cKDTree

from mip.agent import TrainingAgent
from mip.datasets.robomimic_dataset import make_dataset
from mip.samplers import get_sampler

Z_INS = float(os.environ.get("Z_INS", "0.86"))

ARMS = [
    ("L2-200", "regression", "logs/mp200_l2_s1000/models/snap_300000.pt",
     "data/tool_hang_full2ins_mp_200.hdf5", "l2mp200v2"),
    ("MIP-200", "mip", "logs/mp200_mip_s1000/models/snap_300000.pt",
     "data/tool_hang_full2ins_mp_200.hdf5", "mipmp200"),
    ("L2-2k", "regression", "logs/mpf2i_l2_s1000/models/snap_300000.pt",
     "data/tool_hang_full2ins_mp_2000.hdf5", "l2mp2k"),
    ("MIP-2k", "mip", "logs/mpf2i_mip_s1000/models/snap_300000.pt",
     "data/tool_hang_full2ins_mp_2000.hdf5", "mipmp2k"),
]

POL = {}
for name, loss, ck, dset, tag in ARMS:
    with initialize_config_dir(version_base=None,
                               config_dir=os.path.abspath("examples/configs")):
        cfg = compose(config_name="main", overrides=[
            "task=tool_hang_ph_state_delta_legacy",
            "+task.dataset_path=" + os.path.abspath(dset),
            "network=chiunet", f"optimization.loss_type={loss}",
            "optimization.auto_resume=false", "log.wandb_mode=disabled"])
    OmegaConf.set_struct(cfg, False)
    cfg.task.obs_dim = 53
    H = int(2 ** np.ceil(np.log2(cfg.task.horizon)))
    cfg.task.horizon = H
    ds = make_dataset(cfg.task)
    ag = TrainingAgent(cfg)
    ag.load(ck, load_optimizer=False)
    ag.eval()
    POL[name] = (cfg, ds, ag, get_sampler(loss), H)

# demo action pool (from the 200-demo set; the 2k set is a superset in style)
cfg0, ds0, _, _, H0 = POL["L2-200"]
rb = ds0.replay_buffer
ends = rb.episode_ends[:]
obs_entry = rb["obs"]
try:
    S_all = obs_entry["state"][:]
except (TypeError, IndexError, KeyError):
    S_all = obs_entry[:]
A_all = rb["action"][:]
no0 = ds0.normalizer["obs"]["state"]
na0 = ds0.normalizer["action"]
starts = np.concatenate([[0], ends[:-1]])
WD, AD_ = [], []
for e in range(len(ends)):
    s0, e0 = int(starts[e]), int(ends[e])
    if e0 - s0 < H0 + 3:
        continue
    S, A = S_all[s0:e0], A_all[s0:e0]
    for i in range(1, e0 - s0 - H0, 2):
        WD.append(no0.normalize(np.stack([S[i - 1], S[i]])).reshape(-1))
        AD_.append(A[i:i + 8].reshape(-1))
WD = np.stack(WD)
AD_ = np.stack(AD_)      # raw (unnormalized) demo action chunks, 8x10
demo_tree = cKDTree(WD)
print(f"demo pool {len(WD)}", flush=True)

start = 1  # obs_steps - 1


def act_of(name, W_raw):
    cfg, ds, ag, sampler, H = POL[name]
    no = ds.normalizer["obs"]["state"]
    na = ds.normalizer["action"]
    outs = []
    for i in range(0, len(W_raw), 256):
        w = W_raw[i:i + 256]
        x = torch.tensor(np.stack([no.normalize(wi) for wi in w]),
                         device=cfg.optimization.device, dtype=torch.float32)
        with torch.no_grad():
            a0 = torch.zeros((len(w), H, 10), device=x.device)
            an = sampler(cfg.optimization, ag.flow_map_ema, ag.encoder_ema,
                         a0, {"state": x})
        outs.append(na.unnormalize(an.detach().cpu().numpy())[:, start:start + 8])
    return np.concatenate(outs)


def grp(a, b):
    """componentwise |da|: pos-mean, pos-median(per-state), rot, grip"""
    d = np.abs(a - b)
    pm = d[..., 0:3].mean(axis=(1, 2))
    return (d[..., 0:3].mean(), float(np.median(pm)),
            d[..., 3:9].mean(), d[..., 9].mean())


for src_name, _, _, _, tag in ARMS:
    d = np.load(f"analysis/failvids/{tag}_trajs.npz")
    seeds = sorted({int(k[1:]) for k in d.files if k.startswith("W")})
    sets = {"insert_on": [], "insert_off": [], "preonset": [],
            "posteject": []}
    nfail = 0
    for sd in seeds:
        W, D, P = d[f"W{sd}"], d[f"D{sd}"], d[f"P{sd}"]
        ok = bool(d[f"O{sd}"][0])
        n = min(len(W), len(D), len(P))
        low = P[:n, 2] < Z_INS
        sets["insert_on"].append(W[:n][low & (D[:n] < 2)])
        sets["insert_off"].append(W[:n][low & (D[:n] >= 2)])
        if not ok:
            nfail += 1
            hit2 = np.where(D[:n] >= 2)[0]
            hit4 = np.where(D[:n] >= 4)[0]
            if len(hit2):
                i0 = int(hit2[0])
                sets["preonset"].append(W[max(0, i0 - 40):i0])
            if len(hit4):
                i4 = int(hit4[0])
                sets["posteject"].append(W[i4:min(n, i4 + 120)])
    for sname, chunks in sets.items():
        if not chunks:
            continue
        Wc = np.concatenate(chunks)
        if len(Wc) < 10:
            continue
        if len(Wc) > 350:
            Wc = Wc[np.linspace(0, len(Wc) - 1, 350).astype(int)]
        acts = {n_: act_of(n_, Wc) for n_ in POL}
        # nearest demo action chunk for each state (normalized-window space)
        xq = np.stack([no0.normalize(w).reshape(-1) for w in Wc])
        _, nn = demo_tree.query(xq)
        a_nn = AD_[nn].reshape(len(Wc), 8, 10)
        p200 = grp(acts["L2-200"], acts["MIP-200"])
        p2k = grp(acts["L2-2k"], acts["MIP-2k"])
        sgn = np.median((acts["MIP-200"] - acts["L2-200"])[..., 0:3],
                        axis=(0, 1))
        nnd = " ".join(
            f"{n_}:{np.abs(acts[n_] - a_nn).mean():.4f}/"
            f"{np.median(np.abs(acts[n_] - a_nn).mean(axis=(1, 2))):.4f}"
            for n_ in POL)
        mags = " ".join(
            f"{n_}:{np.abs(acts[n_][..., 0:3]).mean():.4f}/"
            f"{np.median(np.abs(acts[n_][..., 0:3]).mean(axis=(1, 2))):.4f}"
            for n_ in POL)
        print(f"EJACT2 {src_name} {sname} n={len(Wc)} nfail={nfail} "
              f"d200[pos {p200[0]:.4f} med {p200[1]:.4f} rot {p200[2]:.4f} "
              f"grip {p200[3]:.4f}] "
              f"d2k[pos {p2k[0]:.4f} med {p2k[1]:.4f}] "
              f"medsign_MIPminusL2_pos[{sgn[0]:+.4f} {sgn[1]:+.4f} {sgn[2]:+.4f}] "
              f"|a-demoNN|mean/med[{nnd}] |apos|mean/med[{mags}]", flush=True)
print("EJACT done", flush=True)
