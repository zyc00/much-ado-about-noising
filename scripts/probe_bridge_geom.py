"""Bridge-geometry probe: is settle glued to transit via a shared-zero-action
junction region, and does interpolation across that bridge emit translation?
Micro-classes by r = t - c1 (closure): APP [-40,-16), SET [-10,0), SHO [2,10),
MID [14,50), PST [72,120). Reports (a) step-|a_pos| per class (premise check),
(b) pair distances + settle-kNN composition in state / chunk-label / embedding
space, (c) interpolated-state action readout (magnitude, stroke-cos, lin-blend).
Envs: CKPT, LOSS, TAG, ACT_DIM (10|11), DATATAB (1 -> print model-free tables)."""
import os

os.environ["MUJOCO_GL"] = "egl"
import sys

import numpy as np
import torch

sys.path.insert(0, "scripts"); sys.path.insert(0, ".")
import h5py
from hydra import compose, initialize_config_dir
from omegaconf import OmegaConf

from mip.agent import TrainingAgent
from mip.datasets.robomimic_dataset import make_dataset

OK = ["object", "robot0_eef_pos", "robot0_eef_quat", "robot0_gripper_qpos"]
NET = os.environ.get("NET", "chiunet")
HZ = int(os.environ.get("HZ", "16"))
with initialize_config_dir(version_base=None, config_dir=os.path.abspath("examples/configs")):
    cfg = compose(config_name="main", overrides=["task=tool_hang_ph_state_delta_legacy",
        "+task.dataset_path=data/tool_hang_full2ins_2000.hdf5", f"network={NET}",
        f"optimization.loss_type={os.environ['LOSS']}", "optimization.auto_resume=false", "log.wandb_mode=disabled"])
OmegaConf.set_struct(cfg, False); cfg.task.obs_dim = 53; cfg.task.horizon = HZ
if int(os.environ.get("ACT_DIM", "10")) == 11:
    cfg.task.act_dim = 11; cfg.task.progress_indicator = True
AD = int(os.environ.get("ACT_DIM", "10"))
ds = make_dataset(cfg.task)
ag = TrainingAgent(cfg)
dev = cfg.optimization.device
no = ds.normalizer["obs"]["state"]; na = ds.normalizer["action"]
TAG = os.environ.get("TAG", "ckpt")

CLS = ["APP", "SET", "SHO", "MID", "PST"]
BANDS = {"APP": (-40, -16), "SET": (-10, 0), "SHO": (2, 10), "MID": (14, 50), "PST": (72, 120)}

h = h5py.File("data/tool_hang_full2ins_2000.hdf5", "r")
keys = sorted(h["data"].keys(), key=lambda k: int(k.split("_")[-1]))[:150]
bank = {c: [] for c in CLS}   # entries: (demo, obs window, step |a_pos| raw, chunk label normalized)
interp_trip = []              # per demo: (W_set, W_sho, W_mid, stroke_dir_normspace)
for di, k in enumerate(keys):
    o = h[f"data/{k}/obs"]
    ov_ = np.concatenate([np.asarray(o[q]) for q in OK], axis=1).astype(np.float32)
    a = np.clip(np.asarray(h[f"data/{k}/actions"]), -1, 1).astype(np.float32)
    g = a[:, 6]; T = len(a)
    cl = [t for t in range(1, len(g)) if g[t - 1] < 0 and g[t] >= 0]
    if not cl: continue
    c1 = cl[0]
    an_ = na.normalize(a[:, :AD] if a.shape[1] >= AD else np.pad(a, ((0, 0), (0, AD - a.shape[1]))))
    for t in range(1, T):
        r = t - c1
        for c, (lo, hi) in BANDS.items():
            if lo <= r < hi:
                ch = an_[t:t + HZ]
                if len(ch) < HZ:
                    ch = np.pad(ch, ((0, HZ - len(ch)), (0, 0)), "edge")
                bank[c].append((di, np.stack([ov_[t - 1], ov_[t]]), float(np.linalg.norm(a[t, :3])), ch.reshape(-1)))
    if c1 - 7 >= 1 and c1 + 25 < T:
        sd = an_[c1 + 2:c1 + 20, :3].mean(0)
        sd = sd / (np.linalg.norm(sd) + 1e-8)
        interp_trip.append((np.stack([ov_[c1 - 7], ov_[c1 - 6]]), np.stack([ov_[c1 + 5], ov_[c1 + 6]]),
                            np.stack([ov_[c1 + 23], ov_[c1 + 24]]), sd))
h.close()

rng = np.random.RandomState(0)
CAP = 400
for c in CLS:
    if len(bank[c]) > CAP:
        bank[c] = [bank[c][i] for i in rng.choice(len(bank[c]), CAP, replace=False)]
W = {c: np.stack([e[1] for e in bank[c]]) for c in CLS}
DEMO = {c: np.array([e[0] for e in bank[c]]) for c in CLS}
AMAG = {c: np.array([e[2] for e in bank[c]]) for c in CLS}
LBL = {c: np.stack([e[3] for e in bank[c]]) for c in CLS}
SV = {c: torch.tensor(no.normalize(W[c]).reshape(len(W[c]), -1)) for c in CLS}
LV = {c: torch.tensor(LBL[c]) for c in CLS}

PAIRS = [("SET", "SHO"), ("SET", "MID"), ("SHO", "MID"), ("SET", "PST"), ("SET", "APP"), ("APP", "MID"), ("SHO", "PST")]
def pairmed(V):
    out = {}
    for c1_, c2_ in PAIRS:
        D = torch.cdist(V[c1_], V[c2_])
        out[(c1_, c2_)] = float(D.median())
    return out

def knn_comp(V, qc="SET", k=10):
    qd = DEMO[qc]
    banks = [c for c in CLS if c != qc]
    Vb = torch.cat([V[c] for c in banks]); lab = np.concatenate([[c] * len(V[c]) for c in banks])
    dm = np.concatenate([DEMO[c] for c in banks])
    D = torch.cdist(V[qc], Vb)
    D[torch.tensor(qd[:, None] == dm[None, :])] = 1e9   # exclude same-demo
    nn = D.topk(k, largest=False).indices.numpy()
    return {c: float((lab[nn] == c).mean()) for c in banks}

if os.environ.get("DATATAB", "0") == "1":
    print("BRIDGE DATA amag |a_pos| p50/p90 per class: " +
          " ".join(f"{c}={np.median(AMAG[c]):.4f}/{np.percentile(AMAG[c], 90):.4f}" for c in CLS), flush=True)
    for nm, V in [("state", SV), ("label", LV)]:
        pm = pairmed(V)
        print(f"BRIDGE DATA {nm}-space pair-dist p50: " +
              " ".join(f"{a_}-{b_}={pm[(a_, b_)]:.3f}" for a_, b_ in PAIRS), flush=True)
        kc = knn_comp(V)
        print(f"BRIDGE DATA {nm}-space SET-query 10NN comp: " +
              " ".join(f"{c}:{kc[c]:.0%}" for c in ["APP", "SHO", "MID", "PST"]), flush=True)

ag.load(os.environ["CKPT"], load_optimizer=False)
ag.encoder.eval()

def embed(batch_raw):
    out = []
    with torch.no_grad():
        for b0 in range(0, len(batch_raw), 512):
            wb = torch.tensor(no.normalize(batch_raw[b0:b0 + 512]), device=dev, dtype=torch.float32)
            out.append(ag.encoder(wb, None).reshape(len(wb), -1).cpu())
    e = torch.cat(out)
    return e / (e.norm(dim=1, keepdim=True) + 1e-8)

EV = {c: embed(W[c]) for c in CLS}
pm = pairmed(EV)
print(f"BRIDGE {TAG} emb-space pair-dist p50: " +
      " ".join(f"{a_}-{b_}={pm[(a_, b_)]:.3f}" for a_, b_ in PAIRS), flush=True)
kc = knn_comp(EV)
print(f"BRIDGE {TAG} emb-space SET-query 10NN comp: " +
      " ".join(f"{c}:{kc[c]:.0%}" for c in ["APP", "SHO", "MID", "PST"]), flush=True)
kc2 = knn_comp(EV, qc="SHO")
print(f"BRIDGE {TAG} emb-space SHO-query 10NN comp: " +
      " ".join(f"{c}:{kc2[c]:.0%}" for c in ["APP", "SET", "MID", "PST"]), flush=True)

# ---- interpolation readout ----
start = cfg.task.obs_steps - 1
def act_batch(Wq):
    outs = []
    with torch.no_grad():
        for b0 in range(0, len(Wq), 256):
            wb = torch.tensor(no.normalize(Wq[b0:b0 + 256]), device=dev, dtype=torch.float32)
            an = ag.sample(act_0=torch.randn((len(wb), HZ, cfg.task.act_dim), device=dev),
                           obs={"state": wb}, use_ema=True)
            outs.append(an[:, start, :3].cpu().numpy())
    return np.concatenate(outs)

W1 = np.stack([t[0] for t in interp_trip]); W2 = np.stack([t[1] for t in interp_trip])
W3 = np.stack([t[2] for t in interp_trip]); SDIR = np.stack([t[3] for t in interp_trip])
ends = {"SET": act_batch(W1), "SHO": act_batch(W2), "MID": act_batch(W3)}
for nm, e in ends.items():
    mag = np.linalg.norm(e, axis=1)
    cosd = (e * SDIR).sum(1) / (mag + 1e-8)
    print(f"BRIDGE {TAG} endpoint {nm}: |a_pos|p50={np.median(mag):.4f} cos(stroke)p50={np.median(cosd):.2f}", flush=True)
for pa, pb, Wa, Wb in [("SET", "SHO", W1, W2), ("SET", "MID", W1, W3), ("SHO", "MID", W2, W3)]:
    for al in (0.3, 0.5, 0.7):
        Aq = act_batch((1 - al) * Wa + al * Wb)
        mag = np.linalg.norm(Aq, axis=1)
        cosd = (Aq * SDIR).sum(1) / (mag + 1e-8)
        blend = (1 - al) * ends[pa] + al * ends[pb]
        cb = (Aq * blend).sum(1) / ((mag * np.linalg.norm(blend, axis=1)) + 1e-8)
        print(f"BRIDGE {TAG} interp {pa}-{pb} a={al}: |a_pos|p50={np.median(mag):.4f} "
              f"cos(stroke)p50={np.median(cosd):.2f} cos(linblend)p50={np.median(cb):.2f}", flush=True)
print("BRIDGE-DONE", flush=True)
