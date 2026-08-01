"""Boundary-FORMATION ladder: interpolation flip-alpha vs training snapshot
on the scripted f2i checkpoints. For matched (settle, stroke) state pairs from
the same episode, walk the linear path in raw obs-window space and record each
model's action chunk: cosine to the settle anchor chunk, cosine to the stroke
anchor chunk, output norm, and 1-NN distance to the dataset action-chunk pool
(validity). Prints CURVE lines for local harvesting."""
import glob
import os
import sys

os.environ["MUJOCO_GL"] = "egl"
sys.path.insert(0, "scripts")
sys.path.insert(0, ".")
import numpy as np
import torch
from hydra import compose, initialize_config_dir
from omegaconf import OmegaConf

from mip.agent import TrainingAgent
from mip.datasets.robomimic_dataset import make_dataset
from mip.samplers import get_sampler


DATASET = os.environ.get("IL_DATASET", "data/tool_hang_full2ins_2000.hdf5")


def load(loss):
    with initialize_config_dir(version_base=None,
                               config_dir=os.path.abspath("examples/configs")):
        cfg = compose(config_name="main", overrides=[
            "task=tool_hang_ph_state_delta_legacy",
            "+task.dataset_path=" + os.path.abspath(DATASET),
            "network=chiunet", f"optimization.loss_type={loss}",
            "optimization.auto_resume=false", "log.wandb_mode=disabled"])
    OmegaConf.set_struct(cfg, False)
    cfg.task.obs_dim = 53
    cfg.task.horizon = int(2 ** np.ceil(np.log2(cfg.task.horizon)))  # chiunet pow2
    ds = make_dataset(cfg.task)
    return cfg, ds, TrainingAgent(cfg)


cfg0, ds0, _ = load("regression")
rb = ds0.replay_buffer
print("RB keys:", list(rb.keys()), flush=True)
ends = rb.episode_ends[:]
obs_entry = rb["obs"]
try:
    S_all = obs_entry["state"][:]
except (TypeError, IndexError, KeyError):
    S_all = obs_entry[:]
A_all = rb["action"][:]
H = int(cfg0.task.horizon)  # already pow2-adjusted (16)
AD = A_all.shape[1]
TA = H
print(f"episodes {len(ends)} obs {S_all.shape} act {A_all.shape} H {H}", flush=True)

no = ds0.normalizer["obs"]["state"]
na = ds0.normalizer["action"]
dev = cfg0.optimization.device

# ---- mine (settle, stroke) pairs -----------------------------------------
pairs = []
start = 0
for e in range(min(80, len(ends))):
    end = int(ends[e])
    S, A = S_all[start:end], A_all[start:end]
    T = len(S)
    if T < 60:
        start = end
        continue
    an = np.linalg.norm(A, axis=1)
    lo, hi = T // 3, 2 * T // 3
    settle = lo + int(np.argmin(an[lo:hi]))
    seg = an[settle:min(settle + 40, T - H - 1)]
    if len(seg) < 6:
        start = end
        continue
    stroke = settle + int(np.argmax(seg))
    if stroke - settle >= 5 and settle >= 1 and stroke + H < T:
        w_set = np.stack([S[settle - 1], S[settle]])
        w_str = np.stack([S[stroke - 1], S[stroke]])
        a_set = A[settle:settle + H].reshape(-1)
        a_str = A[stroke:stroke + H].reshape(-1)
        pairs.append((w_set, w_str, a_set, a_str))
    start = end
print(f"pairs mined: {len(pairs)}", flush=True)
pairs = pairs[:50]

# action-chunk pool for validity NN (raw space)
pool = []
rngp = np.random.RandomState(0)
for _ in range(3000):
    i = rngp.randint(0, len(A_all) - H)
    pool.append(A_all[i:i + H].reshape(-1))
POOL = np.stack(pool)

MODELS = [
    ("L2", "regression", "logs/f2i_l2_s1000"),
    ("HG", "regression_hetero_gauss", "logs/f2i_hg_s1000"),
    ("HT", "regression_hetero_t", "logs/f2i_ht_s1000"),
    ("MIP", "mip", None),
]
if os.environ.get("IL_ARMS"):
    MODELS = [tuple(a.split(":")) for a in os.environ["IL_ARMS"].split(",")]
mip_dirs = sorted(glob.glob("logs/full_mip_2000*"))
print("MIP dirs:", mip_dirs, flush=True)

ALPHAS = np.linspace(0.0, 1.0, 21)

PAIRS = pairs[:30]

for name, loss, root in MODELS:
    if root is None:
        if not mip_dirs:
            print(f"LADDER {name} MISSING", flush=True)
            continue
        root = mip_dirs[-1]
    snaps = sorted(glob.glob(os.path.join(root, "models", "snap_*.pt")),
                   key=lambda p: int(p.split("_")[-1].split(".")[0]))
    if not snaps:
        snaps = [os.path.join(root, "models", "model_latest.pt")]
    # subsample to <= 8 checkpoints
    if len(snaps) > 8:
        idx = np.linspace(0, len(snaps) - 1, 8).astype(int)
        snaps = [snaps[i] for i in idx]
    cfg, ds, ag = load(loss)
    sampler = get_sampler(loss)
    for ck in snaps:
        if not os.path.exists(ck):
            continue
        step = ck.split("_")[-1].split(".")[0]
        ag.load(ck, load_optimizer=False)
        ag.eval()
        curves = []
        for ai, al in enumerate(ALPHAS):
            vals = []
            for (w_set, w_str, a_set, a_str) in PAIRS:
                w = (1 - al) * w_set + al * w_str
                x = torch.tensor(no.normalize(w[None]), device=dev,
                                 dtype=torch.float32).reshape(1, 2, 53)
                act0 = torch.zeros(1, TA, AD, device=dev)
                with torch.no_grad():
                    out = sampler(cfg.optimization, ag.flow_map_ema,
                                  ag.encoder_ema, act0, {"state": x})
                out = na.unnormalize(out.cpu().numpy()).reshape(TA, AD)[:H].reshape(-1)
                cs = np.dot(out, a_set) / (np.linalg.norm(out) * np.linalg.norm(a_set) + 1e-9)
                ct = np.dot(out, a_str) / (np.linalg.norm(out) * np.linalg.norm(a_str) + 1e-9)
                vals.append((cs, ct))
            v = np.mean(vals, 0)
            curves.append((al, v[0], v[1]))
        arr = np.array(curves)
        diff = arr[:, 1] - arr[:, 2]
        ix = np.where(np.diff(np.sign(diff)))[0]
        if len(ix):
            i = ix[0]
            flip = arr[i, 0] + 0.05 * diff[i] / (diff[i] - diff[i + 1])
        else:
            flip = float("nan")
        marg = float(np.min(np.abs(diff[6:15])))
        print(f"LADDER {name} {step} flip {flip:.3f} midmargin {marg:.3f}",
              flush=True)
print("LADDER done", flush=True)
