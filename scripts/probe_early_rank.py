"""Dense EARLY dynamics of the embedding-covariance rank (0 -> 20k steps).

The saved snapshot grid starts at 20k, by which point L2 is already at
emb_PR 3.4 and the others are at 5.0-6.5. This trains each arm from scratch
and measures emb_PR (plus the encoder-Jacobian PR and train error) on a
FIXED state batch every few steps, including step 0 (random init) — which
distinguishes "L2 drops away from initialization" from "the others build up
from it".
Env: ER_ARM (name:loss), ER_DATASET, ER_STEPS, ER_N. Prints ERANK lines.
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

from mip.agent import TrainingAgent
from mip.datasets.robomimic_dataset import make_dataset
from mip.losses import get_loss_fn

DSET = os.environ.get("ER_DATASET", "data/tool_hang_full2ins_mp_200.hdf5")
STEPS = int(os.environ.get("ER_STEPS", "20000"))
NST = int(os.environ.get("ER_N", "512"))
NJ = int(os.environ.get("ER_NJ", "16"))
NAME, LOSS = os.environ["ER_ARM"].split(":")

with initialize_config_dir(version_base=None,
                           config_dir=os.path.abspath("examples/configs")):
    cfg = compose(config_name="main", overrides=[
        "task=tool_hang_ph_state_delta_legacy",
        "+task.dataset_path=" + os.path.abspath(DSET),
        "network=chiunet", f"optimization.loss_type={LOSS}",
        "optimization.auto_resume=false", "log.wandb_mode=disabled"])
OmegaConf.set_struct(cfg, False)
cfg.task.obs_dim = 53
cfg.task.horizon = int(2 ** np.ceil(np.log2(cfg.task.horizon)))
H = int(cfg.task.horizon)
ds = make_dataset(cfg.task)
torch.manual_seed(1000)
np.random.seed(1000)
ag = TrainingAgent(cfg)
dev = cfg.optimization.device
no, na = ds.normalizer["obs"]["state"], ds.normalizer["action"]
rb = ds.replay_buffer
ends = rb.episode_ends[:]
obs_e = rb["obs"]
try:
    S_all = obs_e["state"][:]
except (TypeError, IndexError, KeyError):
    S_all = obs_e[:]
A_all = rb["action"][:]
starts = np.concatenate([[0], ends[:-1]])
Wl, Yl = [], []
for e in range(len(ends)):
    s0, e0 = int(starts[e]), int(ends[e])
    if e0 - s0 < H + 3:
        continue
    S, A = S_all[s0:e0], A_all[s0:e0]
    for i in range(1, e0 - s0 - H):
        Wl.append(no.normalize(np.stack([S[i - 1], S[i]])))
        Yl.append(na.normalize(A[i:i + H]))
Wl, Yl = np.stack(Wl), np.stack(Yl)
rng = np.random.RandomState(0)
sel = rng.choice(len(Wl), NST, replace=False)
Xf = torch.tensor(Wl[sel], device=dev, dtype=torch.float32)
Yf = torch.tensor(Yl[sel], device=dev, dtype=torch.float32)
print(f"windows {len(Wl)} probe-states {NST} arm {NAME} loss {LOSS}",
      flush=True)

fn = get_loss_fn(LOSS)
params = [p for p in list(ag.flow_map.parameters()) +
          list(ag.encoder.parameters()) if p.requires_grad]
opt = torch.optim.AdamW(params, lr=cfg.optimization.lr, weight_decay=1e-6)
brng = np.random.RandomState(7)
B = int(cfg.optimization.batch_size)


def pr(ev):
    ev = np.clip(np.asarray(ev, dtype=np.float64), 0, None)
    return float(ev.sum() ** 2 / ((ev ** 2).sum() + 1e-30))


def measure(step):
    ag.eval()
    with torch.no_grad():
        E = ag.encoder({"state": Xf}, None).reshape(NST, -1)
        t0 = torch.zeros(NST, device=dev)
        a0 = torch.zeros(NST, H, 10, device=dev)
        pa, _ = ag.flow_map.net(a0, t0, t0, ag.encoder({"state": Xf}, None))
        err = float((pa - Yf).abs().mean())
    Ec = (E - E.mean(0)).double()
    ev = torch.linalg.svdvals(Ec).cpu().numpy() ** 2
    jp = []
    for i in range(NJ):
        x = Xf[i].reshape(-1).clone()

        def f(inp):
            return ag.encoder({"state": inp.reshape(1, 2, 53)}, None).reshape(-1)

        J = torch.autograd.functional.jacobian(f, x, vectorize=True)
        s2 = (torch.linalg.svdvals(J.reshape(-1, x.numel()).double()) ** 2)
        jp.append(pr(s2.cpu().numpy()))
    print(f"ERANK {NAME} {step} emb_PR {pr(ev):.3f} jac_PR {np.mean(jp):.3f} "
          f"train_err {err:.5f} embnorm {float(E.std()):.4f}", flush=True)
    ag.train()


GRID = sorted(set([0, 10, 25, 50, 100, 150, 200, 300, 400, 600, 800, 1000,
                   1250, 1500, 1750, 2000] +
                  list(range(2500, STEPS + 1, 500))))
measure(0)
for it in range(1, STEPS + 1):
    idx = torch.tensor(brng.randint(0, len(Wl), B), device=dev)
    xb = torch.tensor(Wl, device=dev, dtype=torch.float32)[idx] \
        if False else torch.tensor(Wl[idx.cpu().numpy()], device=dev,
                                   dtype=torch.float32)
    yb = torch.tensor(Yl[idx.cpu().numpy()], device=dev, dtype=torch.float32)
    dt = torch.zeros(len(idx), device=dev)
    loss, _ = fn(cfg.optimization, ag.flow_map, ag.encoder, ag.interpolant,
                 yb, {"state": xb}, dt)
    opt.zero_grad()
    loss.backward()
    torch.nn.utils.clip_grad_norm_(params, 10.0)
    opt.step()
    if it in GRID:
        measure(it)
print("ERANK done", flush=True)
