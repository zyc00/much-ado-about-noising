"""Per-replan-cycle drift dynamics from twofactor DUMPTRAJ dumps.
Splits each episode's per-step tube-distance series dser into consecutive
AS-step execution cycles and reports, per d_query band: mean per-cycle drift
dd = d_end - d_query, frac expansive, and the within-cycle growth profile
(d(j) - d_query for j=2,4,8) — i.e., where in the open-loop window the drift
is generated. Expansive events (dq in [1,4), dd > +1) get NN-phase attribution
(FAILQ-style bank). Envs: CKPT, LOSS, QD, TAG, AS (default 8), ACT_DIM."""
import glob
import os
import sys

os.environ["MUJOCO_GL"] = "egl"
import h5py
import numpy as np
import torch

sys.path.insert(0, "scripts"); sys.path.insert(0, ".")
from hydra import compose, initialize_config_dir
from omegaconf import OmegaConf

from mip.agent import TrainingAgent
from mip.datasets.robomimic_dataset import make_dataset

AS = int(os.environ.get("AS", "8"))
OK = ["object", "robot0_eef_pos", "robot0_eef_quat", "robot0_gripper_qpos"]
with initialize_config_dir(version_base=None, config_dir=os.path.abspath("examples/configs")):
    cfg = compose(config_name="main", overrides=["task=tool_hang_ph_state_delta_legacy",
        "+task.dataset_path=data/tool_hang_full2ins_2000.hdf5", "network=chiunet",
        f"optimization.loss_type={os.environ['LOSS']}", "optimization.auto_resume=false", "log.wandb_mode=disabled"])
OmegaConf.set_struct(cfg, False); cfg.task.obs_dim = 53; cfg.task.horizon = 16
if int(os.environ.get("ACT_DIM", "10")) == 11:
    cfg.task.act_dim = 11; cfg.task.progress_indicator = True
ds = make_dataset(cfg.task)
ag = TrainingAgent(cfg)
dev = cfg.optimization.device
no = ds.normalizer["obs"]["state"]

h = h5py.File("data/tool_hang_full2ins_2000.hdf5", "r")
keys = sorted(h["data"].keys(), key=lambda k: int(k.split("_")[-1]))
W, phase = [], []
PH = ["appr", "settle", "lift-ins", "post"]
for k in keys[:150]:
    o = h[f"data/{k}/obs"]
    ov_ = np.concatenate([np.asarray(o[q]) for q in OK], axis=1).astype(np.float32)
    a = np.clip(np.asarray(h[f"data/{k}/actions"]), -1, 1)
    g = a[:, 6]
    cl = [t for t in range(1, len(g)) if g[t - 1] < 0 and g[t] >= 0]
    if not cl: continue
    c1 = cl[0]
    for t in range(1, len(g), 2):
        r = t - c1
        W.append(np.stack([ov_[t - 1], ov_[t]]))
        phase.append(0 if r < -12 else (1 if r < 5 else (2 if r < 70 else 3)))
h.close()
W = np.stack(W); phase = np.array(phase)

ag.load(os.environ["CKPT"], load_optimizer=False)
ag.encoder.eval()
def embed(batch):
    out = []
    with torch.no_grad():
        for b0 in range(0, len(batch), 512):
            wb = torch.tensor(no.normalize(batch[b0:b0 + 512]), device=dev, dtype=torch.float32)
            out.append(ag.encoder(wb, None).reshape(len(wb), -1).cpu())
    e = torch.cat(out)
    return e / (e.norm(dim=1, keepdim=True) + 1e-8)
E_bank = embed(W)
TAG = os.environ.get("TAG", "ckpt")

BANDS = [(0, 1), (1, 2), (2, 4), (4, 16), (16, 1e9)]
acc = {b: [] for b in BANDS}          # (dd, prof2, prof4, prof8)
exp_states = []
for f in glob.glob(os.environ["QD"] + "/ep_*.npz"):
    z = np.load(f)
    dser = np.asarray(z["dser"], dtype=np.float64); obsw = z["obsw"]
    T = len(dser)
    for q in range(0, T - AS, AS):
        dq = dser[q]; seg = dser[q:q + AS + 1]
        dd = seg[-1] - dq
        prof = [seg[min(j, AS)] - dq for j in (2, 4, 8)]
        for b in BANDS:
            if b[0] <= dq < b[1]:
                acc[b].append((dd, *prof)); break
        if 1 <= dq < 4 and dd > 1 and q < len(obsw):
            exp_states.append(obsw[q])
for b in BANDS:
    if not acc[b]: continue
    A = np.array(acc[b])
    print(f"CYCLES {TAG} AS={AS} dq[{b[0]},{b[1] if b[1] < 1e8 else 'inf'}) n={len(A)} | "
          f"dd mean={A[:,0].mean():+.3f} p90={np.percentile(A[:,0],90):+.2f} frac_dd>0={(A[:,0]>0).mean():.2f} | "
          f"growth j2/j4/j8 = {A[:,1].mean():+.3f}/{A[:,2].mean():+.3f}/{A[:,3].mean():+.3f}", flush=True)
if exp_states:
    Q = np.stack(exp_states)
    Eq = embed(Q)
    nn = (Eq @ E_bank.T).topk(10, dim=1).indices.numpy()
    comp = [(phase[nn] == p).mean() for p in range(4)]
    print(f"CYCLES {TAG} AS={AS} EXPANSIVE events (dq in [1,4), dd>1) n={len(Q)} | NN phase "
          f"appr/settle/lift-ins/post = " + "/".join(f"{c:.0%}" for c in comp), flush=True)
else:
    print(f"CYCLES {TAG} AS={AS} EXPANSIVE events: none", flush=True)
print("CYCLES-DONE")
