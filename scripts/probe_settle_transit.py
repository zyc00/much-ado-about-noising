"""Directed fold probe: interpolate settle states toward their nearest transit
state (raw obs space) and measure, per interpolation fraction alpha:
(a) 10-NN phase composition, (b) whose label the predicted action matches
(cos to settle-label vs transit-label). Envs: CKPT, LOSS, TAG."""
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
with initialize_config_dir(version_base=None, config_dir=os.path.abspath("examples/configs")):
    cfg = compose(config_name="main", overrides=["task=tool_hang_ph_state_delta_legacy",
        "+task.dataset_path=data/tool_hang_full2ins_2000.hdf5", "network=chiunet",
        f"optimization.loss_type={os.environ['LOSS']}", "optimization.auto_resume=false", "log.wandb_mode=disabled"])
OmegaConf.set_struct(cfg, False); cfg.task.obs_dim = 53; cfg.task.horizon = 16
ds = make_dataset(cfg.task)
ag = TrainingAgent(cfg)
dev = cfg.optimization.device
no = ds.normalizer["obs"]["state"]; na = ds.normalizer["action"]

h = h5py.File("data/tool_hang_full2ins_2000.hdf5", "r")
keys = sorted(h["data"].keys(), key=lambda k: int(k.split("_")[-1]))
W, phase, LBL = [], [], []
for k in keys[:150]:
    o = h[f"data/{k}/obs"]
    ov = np.concatenate([np.asarray(o[q]) for q in OK], axis=1).astype(np.float32)
    a = np.clip(np.asarray(h[f"data/{k}/actions"]), -1, 1); g = a[:, 6]
    cl = [t for t in range(1, len(g)) if g[t-1] < 0 and g[t] >= 0]
    if not cl: continue
    c1 = cl[0]
    for t in range(1, len(g), 2):
        r = t - c1
        ph = 0 if r < -12 else (1 if r < 5 else (2 if r < 70 else 3))
        W.append(np.stack([ov[t-1], ov[t]])); phase.append(ph); LBL.append(a[t])
h.close()
W = np.stack(W); phase = np.array(phase); LBL = np.stack(LBL)
ag.load(os.environ["CKPT"], load_optimizer=False)
ag.encoder.eval(); ag.flow_map.eval()

def embed(batch):
    out = []
    with torch.no_grad():
        for b0 in range(0, len(batch), 512):
            wb = torch.tensor(no.normalize(batch[b0:b0+512]), device=dev, dtype=torch.float32)
            out.append(ag.encoder(wb, None).reshape(len(wb), -1).cpu())
    e = torch.cat(out)
    return e / (e.norm(dim=1, keepdim=True) + 1e-8)

E_bank = embed(W)
TAG = os.environ.get("TAG", "ckpt")
rs = np.random.RandomState(0)
S = np.where(phase == 1)[0]; T2 = np.where(phase == 2)[0]
S = S[rs.choice(len(S), 250, replace=False)]
# nearest transit state per settle query, raw z-scored obs distance (current frame)
Wf = W[:, 1]; mu, sd = Wf.mean(0), Wf.std(0) + 1e-6
Zs = (Wf[S] - mu) / sd; Zt = (Wf[T2] - mu) / sd
Dst = torch.cdist(torch.tensor(Zs), torch.tensor(Zt))
jt = T2[Dst.argmin(dim=1).numpy()]
for alpha in [0.0, 0.1, 0.2, 0.3, 0.5]:
    Q = (1 - alpha) * W[S] + alpha * W[jt]
    Eq = embed(Q)
    sims = Eq @ E_bank.T
    for i, s in enumerate(S): sims[i, s] = -2
    nn = sims.topk(10, dim=1).indices.numpy()
    comp = np.array([(phase[nn] == p).mean() for p in range(4)])
    with torch.no_grad():
        wb = torch.tensor(no.normalize(Q), device=dev, dtype=torch.float32)
        emb = ag.encoder(wb, None)
        t0 = torch.zeros(len(Q), device=dev)
        an = ag.flow_map.get_velocity(t0, torch.zeros((len(Q), 16, 10), device=dev), emb)
        act = na.unnormalize(an.cpu().numpy())[:, 0]
    def coss(A, B):
        return (A * B).sum(1) / (np.linalg.norm(A, axis=1) * np.linalg.norm(B, axis=1) + 1e-9)
    cS = coss(act[:, :6], LBL[S][:, :6]); cT = coss(act[:, :6], LBL[jt][:, :6])
    mag = np.linalg.norm(act[:, :3], axis=1)
    magS = np.median(np.linalg.norm(LBL[S][:, :3], axis=1)); magT = np.median(np.linalg.norm(LBL[jt][:, :3], axis=1))
    print(f"ST {TAG} a={alpha:.1f} | NN settle={comp[1]:.0%} transit={comp[2]:.0%} appr={comp[0]:.0%} "
          f"| cos(act,settleLbl) p50={np.median(cS):.2f} cos(act,transitLbl) p50={np.median(cT):.2f} "
          f"| |a_pos| p50={np.median(mag):.4f} (settleLbl {magS:.4f} / transitLbl {magT:.4f})", flush=True)
print("ST-DONE")
