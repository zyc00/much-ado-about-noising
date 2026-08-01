"""Off-support fold probe: perturb settle states' eef position by physical offsets,
embed, and measure (a) 10-NN phase composition vs offset size, (b) action pullback
(does the predicted action oppose the offset?). Envs: CKPT, LOSS, TAG."""
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
POS = slice(44, 47)  # robot0_eef_pos columns in the 53-dim vector
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
W, phase = [], []
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
        W.append(np.stack([ov[t-1], ov[t]])); phase.append(ph)
h.close()
W = np.stack(W); phase = np.array(phase)
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
S = np.where(phase == 1)[0]
S = S[rs.choice(len(S), 300, replace=False)]
for mm in [5, 10, 20, 30]:
    d = rs.randn(len(S), 3)
    d /= np.linalg.norm(d, axis=1, keepdims=True)
    off = d * (mm / 1000.0)
    Q = W[S].copy()
    Q[:, 1, POS] += off  # perturb current frame eef pos
    Eq = embed(Q)
    sims = Eq @ E_bank.T
    for i, s in enumerate(S): sims[i, s] = -2
    nn = sims.topk(10, dim=1).indices.numpy()
    comp = np.array([(phase[nn] == p).mean() for p in range(4)])
    # pullback: predicted pos action vs -offset
    with torch.no_grad():
        wb = torch.tensor(no.normalize(Q), device=dev, dtype=torch.float32)
        emb = ag.encoder(wb, None)
        t0 = torch.zeros(len(Q), device=dev)
        an = ag.flow_map.get_velocity(t0, torch.zeros((len(Q), 16, 10), device=dev), emb)
        act = na.unnormalize(an.cpu().numpy())
    apos = act[:, 0, :3]
    pull = (apos * (-off)).sum(1) / (np.linalg.norm(apos, axis=1) * np.linalg.norm(off, axis=1) + 1e-9)
    print(f"OFFSUP {TAG} off={mm}mm | settleNN={comp[1]:.0%} (appr {comp[0]:.0%} / lift-ins {comp[2]:.0%}) "
          f"| pullback cos p50={np.median(pull):.2f} frac>0={np.mean(pull>0):.0%} |a_pos| p50={np.median(np.linalg.norm(apos,axis=1)):.4f}", flush=True)
print("OFFSUP-DONE")
