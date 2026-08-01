"""Does human MSE show cross-window action FUSION anywhere (not necessarily settle)?

Bank: human demos labeled by task-leg windows via gripper events (c1 = frame grasp,
r1 = frame release, c2 = tool grasp): approach1 / settle1 / carry1 / align1(pre-r1 hover) /
between / settle2 / carry2 / align2(final hover) / other.

Queries: dither states from FAILED hMSE_s5 rollouts (last 120 steps of episodes that timed
out without success — the align-hover failure window).

Instruments:
 (a) query 10-NN window composition in the hMSE encoder embedding (fusion = cross-window
     neighbors, esp. align1<->align2 or align<->carry);
 (b) on-support control: bank align1/align2 states' own 10-NN composition;
 (c) XPINT-style action check at queries: cosine of the policy's executed action (recorded
     in the dump) with (i) own-window neighbor label mean, (ii) cross-window neighbor label
     mean, (iii) all-NN label mean — fusion = alignment with cross-window labels.
"""
import os, sys
os.environ["MUJOCO_GL"] = "egl"
sys.path.insert(0, "scripts"); sys.path.insert(0, ".")
import numpy as np, h5py, torch
from hydra import initialize_config_dir, compose
from omegaconf import OmegaConf
from mip.datasets.robomimic_dataset import make_dataset
from mip.agent import TrainingAgent

DSP = "/home/jigu/.cache/huggingface/hub/datasets--ChaoyiPan--mip-dataset/blobs/c5cb01b119a109a3d4842ca515906e86f1f35a8ee1a333d22b3503c1a513a8f4"
OK = ["object", "robot0_eef_pos", "robot0_eef_quat", "robot0_gripper_qpos"]
OLD = "/home/jigu/projects/much-ado-about-noising-old/checkpoints"
WNAMES = ["approach1", "settle1", "carry1", "align1", "between", "settle2", "carry2", "align2", "other"]

def load(loss):
    with initialize_config_dir(version_base=None, config_dir=os.path.abspath("examples/configs")):
        cfg = compose(config_name="main", overrides=["task=tool_hang_ph_state_delta_legacy",
            "+task.dataset_path=" + DSP, "network=chiunet",
            f"optimization.loss_type={loss}", "optimization.auto_resume=false", "log.wandb_mode=disabled"])
    OmegaConf.set_struct(cfg, False); cfg.task.obs_dim = 53
    ds = make_dataset(cfg.task)
    return cfg, ds, TrainingAgent(cfg)

cfg, ds, _ = load("regression")
no = ds.normalizer["obs"]["state"]; dev = cfg.optimization.device

# ---- bank with two-leg window labels + raw action labels
h = h5py.File(DSP, "r")
keys = sorted(h["data"].keys(), key=lambda k: int(k.split("_")[-1]))
W, wid, LBL = [], [], []
for k in keys:
    o = h[f"data/{k}/obs"]
    ov = np.concatenate([np.asarray(o[q]) for q in OK], axis=1).astype(np.float32)
    a = np.clip(np.asarray(h[f"data/{k}/actions"]), -1, 1); g = a[:, 6]; T = len(a)
    cl = [t for t in range(1, T) if g[t-1] < 0 and g[t] >= 0]
    op = [t for t in range(1, T) if g[t-1] >= 0 and g[t] < 0]
    if not cl: continue
    c1 = cl[0]
    r1 = next((t for t in op if t > c1), None)
    c2 = next((t for t in cl if r1 and t > r1), None)
    for t in range(1, T, 2):
        w = 8  # other
        if t < c1 - 12: w = 0
        elif c1 - 12 <= t < c1 + 5: w = 1
        elif r1 and c1 + 5 <= t < r1 - 30: w = 2
        elif r1 and r1 - 30 <= t < r1: w = 3
        elif r1 and c2 and r1 <= t < c2 - 12: w = 4
        elif c2 and c2 - 12 <= t < c2 + 5: w = 5
        elif c2 and c2 + 5 <= t < T - 40: w = 6
        elif c2 and T - 40 <= t: w = 7
        W.append(np.stack([ov[t-1], ov[t]])); wid.append(w); LBL.append(a[t])
h.close()
W = np.stack(W); wid = np.array(wid); LBL = np.stack(LBL)
print("bank=" + str(len(W)) + " | " + ", ".join(f"{n}={int((wid==i).sum())}" for i, n in enumerate(WNAMES)), flush=True)

# ---- queries from failed rollouts
z = np.load("analysis/traj_vis/human_hMSE_s5.npz")
qs, qacts = [], []
i = 0
while f"ep{i}_obs" in z.files:
    meta = z[f"ep{i}_meta"]
    if int(meta[0]) == 0:  # failed episode
        obs = z[f"ep{i}_obs"]; act = z[f"ep{i}_act"]
        for t in range(max(1, len(obs) - 120), len(obs) - 1):
            qs.append(np.stack([obs[t-1], obs[t]])); qacts.append(act[min(t, len(act)-1)])
    i += 1
qs = np.stack(qs[::3]); qacts = np.stack(qacts[::3])
print(f"queries={len(qs)} (dither windows of failed eps)", flush=True)

for name, loss, ck in [("hMSE_s5", "regression", f"{OLD}/tool_hang_ph_state_regression_chiunet_256_seed5_success52.pt"),
                       ("hMIP_s5", "mip", f"{OLD}/tool_hang_ph_state_mip_chiunet_256_seed5_success82.pt")]:
    cfg, ds, ag = load(loss)
    ag.load(ck, load_optimizer=False); ag.eval()
    enc = ag.encoder_ema
    def emb(X):
        outs = []
        for i in range(0, len(X), 512):
            x = torch.tensor(no.normalize(X[i:i+512]), device=dev, dtype=torch.float32)
            with torch.no_grad():
                e = enc({"state": x}, None)
            outs.append(e.reshape(len(x), -1))
        E = torch.cat(outs)
        return (E / (E.norm(dim=1, keepdim=True) + 1e-9)).cpu().numpy()
    EB = emb(W); EQ = emb(qs)
    nb = np.argsort(EQ @ EB.T, axis=1)[:, -10:]
    comp = np.zeros(9)
    for i in range(9): comp[i] = (wid[nb] == i).mean()
    print("FUSE queryNN comp: " + ", ".join(f"{n}={comp[i]:.0%}" for i, n in enumerate(WNAMES) if comp[i] >= 0.02), flush=True)
    # on-support FULL confusion matrix: every window's NN composition over all windows
    # (fusion condition = label-similar + state-different, NOT only small/constant windows:
    #  carry1<->carry2 and approach1<->approach2 are candidates too)
    print(f"FUSE {name} confusion (rows=query window, cols=NN window, % of 10-NN):", flush=True)
    hdr = "        " + " ".join(f"{n[:6]:>7s}" for n in WNAMES)
    print(hdr, flush=True)
    for wq in range(9):
        idx = np.where(wid == wq)[0]
        if len(idx) < 20: continue
        idx = idx[::max(1, len(idx)//300)][:300]
        sim = EB[idx] @ EB.T
        for j, q in enumerate(idx): sim[j, q] = -2
        nbb = np.argsort(sim, axis=1)[:, -10:]
        row = [ (wid[nbb] == c).mean() for c in range(9) ]
        print(f"{WNAMES[wq][:8]:>8s} " + " ".join(f"{v:6.0%} " if v >= 0.005 else "    .  " for v in row), flush=True)
    if name != "hMSE_s5":
        continue  # queries/actions are hMSE rollouts
    # action-level: executed action vs neighbor label means (pos xyz, raw action space)
    own_cos, cross_cos, all_cos = [], [], []
    qwin = wid[nb[:, -1]]  # top-1 neighbor's window as the query's presumptive window
    for i in range(len(qs)):
        a_exec = qacts[i][:3]
        if np.linalg.norm(a_exec) < 1e-6: continue
        nbl = LBL[nb[i]][:, :3]; nw = wid[nb[i]]
        m_all = nbl.mean(0)
        own = nbl[nw == qwin[i]]; cross = nbl[nw != qwin[i]]
        def cos(u, v):
            return float(u @ v / (np.linalg.norm(u) * np.linalg.norm(v) + 1e-9))
        if len(own): own_cos.append(cos(a_exec, own.mean(0)))
        if len(cross): cross_cos.append(cos(a_exec, cross.mean(0)))
        all_cos.append(cos(a_exec, m_all))
    print(f"FUSE action-cos: own-window p50={np.median(own_cos):+.2f} (n={len(own_cos)}) | cross-window p50={np.median(cross_cos):+.2f} (n={len(cross_cos)}) | all-NN p50={np.median(all_cos):+.2f}", flush=True)
print("FUSE-DONE")
