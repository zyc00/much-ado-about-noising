"""Human-demo (ToolHang PH) encoder spectrum battery for the local human MSE arms.

Windows: settle1 = first-grasp settle (r in [-12,5) around first gripper closure c1);
align = insert-align hover ([rel1-30, rel1) before the first release after c1 — the human
lethal window); all = pooled bank. Instruments: per-window embedding PCA (PR/erank/top1),
encoder-Jacobian kappa10/PR(s2)/k90 + COLCOS (top-10 column pairwise |cos| and vs-u1) at 30
window states. OBS_MASK set per model to match training (mask applied inside encoder fwd).
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

h = h5py.File(DSP, "r")
keys = sorted(h["data"].keys(), key=lambda k: int(k.split("_")[-1]))
W, wid = [], []   # wid: 0 other, 1 settle1, 2 align
for k in keys:
    o = h[f"data/{k}/obs"]
    ov = np.concatenate([np.asarray(o[q]) for q in OK], axis=1).astype(np.float32)
    a = np.clip(np.asarray(h[f"data/{k}/actions"]), -1, 1); g = a[:, 6]
    cl = [t for t in range(1, len(g)) if g[t-1] < 0 and g[t] >= 0]
    if not cl: continue
    c1 = cl[0]
    rels = [t for t in range(c1 + 1, len(g)) if g[t-1] >= 0 and g[t] < 0]
    rel1 = rels[0] if rels else None
    for t in range(1, len(g), 2):
        wd = 0
        if -12 <= t - c1 < 5: wd = 1
        elif rel1 is not None and rel1 - 30 <= t < rel1: wd = 2
        W.append(np.stack([ov[t-1], ov[t]])); wid.append(wd)
h.close()
W = np.stack(W); wid = np.array(wid)
print(f"bank={len(W)} settle1={(wid==1).sum()} align={(wid==2).sum()}", flush=True)

def specsum(ev):
    ev = np.clip(ev, 0, None); ev = ev[ev > 1e-12 * max(ev.max(), 1e-30)]
    p = ev / ev.sum()
    return float(1.0/(p**2).sum()), float(np.exp(-(p*np.log(p)).sum())), float(np.sort(p)[-1])

NOBASE = ",".join(str(i) for i in range(0, 14))
OLD = "/home/jigu/projects/much-ado-about-noising-old/checkpoints"
MODELS = [("lhbase", "regression", "logs/lhbase/models/model_latest.pt", None),
          ("lhnobase", "regression", "logs/lhnobase/models/model_latest.pt", NOBASE)]
if os.environ.get("HIST_ONLY"):
    MODELS = [
        ("hMIP_s5", "mip", f"{OLD}/tool_hang_ph_state_mip_chiunet_256_seed5_success82.pt", None),
        ("hMSE_s5", "regression", f"{OLD}/tool_hang_ph_state_regression_chiunet_256_seed5_success52.pt", None),
        ("hMIP_s5001", "mip", f"{OLD}/tool_hang_ph_state_mip_chiunet_256_seed5001_success80.pt", None),
        ("hMSE_s5001", "regression", f"{OLD}/tool_hang_ph_state_regression_chiunet_256_seed5001_success60.pt", None),
    ]
rng = np.random.RandomState(0)
for name, loss, ck, mask in MODELS:
    if mask: os.environ["OBS_MASK"] = mask
    else: os.environ.pop("OBS_MASK", None)
    cfg, ds, ag = load(loss)
    ag.load(ck, load_optimizer=False); ag.eval()
    enc = ag.encoder_ema
    outs = []
    for i in range(0, len(W), 512):
        x = torch.tensor(no.normalize(W[i:i+512]), device=dev, dtype=torch.float32)
        with torch.no_grad():
            e = enc({"state": x}, None)
        outs.append(e.reshape(len(x), -1))
    E = torch.cat(outs)
    En = (E / (E.norm(dim=1, keepdim=True) + 1e-9)).cpu().numpy().astype(np.float64)
    line = f"HUMANSPEC {name} PCA:"
    for wn, sel in [("all", np.arange(len(En))), ("settle1", np.where(wid == 1)[0]), ("align", np.where(wid == 2)[0])]:
        X = En[sel] - En[sel].mean(0, keepdims=True)
        pr, er, top1 = specsum(np.linalg.eigvalsh((X.T @ X) / len(X)))
        line += f" | {wn}: PR={pr:.2f} erank={er:.2f} top1={top1:.0%}"
    print(line, flush=True)
    def fea(inp):
        return enc({"state": inp.reshape(1, 2, 53)}, None).reshape(-1)
    for wn, sel in [("settle1", np.where(wid == 1)[0]), ("align", np.where(wid == 2)[0])]:
        canon = W[rng.choice(sel, 30, replace=False)]
        k10s, prj, k90s, pcs, u1s = [], [], [], [], []
        for w in canon:
            x = torch.tensor(no.normalize(w[None]), device=dev, dtype=torch.float32).reshape(1, -1)
            J = torch.autograd.functional.jacobian(fea, x, vectorize=True).squeeze(1).detach().cpu().numpy()
            sv = np.linalg.svd(J, compute_uv=False); e2 = sv ** 2
            cs = np.cumsum(e2) / e2.sum()
            k90s.append(int(np.searchsorted(cs, 0.90) + 1))
            prj.append(float(e2.sum()**2 / (e2**2).sum()))
            k10s.append(float(sv[0] / sv[min(9, len(sv)-1)]))
            gn = np.linalg.norm(J, axis=0); nz = np.where(gn > 1e-8 * gn.max())[0]
            top = nz[np.argsort(gn[nz])[-10:]]
            C = J[:, top] / (np.linalg.norm(J[:, top], axis=0, keepdims=True) + 1e-12)
            S = np.abs(C.T @ C); iu = np.triu_indices(10, 1)
            pcs.append(float(S[iu].mean()))
            U = np.linalg.svd(J, full_matrices=False)[0]
            u1s.append(float(np.abs(C.T @ U[:, 0]).mean()))
        print(f"HUMANSPEC {name} JAC {wn}: k90={np.median(k90s):.0f} PR={np.median(prj):.2f} k10={np.median(k10s):.1f} | COLCOS pair={np.median(pcs):.2f} vs-u1={np.median(u1s):.2f}", flush=True)
print("HUMANSPEC-DONE")
