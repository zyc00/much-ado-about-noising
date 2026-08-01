"""Collapse-sensitivity of the training objective (the restoring-force measurement).

For each arm: rank-truncate the encoder embedding at settle-window training samples to its
top-k settle-PCA directions and evaluate the arm's OWN training-loss views on (obs, action-chunk)
pairs. Reported as ratio L(k)/L(full) per view; normalization constants cancel.

Registered predictions (anti-collapse theory: denoising at scale sigma imposes an O((delta/sigma)^2)
penalty on merging states whose labels differ by delta <= sigma, vs MSE's unamplified delta^2):
  P-a: MSE view-1 ratio stays ~1 even at k=1 (loss blind to collapse -> why it collapses).
  P-b: MIP sigma=0.1 view-2 ratio at small k >> its own view-1 ratio.
  P-c: view-2 excess declines with sigma; sigma>=1 arms show view2 ~ view1 (no force) ~ MSE.
"""
import os, sys
os.environ["MUJOCO_GL"] = "egl"
sys.path.insert(0, "scripts"); sys.path.insert(0, ".")
import numpy as np, h5py, torch
from hydra import initialize_config_dir, compose
from omegaconf import OmegaConf
from mip.datasets.robomimic_dataset import make_dataset
from mip.agent import TrainingAgent

OK = ["object", "robot0_eef_pos", "robot0_eef_quat", "robot0_gripper_qpos"]
def load(loss):
    with initialize_config_dir(version_base=None, config_dir=os.path.abspath("examples/configs")):
        cfg = compose(config_name="main", overrides=["task=tool_hang_ph_state_delta_legacy",
            "+task.dataset_path=" + os.path.abspath("data/tool_hang_full2ins_2000.hdf5"), "network=chiunet",
            f"optimization.loss_type={loss}", "optimization.auto_resume=false", "log.wandb_mode=disabled"])
    OmegaConf.set_struct(cfg, False); cfg.task.obs_dim = 53
    ds = make_dataset(cfg.task)
    return cfg, ds, TrainingAgent(cfg)

cfg, ds, _ = load("regression")
no = ds.normalizer["obs"]["state"]; na = ds.normalizer["action"]; dev = cfg.optimization.device

# ---- settle-window training samples: obs window (2,53) + action chunk (16,10) in TRAINING space
h = h5py.File("data/tool_hang_full2ins_2000.hdf5", "r")
keys = sorted(h["data"].keys(), key=lambda k: int(k.split("_")[-1]))
WS, AC = [], []
for k in keys[:400]:
    o = h[f"data/{k}/obs"]
    ov = np.concatenate([np.asarray(o[q]) for q in OK], axis=1).astype(np.float32)
    a = np.clip(np.asarray(h[f"data/{k}/actions"]), -1, 1); g = a[:, 6]
    cl = [t for t in range(1, len(g)) if g[t-1] < 0 and g[t] >= 0]
    if not cl: continue
    c1 = cl[0]; T = len(g)
    for t in range(max(1, c1-12), min(c1+5, T-16)):
        WS.append(np.stack([ov[t-1], ov[t]])); AC.append(a[t:t+16])
    if len(WS) >= 2000: break
h.close()
WS = np.stack(WS[:2000]); AC = np.stack(AC[:2000])
rc = ds.rotation_transformer.forward(AC[:, :, 3:6].reshape(-1, 3)).reshape(len(AC), 16, 6)
A10 = na.normalize(np.concatenate([AC[:, :, :3], rc, AC[:, :, 6:7]], axis=2).astype(np.float32))
print(f"settle samples={len(WS)}", flush=True)

SIGMA = {"MIPs1": 0.1, "orig_sig001": 0.01, "orig_sig003": 0.03, "orig_sig03": 0.3,
         "orig_sig10": 1.0, "sig20": 2.0, "sig50": 5.0}
MODELS = [("MSE", "regression", "logs/full_regression_2000/models/model_latest.pt"),
          ("MIPs1", "mip", "logs/full_mip_2000_s1/models/model_latest.pt")]
for d in ["orig_sig001", "orig_sig003", "orig_sig03", "orig_sig10", "sig20", "sig50"]:
    p = f"logs/{d}/models/model_latest.pt"
    if os.path.exists(p): MODELS.append((d, "mip", p))

KS = [1, 2, 4, 8, 16, 32]
for name, loss, ck in MODELS:
    cfg, ds, ag = load(loss)
    ag.load(ck, load_optimizer=False); ag.eval()
    enc = ag.encoder_ema; fm = ag.flow_map_ema if hasattr(ag, "flow_map_ema") else ag.flow_map
    # raw embeddings + settle PCA basis
    embs = []
    for i in range(0, len(WS), 256):
        x = torch.tensor(no.normalize(WS[i:i+256]), device=dev, dtype=torch.float32)
        with torch.no_grad():
            embs.append(enc({"state": x}, None))
    E = torch.cat(embs)                      # (n, ...) raw embedding, native shape
    Ef = E.reshape(len(E), -1)
    mu = Ef.mean(0, keepdim=True)
    Xc = (Ef - mu).cpu().numpy().astype(np.float64)
    C = (Xc.T @ Xc) / len(Xc)
    ev, V = np.linalg.eigh(C); order = np.argsort(ev)[::-1]; V = torch.tensor(V[:, order], device=dev, dtype=torch.float32)

    def losses_with(Ek):
        Aten = torch.tensor(np.asarray(A10), device=dev, dtype=torch.float32)
        sig = SIGMA.get(name); tts = None if sig is None else 1.0 - sig
        l1s, l2s, lcs = [], [], []
        torch.manual_seed(0)
        for i in range(0, len(Ek), 256):
            e = Ek[i:i+256].reshape(*E[i:i+256].shape)
            a = Aten[i:i+256]
            t0 = torch.zeros(len(a), device=dev)
            with torch.no_grad():
                y0 = fm.get_velocity(t0, torch.zeros_like(a), e)
                l1s.append(((y0 - a) ** 2).mean().item() * len(a))
                lcs.append(torch.log1p(((y0 - a) / 0.2) ** 2).mean().item() * len(a))
                if sig is not None:
                    acc = 0.0
                    for r in range(4):
                        yin = a + sig * torch.randn_like(a)
                        y1 = fm.get_velocity(torch.full((len(a),), tts, device=dev), yin, e)
                        acc += (((y1 - a) / sig) ** 2).mean().item()
                    l2s.append(acc / 4 * len(a))
        return sum(l1s) / len(Ek), (sum(l2s) / len(Ek) if l2s else None), sum(lcs) / len(Ek)

    L1f, L2f, LCf = losses_with(Ef)
    r1, r2, rc = [], [], []
    for k in KS:
        Ek = mu + (Ef - mu) @ V[:, :k] @ V[:, :k].T
        a_, b_, c_ = losses_with(Ek)
        r1.append(a_ / L1f); r2.append(b_ / L2f if b_ is not None else None); rc.append(c_ / LCf)
    line1 = f"COLSENS {name} view1 (Lfull={L1f:.5f}):" + "".join(f" k{k}={r:.2f}" for k, r in zip(KS, r1))
    print(line1, flush=True)
    print(f"COLSENS {name} CAUCHYview (Lfull={LCf:.5f}):" + "".join(f" k{k}={r:.2f}" for k, r in zip(KS, rc)), flush=True)
    if L2f is not None:
        line2 = f"COLSENS {name} view2 sigma={SIGMA[name]} (Lfull={L2f:.5f}):" + "".join(f" k{k}={r:.2f}" for k, r in zip(KS, r2))
        print(line2, flush=True)
print("COLSENS-DONE")
