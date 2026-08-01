"""Effective conditioning-Lipschitz on support->failure paths (tests the Jacobian-
regularization note vs the projection story vs kappa=1 x0-caveat).
For each hetero-t FAIL near-gate state s': find nearest demo support state s0 (normalized
obs window space); walk the segment s(u)=s0+u(s'-s0), u in {0,.25,.5,.75,1}; evaluate:
  REG  : hetero-t single-pass action a(s(u))
  MIP1 : MIP step-1 action
  MIP2 : MIP step-2 action F(s, t* a0(s))
Report: effective Lipschitz slope ||a(u)-a(0)|| / ||s(u)-s(0)|| at u=1 (and u=.5),
and anchoring cos( a(1), a(0) ) — how much the off-support action stays near the
nearest-support behavior. Note predicts MIP2 slope << MIP1 ~ REG; x0-caveat predicts
MIP2 ~ MIP1."""
import os
os.environ["MUJOCO_GL"] = "egl"
import numpy as np, torch, sys, h5py
sys.path.insert(0, "scripts"); sys.path.insert(0, ".")
from hydra import initialize_config_dir, compose
from omegaconf import OmegaConf
from mip.datasets.robomimic_dataset import make_dataset
from mip.agent import TrainingAgent
DSP = os.environ.get("DSP", "/home/jigu/.cache/huggingface/hub/datasets--ChaoyiPan--mip-dataset/blobs/c5cb01b119a109a3d4842ca515906e86f1f35a8ee1a333d22b3503c1a513a8f4")
OK = ["object", "robot0_eef_pos", "robot0_eef_quat", "robot0_gripper_qpos"]
BP = slice(7, 10); FP = slice(21, 24)
def load(loss, net="chiunet"):
    with initialize_config_dir(version_base=None, config_dir=os.path.abspath("examples/configs")):
        cfg = compose(config_name="main", overrides=["task=tool_hang_ph_state_delta_legacy",
            "+task.dataset_path=" + DSP, f"network={net}",
            f"optimization.loss_type={loss}", "optimization.auto_resume=false", "log.wandb_mode=disabled"])
    OmegaConf.set_struct(cfg, False); cfg.task.obs_dim = 53; cfg.task.horizon = 10 if net == "mlp" else 16
    ds = make_dataset(cfg.task)
    return cfg, ds, TrainingAgent(cfg)
cfg, ds, ag = load("regression_hetero_t")
dev = cfg.optimization.device
no = ds.normalizer["obs"]["state"]
# demo support bank (held near states, with prev frames)
h = h5py.File(DSP, "r")
keys = sorted(h["data"].keys(), key=lambda k: int(k.split("_")[-1]))
offs, bank = [], []
for k in keys[:200]:
    o = h[f"data/{k}/obs"]
    a = np.clip(np.asarray(h[f"data/{k}/actions"]), -1, 1).astype(np.float32); g = a[:, 6]; T = len(a)
    cl = [t for t in range(1, T) if g[t-1] < 0 and g[t] >= 0]
    op = [t for t in range(1, T) if g[t-1] >= 0 and g[t] < 0]
    if not cl: continue
    c1 = cl[0]; r1 = next((t for t in op if t > c1), None)
    if not r1: continue
    ov = np.concatenate([np.asarray(o[q]) for q in OK], axis=1).astype(np.float32)
    offs.append(ov[r1-1, FP] - ov[r1-1, BP])
    for t in range(c1 + 5, r1, 2):
        v = ov[t, FP] - (ov[t, BP] + offs[-1])
        if np.linalg.norm(v[:2]) < 0.10:
            bank.append(np.stack([ov[t-1], ov[t]]))
h.close()
off = np.median(np.stack(offs), 0)
BK = no.normalize(np.stack(bank))
BKt = torch.tensor(BK.reshape(len(BK), -1))
print(f"bank {len(BK)}", flush=True)
def held_range(o, a):
    L = min(len(o), len(a)); gc = a[:L, 6]
    run = 0; g0 = None; gend = L
    for t in range(L):
        run = run + 1 if gc[t] >= 0 else 0
        if run >= 15 and g0 is None: g0 = t - 14
        if g0 is not None and gc[t] < 0: gend = t; break
    return g0, gend
fails = []
for name in ["hheterot_s5", "hheterot_s1000"]:
    z = np.load(f"analysis/traj_vis/human_{name}.npz")
    i = 0
    while f"ep{i}_obs" in z.files:
        o = z[f"ep{i}_obs"]; a = z[f"ep{i}_act"]; m = z[f"ep{i}_meta"]
        g0, gend = held_range(o, a)
        if g0 is not None and not int(m[1]):
            v = o[g0:gend, FP] - (o[g0:gend, BP] + off)
            lat = np.linalg.norm(v[:, :2], axis=1); alt = v[:, 2]
            ts = [t for t in range(1, len(v)) if 0.010 <= lat[t] < 0.080 and 0.005 <= alt[t] < 0.120]
            for t in ts[::4][:20]:
                fails.append(no.normalize(np.stack([o[g0 + t - 1], o[g0 + t]])[None])[0])
        i += 1
print(f"fail states {len(fails)}", flush=True)
US = [0.0, 0.5, 1.0]
MODELS = [("HT-reg", "regression_hetero_t", os.environ["CK_HT"], "reg"),
          ("MIP", "mip", os.environ["CK_MIP"], "mip")]
if os.environ.get("CK_MLP"):
    MODELS.append(("MLP-reg", "regression", os.environ["CK_MLP"], "mlpreg"))
for mname, loss, ck, kind in MODELS:
    cfg2, ds2, ag2 = load(loss, net=("mlp" if kind == "mlpreg" else "chiunet"))
    NH = 10 if kind == "mlpreg" else 16
    ag2.load(ck, load_optimizer=False); ag2.eval()
    outs = {u: {"1": [], "2": []} for u in US}
    for s1 in fails:
        q = torch.tensor(s1.reshape(1, -1), dtype=torch.float32)
        j = int(torch.cdist(q, BKt)[0].argmin())
        s0 = BK[j]
        for u in US:
            w = torch.tensor((1 - u) * s0 + u * s1, device=dev, dtype=torch.float32)[None]
            with torch.no_grad():
                e = ag2.encoder(w, None)
                t0 = torch.zeros(1, device=dev)
                a0 = ag2.flow_map.get_velocity(t0, torch.zeros((1, NH, 10), device=dev), e)
                outs[u]["1"].append(a0[0].cpu().numpy())
                if kind == "mip":
                    tt = torch.full((1,), 0.9, device=dev)
                    a2 = ag2.flow_map.get_velocity(tt, a0, e)
                    outs[u]["2"].append(a2[0].cpu().numpy())
        outs.setdefault("sdist", []).append(float(np.linalg.norm(s1 - s0)))
    sd = np.array(outs["sdist"])
    for stepk in (["1"] if kind == "reg" else ["1", "2"]):
        A0 = np.stack(outs[0.0][stepk]); A5 = np.stack(outs[0.5][stepk]); A1 = np.stack(outs[1.0][stepk])
        slope5 = np.linalg.norm((A5 - A0).reshape(len(A0), -1), axis=1) / (0.5 * sd + 1e-9)
        slope1 = np.linalg.norm((A1 - A0).reshape(len(A0), -1), axis=1) / (sd + 1e-9)
        cos1 = [float(np.dot(a.ravel(), b.ravel()) / (np.linalg.norm(a) * np.linalg.norm(b) + 1e-9)) for a, b in zip(A1, A0)]
        print(f"BLIP {mname} step{stepk}: eff-Lip u=.5 p50={np.median(slope5):.3f} | u=1 p50={np.median(slope1):.3f}"
              f" | cos(a_off, a_support) p50={np.median(cos1):+.2f}", flush=True)
print("BLIP-DONE")
