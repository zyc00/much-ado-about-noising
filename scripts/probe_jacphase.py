"""Encoder-Jacobian spectra across ALL trajectory phases (not just settle).
20 demo states per phase window: approach (c1-40..c1-15), settle (c1-12..c1-1),
transit (c1+20..c1+60), align (c1+72..t_push), insert/push (t_push..T-2).
Reports per model x phase: PR(s^2), smax/smed, s1, colPR, -HEIGHT ablation ratio."""
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
no = ds.normalizer["obs"]["state"]; dev = cfg.optimization.device

h5 = h5py.File("data/tool_hang_full2ins_2000.hdf5", "r")
keys = sorted(h5["data"].keys(), key=lambda k: int(k.split("_")[-1]))
rng = np.random.RandomState(7)
PH_WINDOWS = {}
demo_idx = rng.choice(200, 20, replace=False)
states = {p: [] for p in ["approach", "settle", "transit", "align", "insert"]}
for di in demo_idx:
    d = h5[f"data/demo_{di}"]
    a = np.clip(np.asarray(d["actions"]), -1, 1); g = a[:, 6]
    cl = [t for t in range(1, len(g)) if g[t-1] < 0 and g[t] >= 0]
    if not cl: continue
    c1 = cl[0]; T = len(g)
    fz = np.asarray(d["obs/object"])[:, 23]
    peak = fz[c1+60:].max()
    plateau = np.where(fz >= peak - 0.005)[0]; plateau = plateau[plateau >= c1+60]
    tp = int(plateau[-1]) if len(plateau) else T - 30
    ov = np.concatenate([np.asarray(d["obs"][q]) for q in OK], axis=1).astype(np.float32)
    for pname, (lo, hi) in [("approach", (c1-40, c1-15)), ("settle", (c1-12, c1-1)),
                             ("transit", (c1+20, c1+60)), ("align", (c1+72, tp)),
                             ("insert", (tp, T-2))]:
        if hi <= lo + 1: continue
        t = rng.randint(max(1, lo), min(hi, T-1))
        states[pname].append(np.stack([ov[t-1], ov[t]]))
h5.close()

Z53 = [2, 16, 30, 9, 23, 37, 46]
HDIMS = Z53 + [d + 53 for d in Z53]

import os as _o
if _o.environ.get("ONE_CKPT"):
    MODELS = [(_o.environ.get("ONE_TAG", "one"), _o.environ.get("ONE_LOSS", "regression"), _o.environ["ONE_CKPT"])]
else:
    MODELS = [("MSE", "regression", "logs/full_regression_2000/models/model_latest.pt"),
              ("MIPs1", "mip", "logs/full_mip_2000_s1/models/model_latest.pt"),
              ("fadehint", "regression_fadehint", "logs/orig_fadehint/models/model_latest.pt")]
for name, loss, ck in MODELS:
    cfg, ds, ag = load(loss)
    ag.load(ck, load_optimizer=False); ag.eval(); enc = ag.encoder_ema
    for pname, ws in states.items():
        prs, rats, s1s, cprs, habls = [], [], [], [], []
        for w in ws:
            x = torch.tensor(no.normalize(w[None]), device=dev, dtype=torch.float32).reshape(1, -1)
            def f(inp): return enc({"state": inp.reshape(1, 2, 53)}, None).reshape(-1)
            J = torch.autograd.functional.jacobian(f, x, vectorize=True).squeeze(1).detach().cpu().numpy()
            sv = np.linalg.svd(J, compute_uv=False); s2 = sv ** 2
            prs.append(float((s2.sum() ** 2) / ((s2 ** 2).sum() + 1e-12)))
            rats.append(float(sv[0] / (np.median(sv[sv > 1e-9]) + 1e-12)))
            s1s.append(float(sv[0]))
            g = np.linalg.norm(J, axis=0) ** 2
            cprs.append(float((g.sum() ** 2) / ((g ** 2).sum() + 1e-12)))
            Ja = J.copy(); Ja[:, HDIMS] = 0
            habls.append(float(np.linalg.svd(Ja, compute_uv=False)[0] / (sv[0] + 1e-12)))
        print(f"JACPHASE {name} {pname:8s}: PR p50={np.median(prs):.1f}  smax/smed={np.median(rats):.1f}  s1={np.median(s1s):.2f}  colPR={np.median(cprs):.1f}  -HEIGHT={np.median(habls):.2f}", flush=True)
print("JACPHASE-DONE")
