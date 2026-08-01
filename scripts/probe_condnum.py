"""Condition numbers of the policy Jacobian (da/ds, 10x106, autograd through the
differentiable deployment path) and the encoder Jacobian (dphi/ds) at settle-canon,
settle-deployed, and pooled full-trajectory states. kappa10 = s1/s10; ratio = s1/s_med."""
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
no = ds.normalizer["obs"]["state"]; dev = cfg.optimization.device; start = cfg.task.obs_steps - 1
Q = np.load("scripts/jac_queries.npz")

# pooled full-trajectory states (20 per phase, as in probe_jacphase)
h5f = h5py.File("data/tool_hang_full2ins_2000.hdf5", "r")
rng = np.random.RandomState(7)
full_states = []
for di in rng.choice(200, 20, replace=False):
    d = h5f[f"data/demo_{di}"]
    a = np.clip(np.asarray(d["actions"]), -1, 1); g = a[:, 6]
    cl = [t for t in range(1, len(g)) if g[t-1] < 0 and g[t] >= 0]
    if not cl: continue
    c1 = cl[0]; T = len(g)
    ov = np.concatenate([np.asarray(d["obs"][q]) for q in OK], axis=1).astype(np.float32)
    for lo, hi in [(c1-40, c1-15), (c1+20, c1+60), (c1+72, T-2)]:
        t = rng.randint(max(1, lo), min(hi, T-1))
        full_states.append(np.stack([ov[t-1], ov[t]]))
h5f.close()
SETS = {"settle_canon": Q["canon"], "settle_dep": Q["dep"], "full_traj": np.stack(full_states)}

def kappas(J, k=10):
    sv = np.linalg.svd(J, compute_uv=False)
    sv = sv[sv > 1e-9]
    k = min(k, len(sv))
    return float(sv[0] / sv[k-1]), float(sv[0] / np.median(sv))

MODELS = [("MSE", "regression", "logs/full_regression_2000/models/model_latest.pt"),
          ("MIPs1", "mip", "logs/full_mip_2000_s1/models/model_latest.pt"),
          ("fadehint", "regression_fadehint", "logs/orig_fadehint/models/model_latest.pt")]
for name, loss, ck in MODELS:
    cfg, ds, ag = load(loss)
    ag.load(ck, load_optimizer=False); ag.eval()
    enc = ag.encoder_ema; fm = ag.flow_map_ema if hasattr(ag, "flow_map_ema") else ag.flow_map
    tts = cfg.optimization.t_two_step
    def pol(inp):   # differentiable deployment path -> first executed normalized action (10,)
        e = enc({"state": inp.reshape(1, 2, 53)}, None)
        z = torch.zeros(1, 16, 10, device=inp.device)
        t0 = torch.zeros(1, device=inp.device)
        y0 = fm.get_velocity(t0, z, e)
        if loss == "mip":
            t1 = torch.full((1,), tts, device=inp.device)
            y0 = fm.get_velocity(t1, y0, e)
        return y0[0, start]
    def fea(inp):
        return enc({"state": inp.reshape(1, 2, 53)}, None).reshape(-1)
    for sname, S in SETS.items():
        kp10, kpm, ke10, kem = [], [], [], []
        for w in S:
            x = torch.tensor(no.normalize(w[None]), device=dev, dtype=torch.float32).reshape(1, -1)
            Jp = torch.autograd.functional.jacobian(pol, x, vectorize=True).squeeze(1).detach().cpu().numpy()
            Je = torch.autograd.functional.jacobian(fea, x, vectorize=True).squeeze(1).detach().cpu().numpy()
            a, b = kappas(Jp, 10); kp10.append(a); kpm.append(b)
            a, b = kappas(Je, 10); ke10.append(a); kem.append(b)
        print(f"COND {name} {sname:12s}: policy k10={np.median(kp10):8.1f} s1/smed={np.median(kpm):6.1f} | encoder k10={np.median(ke10):8.1f} s1/smed={np.median(kem):6.1f}", flush=True)
print("COND-DONE")
