"""Fit-branch health of the stressreg arms: teacher-forced regression loss
mean ||f(0,0,phi(s)) - a||^2 on (a) the 2000 settle-window chunks (same set as
probe_collapsesens; MSE reference Lfull=0.02252) and (b) 2000 random all-phase chunks.
Also the stress residual itself on in-batch random pairs (how well the isometry
constraint is satisfied at the endpoint)."""
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

h = h5py.File("data/tool_hang_full2ins_2000.hdf5", "r")
keys = sorted(h["data"].keys(), key=lambda k: int(k.split("_")[-1]))
rng = np.random.RandomState(5)
WS, AC, WR, AR = [], [], [], []
for k in keys[:400]:
    o = h[f"data/{k}/obs"]
    ov = np.concatenate([np.asarray(o[q]) for q in OK], axis=1).astype(np.float32)
    a = np.clip(np.asarray(h[f"data/{k}/actions"]), -1, 1); g = a[:, 6]
    cl = [t for t in range(1, len(g)) if g[t-1] < 0 and g[t] >= 0]
    if not cl: continue
    c1 = cl[0]; T = len(g)
    for t in range(max(1, c1 - 12), min(c1 + 5, T - 16)):
        if len(WS) < 2000:
            WS.append(np.stack([ov[t-1], ov[t]])); AC.append(a[t:t+16])
    for _ in range(6):
        t = rng.randint(1, T - 16)
        if len(WR) < 2000:
            WR.append(np.stack([ov[t-1], ov[t]])); AR.append(a[t:t+16])
h.close()
def to10(A):
    A = np.stack(A)
    rc = ds.rotation_transformer.forward(A[:, :, 3:6].reshape(-1, 3)).reshape(len(A), 16, 6)
    return na.normalize(np.concatenate([A[:, :, :3], rc, A[:, :, 6:7]], axis=2).astype(np.float32))
WS = np.stack(WS); WR = np.stack(WR); A10S = to10(AC); A10R = to10(AR)
print(f"settle={len(WS)} rand={len(WR)}", flush=True)

MODELS = [("MSE", "regression", "logs/full_regression_2000/models/model_latest.pt"),
          ("stress03", "regression_stressreg", "logs/orig_stress03/models/model_latest.pt"),
          ("stress10", "regression_stressreg", "logs/orig_stress10/models/model_latest.pt"),
          ("fadehint", "regression_fadehint", "logs/orig_fadehint/models/model_latest.pt")]
for name, loss, ck in MODELS:
    cfg, ds, ag = load(loss)
    ag.load(ck, load_optimizer=False); ag.eval()
    enc = ag.encoder_ema; fm = ag.flow_map_ema if hasattr(ag, "flow_map_ema") else ag.flow_map
    out = {}
    for tag, Wq, Aq in [("settle", WS, A10S), ("rand", WR, A10R)]:
        tot, n = 0.0, 0
        stress_res = []
        for i in range(0, len(Wq), 256):
            x = torch.tensor(no.normalize(Wq[i:i+256]), device=dev, dtype=torch.float32)
            a = torch.tensor(np.asarray(Aq[i:i+256]), device=dev, dtype=torch.float32)
            with torch.no_grad():
                e = enc({"state": x}, None)
                y = fm.get_velocity(torch.zeros(len(a), device=dev), torch.zeros_like(a), e)
                tot += ((y - a) ** 2).mean().item() * len(a); n += len(a)
                B = len(a)
                ef = e.reshape(B, -1); ef = ef / (ef.norm(dim=1, keepdim=True) + 1e-9)
                xf = x.reshape(B, -1)
                idx = torch.randperm(B, device=dev)
                de = 1.0 - (ef * ef[idx]).sum(1); dx = (xf - xf[idx]).norm(dim=1)
                de_z = (de - de.mean()) / (de.std() + 1e-6); dx_z = (dx - dx.mean()) / (dx.std() + 1e-6)
                stress_res.append(((de_z - dx_z) ** 2).mean().item())
        out[tag] = (tot / n, float(np.mean(stress_res)))
    print(f"STRESSFIT {name}: fit settle={out['settle'][0]:.5f} rand={out['rand'][0]:.5f} | stress-residual settle={out['settle'][1]:.3f} rand={out['rand'][1]:.3f}", flush=True)
print("STRESSFIT-DONE")
