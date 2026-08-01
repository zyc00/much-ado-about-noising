"""Per-sample effective gradient weight vs tube distance d, for MSE vs MIP loss.
Output-space gradient-norm proxy (theta-grad ~ output-grad x const):
  MSE: 2*||f(0,0,o)-a||
  MIP: 2*||(f0-a)||/t^2 + 2*E_noise||(f1-a)||/(1-t)^2   (K noise draws)
Bin by tangent-orthogonalized tube distance d of the state."""
import os, sys
os.environ["MUJOCO_GL"] = "egl"
sys.path.insert(0, "scripts"); sys.path.insert(0, ".")
import numpy as np, torch, h5py
from hydra import initialize_config_dir, compose
from omegaconf import OmegaConf
from mip.datasets.robomimic_dataset import make_dataset
from mip.agent import TrainingAgent

OK = ["object", "robot0_eef_pos", "robot0_eef_quat", "robot0_gripper_qpos"]
POS = slice(44, 47)
DS = os.environ.get("DS", "data/tool_hang_full2ins_2000.hdf5")
PKNOTS = 120; KNOISE = 8

hf = h5py.File(DS, "r")
keys = sorted(hf["data"].keys(), key=lambda k: int(k.split("_")[-1]))
grid = np.linspace(0, 1, PKNOTS)
tube = []
for k in keys[:300]:
    p = np.asarray(hf[f"data/{k}/obs/robot0_eef_pos"]); t = np.linspace(0, 1, len(p))
    tube.append(np.stack([np.interp(grid, t, p[:, i]) for i in range(3)], 1))
tube = np.stack(tube); center = tube.mean(0); sigma = tube.std(0).mean(1) + 1e-9
tg = np.gradient(center, axis=0); tg /= (np.linalg.norm(tg, axis=1, keepdims=True) + 1e-12)

cfgdir = os.path.abspath("examples/configs")
results = {}
for spec in os.environ["MODELS"].replace("|", ":").split(","):
    tag, ckpt, loss = spec.split(":")[:3]
    with initialize_config_dir(version_base=None, config_dir=cfgdir):
        cfg = compose(config_name="main", overrides=["task=tool_hang_ph_state_delta_legacy",
            "+task.dataset_path=" + os.path.abspath(DS),
            "network=chiunet", f"optimization.loss_type={loss}",
            "optimization.auto_resume=false", "log.wandb_mode=disabled"])
    OmegaConf.set_struct(cfg, False); cfg.task.obs_dim = 53
    cfg.task.horizon = 16  # train script rounds horizon up to 2^n for chiunet
    ds = make_dataset(cfg.task)
    dev = cfg.optimization.device; obs_steps = cfg.task.obs_steps
    no = ds.normalizer["obs"]["state"]; na = ds.normalizer["action"]
    ag = TrainingAgent(cfg); ag.load(ckpt, load_optimizer=False); ag.eval()
    tts = cfg.optimization.t_two_step
    net = ag.flow_map
    enc = ag.encoder
    dvals, gvals = [], []
    from torch.utils.data import DataLoader
    dl = DataLoader(ds, batch_size=256, shuffle=True, num_workers=2)
    C = torch.tensor(center, device=dev, dtype=torch.float32)     # (P,3)
    SG = torch.tensor(sigma, device=dev, dtype=torch.float32)
    TG = torch.tensor(tg, device=dev, dtype=torch.float32)
    nb = 0
    for batch in dl:
        obs = batch["obs"]["state"].to(dev).float()[:, :obs_steps]
        a = batch["action"].to(dev).float()
        with torch.no_grad():
            raw = torch.tensor(no.unnormalize(obs.cpu().numpy()), device=dev, dtype=torch.float32)
            eef = raw[:, -1, 44:47]                              # (B,3) last frame
            diff = eef[:, None, :] - C[None]                     # (B,P,3)
            k_idx = torch.argmin(diff.norm(dim=2), dim=1)        # nearest knot
            dvb = eef - C[k_idx]
            tgb = TG[k_idx]
            dvp = dvb - (dvb * tgb).sum(1, keepdim=True) * tgb
            d_b = dvp.norm(dim=1) / SG[k_idx]
            emb = enc(obs, None)
            z = torch.zeros(obs.shape[0], device=dev)
            a0 = torch.zeros_like(a)
            f0 = net.get_velocity(z, a0, emb)
            r0 = (f0 - a).flatten(1).norm(dim=1)
            if loss == "mip":
                g_b = 2 * r0 / (tts ** 2)
                g1 = torch.zeros_like(r0)
                for _ in range(KNOISE):
                    noise = torch.randn_like(a)
                    at = a + (1 - tts) * noise
                    f1 = net.get_velocity(z + tts, at, emb)
                    g1 += 2 * (f1 - a).flatten(1).norm(dim=1) / ((1 - tts) ** 2)
                g_b = g_b + g1 / KNOISE
            else:
                g_b = 2 * r0
        dvals.extend(d_b.cpu().numpy().tolist()); gvals.extend(g_b.cpu().numpy().tolist())
        nb += 1
        if nb >= 40: break
    dvals, gvals = np.array(dvals), np.array(gvals)
    BANDS = [(0, 0.5), (0.5, 1), (1, 1.5), (1.5, 2.5), (2.5, 9)]
    interior = np.median(gvals[dvals < 1])
    print(f"GRADW {tag} n={len(dvals)} interior_p50={interior:.3f}")
    for lo, hi in BANDS:
        m = (dvals >= lo) & (dvals < hi)
        if m.sum() < 15: continue
        print(f"GRADW {tag} d=[{lo},{hi}) n={m.sum()} g_p50={np.median(gvals[m]):.3f} rel={np.median(gvals[m])/interior:.3f}")
print("GRADW-DONE")
