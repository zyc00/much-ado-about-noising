"""Is the embedding distribution 'more Gaussian' under MIP (user's epsilon-prediction
hypothesis)? Epps-Pulley statistic (vs N(0,1), whitened projections) + excess kurtosis of
random 1-D projections of the embedding, scripted + human pairs."""
import os, sys
os.environ["MUJOCO_GL"] = "egl"
sys.path.insert(0, "scripts"); sys.path.insert(0, ".")
import numpy as np, h5py, torch
from hydra import initialize_config_dir, compose
from omegaconf import OmegaConf
from mip.datasets.robomimic_dataset import make_dataset
from mip.agent import TrainingAgent

HUM = "/home/jigu/.cache/huggingface/hub/datasets--ChaoyiPan--mip-dataset/blobs/c5cb01b119a109a3d4842ca515906e86f1f35a8ee1a333d22b3503c1a513a8f4"
SCR = os.path.abspath("data/tool_hang_full2ins_2000.hdf5")
OK = ["object", "robot0_eef_pos", "robot0_eef_quat", "robot0_gripper_qpos"]
OLD = "/home/jigu/projects/much-ado-about-noising-old/checkpoints"

def load(loss, dsp):
    with initialize_config_dir(version_base=None, config_dir=os.path.abspath("examples/configs")):
        cfg = compose(config_name="main", overrides=["task=tool_hang_ph_state_delta_legacy",
            "+task.dataset_path=" + dsp, "network=chiunet",
            f"optimization.loss_type={loss}", "optimization.auto_resume=false", "log.wandb_mode=disabled"])
    OmegaConf.set_struct(cfg, False); cfg.task.obs_dim = 53
    ds = make_dataset(cfg.task)
    return cfg, ds, TrainingAgent(cfg)

def bank(dsp, n_demos):
    h = h5py.File(dsp, "r")
    keys = sorted(h["data"].keys(), key=lambda k: int(k.split("_")[-1]))
    W = []
    for k in keys[:n_demos]:
        o = h[f"data/{k}/obs"]
        ov = np.concatenate([np.asarray(o[q]) for q in OK], axis=1).astype(np.float32)
        for t in range(1, len(ov) - 1, 3):
            W.append(np.stack([ov[t-1], ov[t]]))
    h.close()
    return np.stack(W)

def gaussianity(E, seed=0):
    # whiten per-dim, project on K random directions, EP statistic + excess kurtosis
    X = E - E.mean(0, keepdims=True)
    X = X / (X.std(0, keepdims=True) + 1e-9)
    g = np.random.RandomState(seed)
    U = g.randn(X.shape[1], 128); U /= np.linalg.norm(U, axis=0, keepdims=True)
    P = X @ U
    P = (P - P.mean(0, keepdims=True)) / (P.std(0, keepdims=True) + 1e-9)
    tj = np.linspace(0.5, 4.0, 8); w = np.exp(-tj**2 / 4)
    tx = P[:, :, None] * tj[None, None, :]
    c = np.cos(tx).mean(0); s = np.sin(tx).mean(0)
    tgt = np.exp(-tj**2 / 2)
    ep = (((c - tgt)**2 + s**2) * w).sum(-1).mean() / w.sum()
    kurt = float(np.mean((P**4).mean(0) - 3))
    return float(ep), kurt

JOBS = [("scripted", SCR, [("MSE", "regression", "logs/full_regression_2000/models/model_latest.pt"),
                           ("MIP", "mip", "logs/full_mip_2000_s1/models/model_latest.pt")], 100),
        ("human", HUM, [("hMSE_s5", "regression", f"{OLD}/tool_hang_ph_state_regression_chiunet_256_seed5_success52.pt"),
                        ("hMIP_s5", "mip", f"{OLD}/tool_hang_ph_state_mip_chiunet_256_seed5_success82.pt"),
                        ("hMSE_s5001", "regression", f"{OLD}/tool_hang_ph_state_regression_chiunet_256_seed5001_success60.pt"),
                        ("hMIP_s5001", "mip", f"{OLD}/tool_hang_ph_state_mip_chiunet_256_seed5001_success80.pt")], 80)]
for dom, dsp, models, nd in JOBS:
    W = bank(dsp, nd)
    for name, loss, ck in models:
        if not os.path.exists(ck): print(f"GAUSS {dom} {name}: missing"); continue
        cfg, ds, ag = load(loss, dsp)
        no = ds.normalizer["obs"]["state"]; dev = cfg.optimization.device
        ag.load(ck, load_optimizer=False); ag.eval()
        enc = ag.encoder_ema
        outs = []
        for i in range(0, len(W), 512):
            x = torch.tensor(no.normalize(W[i:i+512]), device=dev, dtype=torch.float32)
            with torch.no_grad():
                e = enc({"state": x}, None)
            outs.append(e.reshape(len(x), -1).cpu().numpy())
        E = np.concatenate(outs)
        ep, ku = gaussianity(E)
        print(f"GAUSS {dom} {name}: EP={ep:.4f} excess-kurtosis={ku:+.2f} (n={len(E)})", flush=True)
print("GAUSS-DONE")
