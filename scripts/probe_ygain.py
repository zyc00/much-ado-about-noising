"""y-channel gain: is chiunet-MIP doing residual/skip-copying of the action input while
mlp-MIP does 'real x-prediction'? At view-2-like inputs (t=0.9, y=a+sigma*eps), perturb y
by small delta and measure gain = ||F(y+delta)-F(y)|| / ||delta|| and the same-slot cosine
cos(dF, delta) (identity copying -> gain~1, cos~1; pure x-prediction -> gain~0).
Also MSE arms at act_0=0 (deployment condition) for completeness.
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

def load(loss, network):
    with initialize_config_dir(version_base=None, config_dir=os.path.abspath("examples/configs")):
        cfg = compose(config_name="main", overrides=["task=tool_hang_ph_state_delta_legacy",
            "+task.dataset_path=" + DSP, f"network={network}",
            f"optimization.loss_type={loss}", "optimization.auto_resume=false", "log.wandb_mode=disabled"])
    OmegaConf.set_struct(cfg, False); cfg.task.obs_dim = 53
    ds = make_dataset(cfg.task)
    return cfg, ds, TrainingAgent(cfg)

cfg, ds, _ = load("regression", "chiunet")
no = ds.normalizer["obs"]["state"]; na = ds.normalizer["action"]; dev = cfg.optimization.device

h = h5py.File(DSP, "r")
keys = sorted(h["data"].keys(), key=lambda k: int(k.split("_")[-1]))
W, A10 = [], []
for k in keys[:30]:
    o = h[f"data/{k}/obs"]
    ov = np.concatenate([np.asarray(o[q]) for q in OK], axis=1).astype(np.float32)
    a = np.clip(np.asarray(h[f"data/{k}/actions"]), -1, 1); T = len(a)
    rc = ds.rotation_transformer.forward(a[:, 3:6])
    a10 = na.normalize(np.concatenate([a[:, :3], rc, a[:, 6:7]], axis=1).astype(np.float32))
    for t in range(30, T - 20, 40):
        W.append(np.stack([ov[t-1], ov[t]])); A10.append(a10[t:t+16])
h.close()
W = np.stack(W); A10 = np.stack(A10)
print(f"states={len(W)}", flush=True)

MODELS = [("chiunet-MIP-s5", "mip", "chiunet", f"{OLD}/tool_hang_ph_state_mip_chiunet_256_seed5_success82.pt"),
          ("mlp-MIP-s5001", "mip", "mlp", f"{OLD}/tool_hang_ph_state_mip_mlp_512_seed5001_success85.pt"),
          ("chiunet-MSE-s5", "regression", "chiunet", f"{OLD}/tool_hang_ph_state_regression_chiunet_256_seed5_success52.pt"),
          ("mlp-MSE-s1", "regression", "mlp", f"{OLD}/tool_hang_ph_state_regression_mlp_512_seed1_success90.pt")]
for name, loss, network, ck in MODELS:
    cfg, ds, ag = load(loss, network)
    ag.load(ck, load_optimizer=False); ag.eval()
    enc = ag.encoder_ema; fm = ag.flow_map_ema if hasattr(ag, "flow_map_ema") else ag.flow_map
    tts = cfg.optimization.t_two_step
    CH = 10 if network == "mlp" else 16
    Ac = torch.tensor(A10[:, :CH], device=dev, dtype=torch.float32)
    x = torch.tensor(no.normalize(W), device=dev, dtype=torch.float32)
    with torch.no_grad():
        e = enc({"state": x}, None)
    torch.manual_seed(0)
    for mode in (["view2"] if loss == "mip" else ["dep0"]):
        if mode == "view2":
            y = Ac + 0.1 * torch.randn_like(Ac)
            t = torch.full((len(x),), tts, device=dev)
        else:
            y = torch.zeros_like(Ac)
            t = torch.zeros(len(x), device=dev)
        delta = 0.05 * torch.randn_like(y)
        with torch.no_grad():
            f0 = fm.get_velocity(t, y, e)
            f1 = fm.get_velocity(t, y + delta, e)
        df = (f1 - f0).reshape(len(x), -1); dl = delta.reshape(len(x), -1)
        gain = (df.norm(dim=1) / (dl.norm(dim=1) + 1e-9)).cpu().numpy()
        cosv = ((df * dl).sum(1) / (df.norm(dim=1) * dl.norm(dim=1) + 1e-9)).cpu().numpy()
        print(f"YGAIN {name} {mode}: gain p50={np.median(gain):.3f} p90={np.quantile(gain,0.9):.3f} | cos(dF,delta) p50={np.median(cosv):+.3f}", flush=True)
print("YGAIN-DONE")
