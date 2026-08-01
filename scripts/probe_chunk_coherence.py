"""Why is chiunet-MSE the worst cell? Architectural hypothesis: chiunet's temporal convs can
express PER-STEP fine structure inside the 16-step chunk (can reproduce tremor within a
chunk); the MLP emits the chunk from one hidden state — chunk-coherent by construction.

Instruments over the 2x2 factorial (chiunet/mlp x MSE/MIP, historical checkpoints):
 W1 WITHIN-CHUNK TV: mean ||chunk[k+1] - chunk[k]|| over the executed slots (1..9), at
    align1 demo states, normalized by median |chunk| — the executed actions come from ONE
    chunk, so within-chunk jitter IS executed jitter.
 W2 chunk hf-match: corr( hf-across-chunk(pred), hf(label chunk) ) — does the model
    reproduce the demonstrator's within-chunk tremor?
 W3 fit: first-slot residual RMS at align1 (fit context for the cell).
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
START = cfg.task.obs_steps - 1; AS = cfg.task.act_steps

# align1 states + label chunks
h = h5py.File(DSP, "r")
keys = sorted(h["data"].keys(), key=lambda k: int(k.split("_")[-1]))
W, LC = [], []
for k in keys[:80]:
    o = h[f"data/{k}/obs"]
    ov = np.concatenate([np.asarray(o[q]) for q in OK], axis=1).astype(np.float32)
    a = np.clip(np.asarray(h[f"data/{k}/actions"]), -1, 1); g = a[:, 6]; T = len(a)
    cl = [t for t in range(1, T) if g[t-1] < 0 and g[t] >= 0]
    op = [t for t in range(1, T) if g[t-1] >= 0 and g[t] < 0]
    if not cl: continue
    c1 = cl[0]
    r1 = next((t for t in op if t > c1), None)
    if not r1: continue
    rc = ds.rotation_transformer.forward(a[:, 3:6])
    a10 = na.normalize(np.concatenate([a[:, :3], rc, a[:, 6:7]], axis=1).astype(np.float32))
    for t in range(r1 - 35, r1 - 2, 4):
        if t < 1 or t + 16 > T: continue
        W.append(np.stack([ov[t-1], ov[t]])); LC.append(a10[t:t+16])
h.close()
W = np.stack(W); LC = np.stack(LC)
print(f"align1 states={len(W)}", flush=True)

MODELS = [("chiunet-MSE", "regression", "chiunet", f"{OLD}/tool_hang_ph_state_regression_chiunet_256_seed5_success52.pt"),
          ("chiunet-MIP", "mip", "chiunet", f"{OLD}/tool_hang_ph_state_mip_chiunet_256_seed5_success82.pt"),
          ("mlp-MSE", "regression", "mlp", f"{OLD}/tool_hang_ph_state_regression_mlp_512_seed5001_success90.pt"),
          ("mlp-MIP", "mip", "mlp", f"{OLD}/tool_hang_ph_state_mip_mlp_512_seed5001_success85.pt"),
          ("chiunet-MSE-s5001", "regression", "chiunet", f"{OLD}/tool_hang_ph_state_regression_chiunet_256_seed5001_success60.pt"),
          ("chiunet-MIP-s5001", "mip", "chiunet", f"{OLD}/tool_hang_ph_state_mip_chiunet_256_seed5001_success80.pt"),
          ("mlp-MSE-s1", "regression", "mlp", f"{OLD}/tool_hang_ph_state_regression_mlp_512_seed1_success90.pt")]
for name, loss, network, ck in MODELS:
    if not os.path.exists(ck):
        print(f"CHUNK {name}: MISSING", flush=True); continue
    cfg, ds, ag = load(loss, network)
    ag.load(ck, load_optimizer=False); ag.eval()
    enc = ag.encoder_ema; fm = ag.flow_map_ema if hasattr(ag, "flow_map_ema") else ag.flow_map
    CH = 10 if network == "mlp" else 16
    chunks = []
    for i in range(0, len(W), 256):
        x = torch.tensor(no.normalize(W[i:i+256]), device=dev, dtype=torch.float32)
        with torch.no_grad():
            e = enc({"state": x}, None)
            y = fm.get_velocity(torch.zeros(len(x), device=dev), torch.zeros(len(x), CH, 10, device=dev), e)
        chunks.append(y.cpu().numpy())
    P = np.concatenate(chunks)          # (N, CH, 10)
    LCc = LC[:, :CH]
    ex = slice(START, START + AS)
    dpe = np.linalg.norm(np.diff(P[:, ex], axis=1), axis=2)         # within-chunk TV, executed slots
    dle = np.linalg.norm(np.diff(LCc[:, ex], axis=1), axis=2)
    mag = np.median(np.linalg.norm(P[:, ex], axis=2))
    wtv = float(np.median(dpe) / (mag + 1e-9))
    wtv_ratio = float(np.median(dpe) / (np.median(dle) + 1e-9))
    # W2: within-chunk hf correlation with the label chunk (per sample, executed slots)
    hp = P[:, ex] - P[:, ex].mean(1, keepdims=True)
    hl = LCc[:, ex] - LCc[:, ex].mean(1, keepdims=True)
    num = (hp * hl).sum((1, 2)); den = np.sqrt((hp**2).sum((1, 2)) * (hl**2).sum((1, 2))) + 1e-12
    w2 = float(np.median(num / den))
    r = np.linalg.norm(P[:, START] - LCc[:, START], axis=1)
    print(f"CHUNK {name}: within-chunk relTV={wtv:.3f} TV(pred)/TV(label)={wtv_ratio:.2f} | chunk-hf-corr={w2:.3f} | firstslot RMS={np.sqrt((r**2).mean()):.3f}", flush=True)
print("CHUNK-DONE")
