"""Adjudicate the smoothness challenge: (1) seed-pair replication + per-demo SE for the
S1/S2 metrics; (2) OFF-GROOVE field coherence — the deployment-relevant discriminator.

Off-groove design: for each demo window (align1, carry1), shift the whole state sequence by
a fixed random lateral eef offset delta (kinematically consistent; 2/5/10mm) and compare the
policy sequence on the shifted groove vs the original:
  coh  = corr( hf(pi(s+delta)), hf(pi(s)) )   — does the fine field stay organized off-groove?
  gtrk = corr( hf(pi(s+delta)), hf(label) )   — does it still track the demonstrator feedback?
If MSE's verbatim feedback field is memorized on the groove, its hf component should
decohere faster with offset than MIP's.
Models: both historical seed pairs (s5 and s5001).
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
EEF_POS = [44, 45, 46]; EEF_QUAT = [47, 48, 49, 50]
REL = {"b": [0, 1, 2], "f": [14, 15, 16], "t": [28, 29, 30]}
def quat2R(q):
    x, y, z, w = q
    return np.array([[1-2*(y*y+z*z),2*(x*y-z*w),2*(x*z+y*w)],[2*(x*y+z*w),1-2*(x*x+z*z),2*(y*z-x*w)],[2*(x*z-y*w),2*(y*z+x*w),1-2*(x*x+y*y)]])

def load(loss):
    with initialize_config_dir(version_base=None, config_dir=os.path.abspath("examples/configs")):
        cfg = compose(config_name="main", overrides=["task=tool_hang_ph_state_delta_legacy",
            "+task.dataset_path=" + DSP, "network=chiunet",
            f"optimization.loss_type={loss}", "optimization.auto_resume=false", "log.wandb_mode=disabled"])
    OmegaConf.set_struct(cfg, False); cfg.task.obs_dim = 53
    ds = make_dataset(cfg.task)
    return cfg, ds, TrainingAgent(cfg)

cfg, ds, _ = load("regression")
no = ds.normalizer["obs"]["state"]; na = ds.normalizer["action"]; dev = cfg.optimization.device
START = cfg.task.obs_steps - 1

h = h5py.File(DSP, "r")
keys = sorted(h["data"].keys(), key=lambda k: int(k.split("_")[-1]))
WINS = []   # (window name, W (T,2,53), a10 (T,10))
for di, k in enumerate(keys[:40]):
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
    for wn, lo, hi in [("align1", r1 - 40, r1), ("carry1", c1 + 10, min(c1 + 60, r1 - 40))]:
        if hi - lo < 25: continue
        Wd = np.stack([np.stack([ov[t-1], ov[t]]) for t in range(lo, hi)])
        WINS.append((wn, Wd, a10[lo:hi]))
h.close()
print(f"windows={len(WINS)}", flush=True)

def smooth_ma(x, w=9):
    k = np.ones(w) / w
    return np.stack([np.convolve(x[:, d], k, mode="same") for d in range(x.shape[1])], 1)
def hfc(x, y):
    num = (x * y).sum(); den = np.sqrt((x**2).sum() * (y**2).sum()) + 1e-12
    return float(num / den)

MODELS = [("hMSE_s5", "regression", f"{OLD}/tool_hang_ph_state_regression_chiunet_256_seed5_success52.pt"),
          ("hMIP_s5", "mip", f"{OLD}/tool_hang_ph_state_mip_chiunet_256_seed5_success82.pt"),
          ("hMSE_s5001", "regression", f"{OLD}/tool_hang_ph_state_regression_chiunet_256_seed5001_success60.pt"),
          ("hMIP_s5001", "mip", f"{OLD}/tool_hang_ph_state_mip_chiunet_256_seed5001_success80.pt")]
rng = np.random.RandomState(0)
for name, loss, ck in MODELS:
    cfg, ds, ag = load(loss)
    ag.load(ck, load_optimizer=False); ag.eval()
    enc = ag.encoder_ema; fm = ag.flow_map_ema if hasattr(ag, "flow_map_ema") else ag.flow_map
    def fwd(X):
        preds = []
        for i in range(0, len(X), 512):
            x = torch.tensor(no.normalize(X[i:i+512]), device=dev, dtype=torch.float32)
            with torch.no_grad():
                e = enc({"state": x}, None)
                y = fm.get_velocity(torch.zeros(len(x), device=dev), torch.zeros(len(x), 16, 10, device=dev), e)
            preds.append(y[:, START].cpu().numpy())
        return np.concatenate(preds)
    # per-demo S1/S2 with SE (on-groove)
    stats = {("align1", "tv"): [], ("align1", "hf"): [], ("carry1", "tv"): [], ("carry1", "hf"): []}
    offres = {}
    for wn, Wd, a10 in WINS:
        p0 = fwd(Wd)
        hf_l = a10 - smooth_ma(a10); hf_p0 = p0 - smooth_ma(p0)
        stats[(wn, "tv")].append(np.linalg.norm(np.diff(p0, axis=0), axis=1).mean() /
                                 (np.linalg.norm(np.diff(a10, axis=0), axis=1).mean() + 1e-9))
        stats[(wn, "hf")].append(hfc(hf_p0, hf_l))
        for mag in [0.002, 0.005, 0.010]:
            d = rng.randn(3); d /= np.linalg.norm(d)
            Ws = Wd.copy()
            for fr in range(2):
                for t in range(len(Ws)):
                    R = quat2R(Ws[t, fr, EEF_QUAT])
                    Ws[t, fr, EEF_POS] += d * mag
                    for rel in REL.values(): Ws[t, fr, rel] += -R.T @ (d * mag)
            ps = fwd(Ws)
            hf_ps = ps - smooth_ma(ps)
            key = (wn, mag)
            offres.setdefault(key, {"coh": [], "gtrk": []})
            offres[key]["coh"].append(hfc(hf_ps, hf_p0))
            offres[key]["gtrk"].append(hfc(hf_ps, hf_l))
    for wn in ["align1", "carry1"]:
        tv = np.array(stats[(wn, "tv")]); hf = np.array(stats[(wn, "hf")])
        print(f"ONGROOVE {name} {wn}: TVratio={tv.mean():.3f}+-{tv.std()/np.sqrt(len(tv)):.3f} hfcorr={hf.mean():.3f}+-{hf.std()/np.sqrt(len(hf)):.3f} (n={len(tv)})", flush=True)
    for wn in ["align1", "carry1"]:
        line = f"OFFGROOVE {name} {wn}:"
        for mag in [0.002, 0.005, 0.010]:
            r = offres[(wn, mag)]
            line += f" | {int(mag*1000)}mm coh={np.mean(r['coh']):.3f} gtrk={np.mean(r['gtrk']):.3f}"
        print(line, flush=True)
print("OFFGROOVE-DONE")
