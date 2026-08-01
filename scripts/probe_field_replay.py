"""Counterfactual field replay at hrotaux-FAIL near-gate orbit states:
evaluate hrotaux_s5, hMIP_s5, hMSE_s5 at the SAME dumped states; predicted chunk's net xy
command projected onto the inward (centering) direction. cos_inward distribution per model
answers whether MIP's field is radial-inward where the repaired policy orbits.
Control: same metric at hMIP-PASS near states (its own support)."""
import os
os.environ["MUJOCO_GL"] = "egl"
import numpy as np, torch, sys
sys.path.insert(0, "scripts"); sys.path.insert(0, ".")
from hydra import initialize_config_dir, compose
from omegaconf import OmegaConf
from mip.datasets.robomimic_dataset import make_dataset
from mip.agent import TrainingAgent
DSP = "/home/jigu/.cache/huggingface/hub/datasets--ChaoyiPan--mip-dataset/blobs/c5cb01b119a109a3d4842ca515906e86f1f35a8ee1a333d22b3503c1a513a8f4"
OLD = "/home/jigu/projects/much-ado-about-noising-old/checkpoints"
BP = slice(7, 10); FP = slice(21, 24)
import h5py
h = h5py.File(DSP, "r")
keys = sorted(h["data"].keys(), key=lambda k: int(k.split("_")[-1]))
offs = []
for k in keys[:80]:
    o = h[f"data/{k}/obs"]
    a = np.clip(np.asarray(h[f"data/{k}/actions"]), -1, 1); g = a[:, 6]; T = len(a)
    cl = [t for t in range(1, T) if g[t-1] < 0 and g[t] >= 0]
    op = [t for t in range(1, T) if g[t-1] >= 0 and g[t] < 0]
    if not cl: continue
    c1 = cl[0]; r1 = next((t for t in op if t > c1), None)
    if not r1: continue
    ov = np.concatenate([np.asarray(o[q]) for q in ["object", "robot0_eef_pos", "robot0_eef_quat", "robot0_gripper_qpos"]], axis=1).astype(np.float32)
    offs.append(ov[r1-1, FP] - ov[r1-1, BP])
h.close()
off = np.median(np.stack(offs), 0)
def held_range(o, a):
    L = min(len(o), len(a)); gc = a[:L, 6]
    run = 0; g0 = None; gend = L
    for t in range(L):
        run = run + 1 if gc[t] >= 0 else 0
        if run >= 15 and g0 is None: g0 = t - 14
        if g0 is not None and gc[t] < 0: gend = t; break
    return g0, gend
def collect_states(name, want_pass):
    z = np.load(f"analysis/traj_vis/human_{name}.npz")
    i = 0; S = []
    while f"ep{i}_obs" in z.files:
        o = z[f"ep{i}_obs"]; a = z[f"ep{i}_act"]; m = z[f"ep{i}_meta"]
        if bool(int(m[1])) == want_pass:
            g0, gend = held_range(o, a)
            if g0 is not None:
                v = o[g0:gend, FP] - (o[g0:gend, BP] + off)
                lat = np.linalg.norm(v[:, :2], axis=1); alt = v[:, 2]
                for t in range(1, len(v), 4):
                    if 0.010 <= lat[t] < 0.060 and 0.005 <= alt[t] < 0.120:
                        S.append((o[g0 + t - 1], o[g0 + t], -v[t, :2] / (lat[t] + 1e-9)))
        i += 1
    return S
SETS = {"hrotauxFAIL-states": collect_states("hrotaux_s5", False),
        "hMIPPASS-states": collect_states("hMIP_s5", True)}
print({k: len(v) for k, v in SETS.items()}, flush=True)
def load(loss, extra=()):
    with initialize_config_dir(version_base=None, config_dir=os.path.abspath("examples/configs")):
        cfg = compose(config_name="main", overrides=["task=tool_hang_ph_state_delta_legacy",
            "+task.dataset_path=" + DSP, "network=chiunet",
            f"optimization.loss_type={loss}", "optimization.auto_resume=false", "log.wandb_mode=disabled"] + list(extra))
    OmegaConf.set_struct(cfg, False); cfg.task.obs_dim = 53
    ds = make_dataset(cfg.task)
    return cfg, ds, TrainingAgent(cfg)
MODELS = [("hrotaux", "regression", "logs/hrotaux_s5/models/model_latest.pt", ("+task.rot_indicator=true", "task.act_dim=13")),
          ("hMIP", "mip", f"{OLD}/tool_hang_ph_state_mip_chiunet_256_seed5_success82.pt", ()),
          ("hMSE", "regression", f"{OLD}/tool_hang_ph_state_regression_chiunet_256_seed5_success52.pt", ())]
for mname, loss, ck, extra in MODELS:
    cfg, ds, ag = load(loss, extra)
    AD = int(cfg.task.act_dim)
    ag.load(ck, load_optimizer=False); ag.eval()
    dev = cfg.optimization.device; AS = cfg.task.act_steps; start = cfg.task.obs_steps - 1
    no = ds.normalizer["obs"]["state"]; na = ds.normalizer["action"]
    torch.manual_seed(0)
    for sname, S in SETS.items():
        cosi, mags = [], []
        B = 64
        for b in range(0, len(S), B):
            batch = S[b:b + B]
            w = np.stack([np.stack([p, c]) for p, c, u in batch])
            ot = {"state": torch.tensor(no.normalize(w), device=dev, dtype=torch.float32)}
            with torch.no_grad():
                an = ag.sample(act_0=torch.randn((len(batch), 16, AD), device=dev), obs=ot, use_ema=True)
            acts = ds.undo_transform_action(na.unnormalize(an.detach().cpu().numpy())[:, start:start + AS])
            for j, (p, c, u) in enumerate(batch):
                net = acts[j][:, 0:2].sum(0)
                n = np.linalg.norm(net)
                if n > 1e-6:
                    cosi.append(float(np.dot(net, u)) / n); mags.append(n)
        cosi = np.array(cosi)
        print(f"FIELD {mname} @ {sname}: cos_inward p50={np.median(cosi):+.2f} frac>0.3={np.mean(cosi>0.3):.2f} frac<-0.3={np.mean(cosi<-0.3):.2f} |net| p50={np.median(mags):.3f} (n={len(cosi)})", flush=True)
print("FIELD-DONE")
