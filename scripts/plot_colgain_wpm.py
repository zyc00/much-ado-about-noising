"""Colgain comparison at wpmatch settle: wpm_MSE (SR36) vs wpm_lam3 (SR82) vs wpm_MIP (SR87)."""
import os, sys
os.environ["MUJOCO_GL"] = "egl"
sys.path.insert(0, "scripts"); sys.path.insert(0, ".")
import numpy as np, h5py, torch
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mp
from hydra import initialize_config_dir, compose
from omegaconf import OmegaConf
from mip.datasets.robomimic_dataset import make_dataset
from mip.agent import TrainingAgent

DSP = "data/tool_hang_full2ins_wpmatch_2000.hdf5"
OK = ["object", "robot0_eef_pos", "robot0_eef_quat", "robot0_gripper_qpos"]
def load(loss):
    with initialize_config_dir(version_base=None, config_dir=os.path.abspath("examples/configs")):
        cfg = compose(config_name="main", overrides=["task=tool_hang_ph_state_delta_legacy",
            "+task.dataset_path=" + os.path.abspath(DSP), "network=chiunet",
            f"optimization.loss_type={loss}", "optimization.auto_resume=false", "log.wandb_mode=disabled"])
    OmegaConf.set_struct(cfg, False); cfg.task.obs_dim = 53
    ds = make_dataset(cfg.task)
    return cfg, ds, TrainingAgent(cfg)

cfg, ds, _ = load("regression")
no = ds.normalizer["obs"]["state"]; dev = cfg.optimization.device
h = h5py.File(DSP, "r")
keys = sorted(h["data"].keys(), key=lambda k: int(k.split("_")[-1]))
W, phase = [], []
for k in keys[:150]:
    o = h[f"data/{k}/obs"]
    ov = np.concatenate([np.asarray(o[q]) for q in OK], axis=1).astype(np.float32)
    a = np.clip(np.asarray(h[f"data/{k}/actions"]), -1, 1); g = a[:, 6]
    cl = [t for t in range(1, len(g)) if g[t-1] < 0 and g[t] >= 0]
    if not cl: continue
    c1 = cl[0]
    for t in range(1, len(g), 2):
        if -12 <= t - c1 < 5:
            W.append(np.stack([ov[t-1], ov[t]]))
h.close()
W = np.stack(W)
NAMES = (["b_relx","b_rely","b_relz"] + [f"b_rq{i}" for i in range(4)] + ["b_px","b_py","b_pz"] +
         [f"b_q{i}" for i in range(4)] +
         ["f_relx","f_rely","f_relz"] + [f"f_rq{i}" for i in range(4)] + ["f_px","f_py","f_pz"] +
         [f"f_q{i}" for i in range(4)] +
         ["t_relx","t_rely","t_relz"] + [f"t_rq{i}" for i in range(4)] + ["t_px","t_py","t_pz"] +
         [f"t_q{i}" for i in range(4)] + ["flag0","flag1","eef_x","eef_y","eef_z"] +
         [f"eef_q{i}" for i in range(4)] + ["grip0","grip1"])
def color(i):
    if i < 14 or (28 <= i < 42) or i in (42, 43): return "#c0392b"   # static distractors + flags
    if 21 <= i < 28: return "#e67e22"                                # frame world pose
    return "#27ae60"                                                  # hand-centric

MODELS = [("wpm_MSE (SR 36)", "regression", "logs/wpmatch_mse/models/model_latest.pt"),
          ("wpm_lam3 (SR 82)", "mip_lambda", "logs/wpm_lam3/models/model_latest.pt"),
          ("wpm_MIP (SR 87)", "mip", "logs/wpmatch_mip/models/model_latest.pt")]
rng = np.random.RandomState(0)
canon = W[rng.choice(len(W), 30, replace=False)]
fig, axes = plt.subplots(3, 1, figsize=(15, 9), sharex=True)
for row, (mname, loss, ck) in enumerate(MODELS):
    cfg, ds, ag = load(loss)
    ag.load(ck, load_optimizer=False); ag.eval()
    enc = ag.encoder_ema
    def fea(inp):
        return enc({"state": inp.reshape(1, 2, 53)}, None).reshape(-1)
    gains = np.zeros(53)
    for w in canon:
        x = torch.tensor(no.normalize(w[None]), device=dev, dtype=torch.float32).reshape(1, -1)
        J = torch.autograd.functional.jacobian(fea, x, vectorize=True).squeeze(1).detach().cpu().numpy()
        g106 = np.linalg.norm(J, axis=0)
        gains += g106[:53] + g106[53:]
    gains /= len(canon)
    red = sum(gains[i] for i in range(53) if color(i) == "#c0392b") / gains.sum()
    ax = axes[row]
    ax.bar(range(53), gains, color=[color(i) for i in range(53)])
    ax.set_title(f"{mname} — settle window  (static-distractor gain share: {red:.0%})", fontsize=11)
    ax.tick_params(axis="y", labelsize=8)
    top = np.argsort(gains)[-5:][::-1]
    print(f"COLGAIN {mname}: " + ", ".join(f"{NAMES[i]}={gains[i]:.1f}" for i in top) + f" | red-share={red:.0%}", flush=True)
axes[-1].set_xticks(range(53)); axes[-1].set_xticklabels(NAMES, rotation=90, fontsize=6)
fig.legend(handles=[mp.Patch(color="#c0392b", label="static distractors (base+tool blocks, flags)"),
                    mp.Patch(color="#e67e22", label="frame world pose"),
                    mp.Patch(color="#27ae60", label="hand-centric (frame-rel, eef, grip)")],
           loc="upper center", ncol=3, fontsize=9, frameon=False)
fig.suptitle("wpmatch (5mm waypoint noise): encoder column gains at settle (f0+f1 summed)", y=1.02, fontsize=13)
plt.tight_layout(rect=[0, 0, 1, 0.95])
plt.savefig("analysis/paper/colgain_wpm.png", dpi=150, bbox_inches="tight")
print("SAVED", flush=True)
