"""Colgain comparison for human matched pair (hMSE_s5 vs hMIP_s5), settle1 + align windows.
Per-53-dim column gain mean_q ||J[:,c]|| with frame pairs (f0+f1) summed; semantic colors."""
import os, sys
os.environ["MUJOCO_GL"] = "egl"
sys.path.insert(0, "scripts"); sys.path.insert(0, ".")
import numpy as np, h5py, torch
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from hydra import initialize_config_dir, compose
from omegaconf import OmegaConf
from mip.datasets.robomimic_dataset import make_dataset
from mip.agent import TrainingAgent

DSP = "/home/jigu/.cache/huggingface/hub/datasets--ChaoyiPan--mip-dataset/blobs/c5cb01b119a109a3d4842ca515906e86f1f35a8ee1a333d22b3503c1a513a8f4"
OK = ["object", "robot0_eef_pos", "robot0_eef_quat", "robot0_gripper_qpos"]
def load(loss):
    with initialize_config_dir(version_base=None, config_dir=os.path.abspath("examples/configs")):
        cfg = compose(config_name="main", overrides=["task=tool_hang_ph_state_delta_legacy",
            "+task.dataset_path=" + DSP, "network=chiunet",
            f"optimization.loss_type={loss}", "optimization.auto_resume=false", "log.wandb_mode=disabled"])
    OmegaConf.set_struct(cfg, False); cfg.task.obs_dim = 53
    ds = make_dataset(cfg.task)
    return cfg, ds, TrainingAgent(cfg)

cfg, ds, _ = load("regression")
no = ds.normalizer["obs"]["state"]; dev = cfg.optimization.device
h = h5py.File(DSP, "r")
keys = sorted(h["data"].keys(), key=lambda k: int(k.split("_")[-1]))
W, wid = [], []
for k in keys:
    o = h[f"data/{k}/obs"]
    ov = np.concatenate([np.asarray(o[q]) for q in OK], axis=1).astype(np.float32)
    a = np.clip(np.asarray(h[f"data/{k}/actions"]), -1, 1); g = a[:, 6]
    cl = [t for t in range(1, len(g)) if g[t-1] < 0 and g[t] >= 0]
    if not cl: continue
    c1 = cl[0]
    rels = [t for t in range(c1 + 1, len(g)) if g[t-1] >= 0 and g[t] < 0]
    rel1 = rels[0] if rels else None
    for t in range(1, len(g), 2):
        wd = 0
        if -12 <= t - c1 < 5: wd = 1
        elif rel1 is not None and rel1 - 30 <= t < rel1: wd = 2
        W.append(np.stack([ov[t-1], ov[t]])); wid.append(wd)
h.close()
W = np.stack(W); wid = np.array(wid)

NAMES = (["b_relx","b_rely","b_relz"] + [f"b_rq{i}" for i in range(4)] + ["b_px","b_py","b_pz"] +
         [f"b_q{i}" for i in range(4)] +
         ["f_relx","f_rely","f_relz"] + [f"f_rq{i}" for i in range(4)] + ["f_px","f_py","f_pz"] +
         [f"f_q{i}" for i in range(4)] +
         ["t_relx","t_rely","t_relz"] + [f"t_rq{i}" for i in range(4)] + ["t_px","t_py","t_pz"] +
         [f"t_q{i}" for i in range(4)] + ["flag0","flag1","eef_x","eef_y","eef_z"] +
         [f"eef_q{i}" for i in range(4)] + ["grip0","grip1"])
def color(i):
    if i < 14: return "#c0392b"                      # base block: static distractor (red)
    if 14 <= i < 21: return "#27ae60"                # frame-rel: hand-centric (green)
    if 21 <= i < 28: return "#e67e22"                # frame world pose (orange)
    if 28 <= i < 42: return "#8e44ad"                # tool block (manipulated on human task)
    if i in (42, 43): return "#7f8c8d"               # flags (gray)
    return "#27ae60"                                  # eef + grip: hand-centric (green)

OLD = "/home/jigu/projects/much-ado-about-noising-old/checkpoints"
MODELS = [("hMSE_s5 (SR 52)", "regression", f"{OLD}/tool_hang_ph_state_regression_chiunet_256_seed5_success52.pt"),
          ("hMIP_s5 (SR 82)", "mip", f"{OLD}/tool_hang_ph_state_mip_chiunet_256_seed5_success82.pt")]
rng = np.random.RandomState(0)
fig, axes = plt.subplots(2, 2, figsize=(17, 7), sharex="col")
G = {}
for col, (wn, wsel) in enumerate([("settle1 (first-grasp settle)", 1), ("align (insert-align hover)", 2)]):
    sel = np.where(wid == wsel)[0]
    canon = W[rng.choice(sel, 30, replace=False)]
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
        G[(mname, wn)] = gains
        ax = axes[row, col]
        ax.bar(range(53), gains, color=[color(i) for i in range(53)])
        ax.set_title(f"{mname} — {wn}", fontsize=11)
        ax.set_xticks(range(53)); ax.set_xticklabels(NAMES, rotation=90, fontsize=5.5)
        ax.tick_params(axis="y", labelsize=8)
        top = np.argsort(gains)[-5:][::-1]
        print(f"COLGAIN {mname} {wn}: " + ", ".join(f"{NAMES[i]}={gains[i]:.1f}" for i in top), flush=True)
import matplotlib.patches as mp
fig.legend(handles=[mp.Patch(color="#c0392b", label="base block (static distractor)"),
                    mp.Patch(color="#8e44ad", label="tool block (manipulated, human task)"),
                    mp.Patch(color="#e67e22", label="frame world pose"),
                    mp.Patch(color="#27ae60", label="hand-centric (frame-rel, eef, grip)"),
                    mp.Patch(color="#7f8c8d", label="flags")],
           loc="upper center", ncol=5, fontsize=9, frameon=False)
fig.suptitle("Human ToolHang: encoder column gains (f0+f1 summed), matched pair seed5", y=1.04, fontsize=13)
plt.tight_layout(rect=[0, 0, 1, 0.94])
plt.savefig("analysis/paper/colgain_human.png", dpi=150, bbox_inches="tight")
print("SAVED analysis/paper/colgain_human.png", flush=True)
