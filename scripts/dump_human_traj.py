"""Roll out the historical human matched pair (hMSE_s5, hMIP_s5) on ToolHang and dump
eef trajectories + success flags; also extract GT demo eef trajectories. For the 3D
trajectory figure (GT vs MSE vs MIP on human data).
NOTE: standalone harness — used for trajectory VISUALIZATION only, not SR claims
(official SR = mode=eval)."""
import os
os.environ["MUJOCO_GL"] = "egl"
import numpy as np, torch, h5py, sys
sys.path.insert(0, "scripts"); sys.path.insert(0, ".")
import robosuite
from collect_tool_hang_demos import ENV_KWARGS
from hydra import initialize_config_dir, compose
from omegaconf import OmegaConf
from mip.datasets.robomimic_dataset import make_dataset
from mip.agent import TrainingAgent

DSP = os.environ.get("DSP", "/home/jigu/.cache/huggingface/hub/datasets--ChaoyiPan--mip-dataset/blobs/c5cb01b119a109a3d4842ca515906e86f1f35a8ee1a333d22b3503c1a513a8f4")
OK = ["object", "robot0_eef_pos", "robot0_eef_quat", "robot0_gripper_qpos"]
KM = {"object": "object-state"}
OLD = "/home/jigu/projects/much-ado-about-noising-old/checkpoints"
OUT = "analysis/traj_vis"
os.makedirs(OUT, exist_ok=True)

def load(loss, network="chiunet", extra=()):
    with initialize_config_dir(version_base=None, config_dir=os.path.abspath("examples/configs")):
        cfg = compose(config_name="main", overrides=["task=tool_hang_ph_state_delta_legacy",
            "+task.dataset_path=" + DSP, f"network={network}",
            f"optimization.loss_type={loss}", "optimization.auto_resume=false", "log.wandb_mode=disabled"] + list(extra))
    OmegaConf.set_struct(cfg, False); cfg.task.obs_dim = 53
    ds = make_dataset(cfg.task)
    return cfg, ds, TrainingAgent(cfg)

# ---- GT trajectories from demos
h = h5py.File(DSP, "r")
gts, gt_len = [], []
for i in range(12):
    p = np.asarray(h[f"data/demo_{i}/obs/robot0_eef_pos"]).astype(np.float32)
    gts.append(p); gt_len.append(len(p))
h.close()
np.savez(f"{OUT}/human_gt.npz", **{f"ep{i}": p for i, p in enumerate(gts)})
print(f"GT: {len(gts)} demos, len p50={int(np.median(gt_len))}", flush=True)

MODELS = [("hMSE_s5", "regression", "chiunet", f"{OLD}/tool_hang_ph_state_regression_chiunet_256_seed5_success52.pt")]
if os.environ.get("WITH_MIP"):
    MODELS.append(("hMIP_s5", "mip", "chiunet", f"{OLD}/tool_hang_ph_state_mip_chiunet_256_seed5_success82.pt"))
if os.environ.get("HMLP"):
    MODELS = [("hMSEmlp_s5001", "regression", "mlp", f"{OLD}/tool_hang_ph_state_regression_mlp_512_seed5001_success90.pt"),
              ("hMIPmlp_s5001", "mip", "mlp", f"{OLD}/tool_hang_ph_state_mip_mlp_512_seed5001_success85.pt")]
if os.environ.get("HROTAUXONLY"):
    MODELS = [("hrotaux_s5N", "regression", "chiunet", "logs/hrotaux_s5/models/model_latest.pt",
               ("+task.rot_indicator=true", "task.act_dim=13"))]
if os.environ.get("MODEL_SPEC"):
    MODELS = []
    for spec in os.environ["MODEL_SPEC"].split(","):
        n_, l_, w_, p_ = spec.split(":")
        MODELS.append((n_, l_, w_, p_))
if os.environ.get("HGRID"):
    MODELS = [("hcauchy_chi_s5", "regression_cauchy", "chiunet", "logs/hcauchy_chi_s5/models/model_latest.pt"),
              ("hmse_mlp_s5", "regression", "mlp", "logs/hmse_mlp_s5/models/model_latest.pt"),
              ("hmip_mlp_s5", "mip", "mlp", "logs/hmip_mlp_s5/models/model_latest.pt"),
              ("hcauchy_mlp_s5", "regression_cauchy", "mlp", "logs/hcauchy_mlp_s5/models/model_latest.pt")]
    MODELS = [m for m in MODELS if os.path.exists(m[3])]
if os.environ.get("HROT"):
    MODELS = [("hrotaux_s5", "regression", "chiunet", "logs/hrotaux_s5/models/model_latest.pt",
               ("+task.rot_indicator=true", "task.act_dim=13")),
              ("hrotw_s5", "regression", "chiunet", "logs/hrotw_s5/models/model_latest.pt", ())]
    MODELS = [m for m in MODELS if os.path.exists(m[3])]
if os.environ.get("H5001"):
    MODELS = [("hMSE_s5001", "regression", "chiunet", f"{OLD}/tool_hang_ph_state_regression_chiunet_256_seed5001_success60.pt"),
              ("hMIP_s5001", "mip", "chiunet", f"{OLD}/tool_hang_ph_state_mip_chiunet_256_seed5001_success80.pt")]
MODELS = [m if len(m) == 5 else (*m, ()) for m in MODELS]
for name, loss, network, ck, extra in MODELS:
    cfg, ds, ag = load(loss, network, extra)
    AD = int(cfg.task.act_dim)
    CH = 10 if network == "mlp" else 16
    ag.load(ck, load_optimizer=False); ag.eval()
    dev = cfg.optimization.device; AS = cfg.task.act_steps; start = cfg.task.obs_steps - 1
    no = ds.normalizer["obs"]["state"]; na = ds.normalizer["action"]
    def ov(o):
        return np.concatenate([np.asarray(o[KM.get(k, k)]) for k in OK]).astype(np.float32)
    def chunk(hist):
        w = np.stack(hist[-2:])[None]
        ot = {"state": torch.tensor(no.normalize(w), device=dev, dtype=torch.float32)}
        with torch.no_grad():
            an = ag.sample(act_0=torch.randn((1, CH, AD), device=dev), obs=ot, use_ema=True)
        return ds.undo_transform_action(na.unnormalize(an.detach().cpu().numpy())[:, start:start + AS])[0]
    env = robosuite.make("ToolHang", horizon=4000, **ENV_KWARGS)
    NUDGE = float(os.environ.get("NUDGE", "0"))
    BP_, FP_ = slice(7, 10), slice(21, 24)
    OFFV = np.array([float(x) for x in os.environ.get("GATEOFF", "0,0,0").split(",")], dtype=np.float32)
    eps = {}
    for j, sd in enumerate(range(31000, 31000 + int(os.environ.get("NEP", "12")))):
        np.random.seed(sd); torch.manual_seed(sd); env.reset()
        arm = env.robots[0].composite_controller.part_controllers["right"]; arm.update(); arm.reset_goal()
        for _ in range(10):
            env.step(np.zeros(7))
        o = env._get_observations(force_update=True); hist = [ov(o), ov(o)]
        pser = [np.asarray(o["robot0_eef_pos"]).copy()]
        oser = [hist[-1].copy()]
        aser = []
        steps, succ, asm = 0, False, False
        while steps < 800 and not succ:
            for a in chunk(hist):
                if NUDGE > 0:
                    s53 = hist[-1]
                    v_ = s53[FP_] - (s53[BP_] + OFFV)
                    lat_ = np.linalg.norm(v_[:2])
                    if a[6] >= 0 and 0.010 <= lat_ < 0.060 and 0.005 <= v_[2] < 0.120:
                        a = a.copy(); a[0:2] += NUDGE * (-v_[:2] / (lat_ + 1e-9))
                o, _, _, _ = env.step(a); steps += 1; hist.append(ov(o))
                pser.append(np.asarray(o["robot0_eef_pos"]).copy())
                oser.append(hist[-1].copy()); aser.append(a.copy())
                if env._check_frame_assembled(): asm = True
                if env._check_success(): succ = True; break
                if steps >= 800: break
        eps[f"ep{j}_pos"] = np.stack(pser)
        eps[f"ep{j}_obs"] = np.stack(oser)
        eps[f"ep{j}_act"] = np.stack(aser)
        eps[f"ep{j}_meta"] = np.array([int(succ), int(asm), steps])
        print(f"TRAJ {name} seed={sd}: steps={steps} assembled={int(asm)} success={int(succ)}", flush=True)
    np.savez(f"{OUT}/human_{name}.npz", **eps)
print("DUMP-DONE", flush=True)
