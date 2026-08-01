"""Closure census for the oracle-phase MSE arm (56-dim obs: 53 + phase one-hot).
Phase is driven online by the same signal that labeled the data: gripper-command
sign crossing -> closed_at; [1,0,0] until closed_at+16, then [0,1,0].
Env: CKPT, TAG, DS (phase dataset)."""
import os, sys
os.environ["MUJOCO_GL"] = "egl"
sys.path.insert(0, "scripts"); sys.path.insert(0, ".")
import numpy as np, torch, h5py, robosuite
from hydra import initialize_config_dir, compose
from omegaconf import OmegaConf
from mip.datasets.robomimic_dataset import make_dataset
from mip.agent import TrainingAgent
from scripted_tool_hang_v2 import ENV_KWARGS
OK = ["object", "robot0_eef_pos", "robot0_eef_quat", "robot0_gripper_qpos"]
KM = {"object": "object-state"}
DS = os.environ.get("DS", "data/tool_hang_full2ins_2000_phase.hdf5")
T2E = list(range(28, 35))
h = h5py.File(DS, "r"); keys = sorted(h["data"].keys(), key=lambda k: int(k.split("_")[-1]))
data_cl = []
for k in keys[:300]:
    o = h[f"data/{k}/obs"]
    ovd = np.concatenate([np.asarray(o[q]) for q in OK], axis=1).astype(np.float32)
    a = np.clip(np.asarray(h[f"data/{k}/actions"]), -1, 1); g = a[:, 6]
    cl = [t for t in range(1, len(g)) if g[t-1] < 0 and g[t] >= 0]
    if cl: data_cl.append(ovd[cl[0], T2E])
h.close()
D = np.stack(data_cl); mu, sd = D.mean(0), D.std(0) + 1e-8
with initialize_config_dir(version_base=None, config_dir=os.path.abspath("examples/configs")):
    cfg = compose(config_name="main", overrides=["task=tool_hang_ph_state_delta_legacy",
        f"+task.dataset_path={os.path.abspath(DS)}", "network=chiunet",
        "optimization.loss_type=regression", "optimization.auto_resume=false", "log.wandb_mode=disabled"])
OmegaConf.set_struct(cfg, False); cfg.task.obs_dim = 56
ds = make_dataset(cfg.task)
ag = TrainingAgent(cfg); ag.load(os.environ["CKPT"], load_optimizer=False); ag.eval()
name = os.environ.get("TAG", "phase")
env = robosuite.make("ToolHang", horizon=4000, **ENV_KWARGS)
dev = cfg.optimization.device; AS = cfg.task.act_steps; start = cfg.task.obs_steps - 1
no = ds.normalizer["obs"]["state"]; na = ds.normalizer["action"]
def ov(o, phase): return np.concatenate([np.concatenate([o[KM.get(k, k)] for k in OK]), phase]).astype(np.float32)
stats = []
for sd_ in range(21000, 21024):
    np.random.seed(sd_); env.reset()
    arm = env.robots[0].composite_controller.part_controllers["right"]; arm.update(); arm.reset_goal()
    for _ in range(10): env.step(np.zeros(7))
    PH_G, PH_T = np.array([1., 0., 0.]), np.array([0., 1., 0.])
    o = env._get_observations(force_update=True); hist = [ov(o, PH_G), ov(o, PH_G)]
    obs_log = [hist[-1].copy()]; g_log = []
    steps = 0; closed_at = None
    while steps < 250:
        w = np.stack(hist[-2:])[None]
        ot = {"state": torch.tensor(no.normalize(w), device=dev, dtype=torch.float32)}
        with torch.no_grad():
            an = ag.sample(act_0=torch.randn((1, 16, 10), device=dev), obs=ot, use_ema=True)
        ch = ds.undo_transform_action(na.unnormalize(an.detach().cpu().numpy())[:, start:start+AS])[0]
        for a in ch:
            o, _, _, _ = env.step(a); steps += 1
            if closed_at is None and a[6] >= 0: closed_at = steps
            phase = PH_G if (closed_at is None or steps < closed_at + 16) else PH_T
            hist.append(ov(o, phase)); obs_log.append(hist[-1].copy()); g_log.append(a[6])
            if steps >= 250: break
    O = np.array(obs_log); G = np.array(g_log)
    cl = next((i for i in range(1, len(G)) if G[i-1] < 0 and G[i] >= 0), None)
    if cl is None: stats.append(dict(closed=0)); continue
    relz = np.linalg.norm((O[cl, T2E] - mu) / sd) / np.sqrt(7)
    spd = np.linalg.norm(O[cl, 44:47] - O[max(0, cl-4), 44:47]) / 4 * 1000
    reopen = any(G[i] < -0.3 for i in range(cl+1, min(cl+56, len(G))))
    flips = sum(1 for i in range(cl+1, len(G)) if G[i-1] >= 0 > G[i])
    fz = O[:, 23] - O[0, 23]
    lifted = int(fz.max() > 0.1 and fz[-1] > 0.05)
    stats.append(dict(closed=1, relz=relz, spd=spd, reopen=int(reopen), flips=flips, lifted=lifted))
S = [s for s in stats if s["closed"]]
lifted = np.mean([s["lifted"] for s in S]) if S else 0
print(f"{name}: n={len(stats)} closed={len(S)} lifted={lifted:.2f} | relz p50={np.median([s['relz'] for s in S]):.1f} spd p50={np.median([s['spd'] for s in S]):.2f}mm/step | reopen={np.mean([s['reopen'] for s in S]):.2f} reflips p50={np.median([s['flips'] for s in S]):.0f}", flush=True)
print("CENSUS-DONE")
