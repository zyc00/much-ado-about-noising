"""Closure census for single-stage grasp specialists (local)."""
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
DS = os.environ.get("DS", "data/tool_hang_init2grasp_2000.hdf5")
T2E = list(range(28, 35))
h = h5py.File(DS, "r"); keys = sorted(h["data"].keys(), key=lambda k: int(k.split("_")[-1]))
data_cl = []
for k in keys[:300]:
    o = h[f"data/{k}/obs"]
    ov = np.concatenate([np.asarray(o[q]) for q in OK], axis=1).astype(np.float32)
    a = np.clip(np.asarray(h[f"data/{k}/actions"]), -1, 1); g = a[:, 6]
    cl = [t for t in range(1, len(g)) if g[t-1] < 0 and g[t] >= 0]
    if cl: data_cl.append(ov[cl[0], T2E])
h.close()
D = np.stack(data_cl); mu, sd = D.mean(0), D.std(0) + 1e-8
def load_agent(loss, ck):
    with initialize_config_dir(version_base=None, config_dir=os.path.abspath("examples/configs")):
        cfg = compose(config_name="main", overrides=["task=tool_hang_ph_state_delta_legacy",
            f"+task.dataset_path={os.path.abspath(DS)}", "network=chiunet",
            f"optimization.loss_type={loss}", "optimization.auto_resume=false", "log.wandb_mode=disabled"])
    OmegaConf.set_struct(cfg, False); cfg.task.obs_dim = 53
    ds = make_dataset(cfg.task)
    ag = TrainingAgent(cfg); ag.load(ck, load_optimizer=False); ag.eval()
    return cfg, ds, ag
env = robosuite.make("ToolHang", horizon=4000, **ENV_KWARGS)
def ov(o): return np.concatenate([o[KM.get(k, k)] for k in OK]).astype(np.float32)
for loss, ck, name in [("regression", os.environ["CKPT"], os.environ.get("TAG", "arm"))]:
    cfg, ds, ag = load_agent(loss, ck)
    dev = cfg.optimization.device; AS = cfg.task.act_steps; start = cfg.task.obs_steps - 1
    no = ds.normalizer["obs"]["state"]; na = ds.normalizer["action"]
    stats = []
    for sd in range(21000, 21024):
        np.random.seed(sd); env.reset()
        arm = env.robots[0].composite_controller.part_controllers["right"]; arm.update(); arm.reset_goal()
        for _ in range(10): env.step(np.zeros(7))
        o = env._get_observations(force_update=True); hist = [ov(o), ov(o)]
        obs_log = [hist[-1].copy()]; g_log = []
        steps = 0
        while steps < 250:
            w = np.stack(hist[-2:])[None]
            ot = {"state": torch.tensor(no.normalize(w), device=dev, dtype=torch.float32)}
            with torch.no_grad():
                an = ag.sample(act_0=torch.randn((1, 16, 10), device=dev), obs=ot, use_ema=True)
            ch = ds.undo_transform_action(na.unnormalize(an.detach().cpu().numpy())[:, start:start+AS])[0]
            for a in ch:
                o, _, _, _ = env.step(a); steps += 1
                hist.append(ov(o)); obs_log.append(hist[-1].copy()); g_log.append(a[6])
                if steps >= 250: break
        O = np.array(obs_log); G = np.array(g_log)
        cl = next((i for i in range(1, len(G)) if G[i-1] < 0 and G[i] >= 0), None)
        if cl is None: stats.append(dict(closed=0)); continue
        relz = np.linalg.norm((O[cl, T2E] - mu) / sd) / np.sqrt(7)
        spd = np.linalg.norm(O[cl, 44:47] - O[max(0,cl-4), 44:47]) / 4 * 1000
        reopen = any(G[i] < -0.3 for i in range(cl+1, min(cl+56, len(G))))
        flips = sum(1 for i in range(cl+1, len(G)) if G[i-1] >= 0 > G[i])
        fz = O[:, 23] - O[0, 23]
        lifted = int(fz.max() > 0.1 and fz[-1] > 0.05)
        stats.append(dict(closed=1, relz=relz, spd=spd, reopen=int(reopen), flips=flips, lifted=lifted))
    S = [s for s in stats if s["closed"]]
    lifted = np.mean([s["lifted"] for s in S]) if S else 0
    print(f"{name}: n={len(stats)} closed={len(S)} lifted={lifted:.2f} | relz p50={np.median([s['relz'] for s in S]):.1f} spd p50={np.median([s['spd'] for s in S]):.2f}mm/step | reopen={np.mean([s['reopen'] for s in S]):.2f} reflips p50={np.median([s['flips'] for s in S]):.0f}", flush=True)
print("CENSUS-DONE")
