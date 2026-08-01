"""Does MSE 'keep driving' off-support while MIP anchors to edge behavior?
At kicked states s_off with matched on-support anchors s_base:
 M1: drive retention = (t.pi(s_off))/(t.pi(s_base)); |Delta a|; tangent/normal split
 M2: secant-Jacobian profile along the anchor->off ray at f={1/3,2/3,1} (synthetic
     eef-dim displacement, Rsyn protocol): |pi(f+)-pi(f)|/(dist*df)  [action per m]
"""
import os, sys
os.environ["MUJOCO_GL"] = "egl"
sys.path.insert(0, "scripts"); sys.path.insert(0, ".")
import numpy as np, torch, h5py
import robosuite
from hydra import initialize_config_dir, compose
from omegaconf import OmegaConf
from mip.datasets.robomimic_dataset import make_dataset
from mip.agent import TrainingAgent
from scripted_tool_hang_v2 import ENV_KWARGS
from collect_tool_hang_demos import extract_obs

OK = ["object", "robot0_eef_pos", "robot0_eef_quat", "robot0_gripper_qpos"]
POS = slice(44, 47)
H = 16
DS = os.environ.get("DS", "data/tool_hang_full2ins_2000.hdf5")
N_DEMOS = int(os.environ.get("DEMOS", "10"))
N_PROBES = int(os.environ.get("PROBES", "8"))
KICKS = [(0.4, 1), (1.0, 1), (1.0, 2), (1.0, 4)]
NDIRS = 3
PKNOTS = 120

def cat_obs(od):
    return np.concatenate([np.asarray(od[k]).ravel() for k in OK]).astype(np.float32)

hf = h5py.File(DS, "r")
keys = sorted(hf["data"].keys(), key=lambda k: int(k.split("_")[-1]))
grid = np.linspace(0, 1, PKNOTS)
tube = []
for k in keys[:300]:
    p = np.asarray(hf[f"data/{k}/obs/robot0_eef_pos"]); t = np.linspace(0, 1, len(p))
    tube.append(np.stack([np.interp(grid, t, p[:, i]) for i in range(3)], 1))
tube = np.stack(tube); center = tube.mean(0); sigma = tube.std(0).mean(1) + 1e-9
tg = np.gradient(center, axis=0); tg /= (np.linalg.norm(tg, axis=1, keepdims=True) + 1e-12)

cfgdir = os.path.abspath("examples/configs")
agents = []
for spec in os.environ["MODELS"].replace("|", ":").split(","):
    tag, ckpt, loss = spec.split(":")[:3]
    with initialize_config_dir(version_base=None, config_dir=cfgdir):
        cfg = compose(config_name="main", overrides=["task=tool_hang_ph_state_delta_legacy",
            "+task.dataset_path=" + os.path.abspath(os.environ.get("NORMDS", DS)),
            "network=chiunet", f"optimization.loss_type={loss}",
            "optimization.auto_resume=false", "log.wandb_mode=disabled"])
    OmegaConf.set_struct(cfg, False); cfg.task.obs_dim = 53
    ds = make_dataset(cfg.task)
    dev = cfg.optimization.device; start = cfg.task.obs_steps - 1; AS = cfg.task.act_steps
    no = ds.normalizer["obs"]["state"]; na = ds.normalizer["action"]
    ag = TrainingAgent(cfg); ag.load(ckpt, load_optimizer=False); ag.eval()
    g = torch.Generator(device="cpu").manual_seed(0)
    act0 = torch.randn((1, H, 10), generator=g).to(dev)
    def mk(ag=ag, no=no, na=na, ds=ds, dev=dev, act0=act0, start=start, AS=AS):
        def f(win):
            x = torch.tensor(no.normalize(np.stack(win)[None]), device=dev, dtype=torch.float32)
            with torch.no_grad():
                an = ag.sample(act_0=act0.clone(), obs=x, use_ema=True)
            return ds.undo_transform_action(na.unnormalize(an.detach().cpu().numpy())[:, start:start + AS])[0][0, :3]
        return f
    agents.append((tag, mk()))
    print(f"MODEL {tag} loaded", flush=True)

env = robosuite.make("ToolHang", horizon=4000, **ENV_KWARGS)
rng = np.random.RandomState(7)
rows = []
used = 0
for k in keys:
    if used >= N_DEMOS: break
    dgrp = hf[f"data/{k}"]
    seed = int(dgrp.attrs.get("seed", -1))
    if seed < 0: continue
    acts = np.asarray(dgrp["actions"])
    rec_eef = np.asarray(dgrp["obs/robot0_eef_pos"])
    T = len(acts)
    np.random.seed(seed); ob = env.reset()
    obs_hist = [cat_obs(extract_obs(ob))]
    ok_rep = True
    probe_at = set(np.linspace(12, T - 20, N_PROBES).astype(int))
    for t in range(T):
        if t == 40 and np.linalg.norm(env._get_observations()["robot0_eef_pos"] - rec_eef[t]) > 0.005:
            ok_rep = False; break
        if t in probe_at:
            o_now = cat_obs(extract_obs(env._get_observations()))
            st = env.sim.get_state()
            win_base = [obs_hist[-1], o_now]
            p_idx = min(int(round(t / max(T - 1, 1) * (PKNOTS - 1))), PKNOTS - 1)
            that = tg[p_idx]
            a_base = {tag: f(win_base) for tag, f in agents}
            for rho, ns in KICKS:
                for _ in range(NDIRS):
                    u = rng.randn(3); u /= np.linalg.norm(u)
                    o_new = o_now
                    for kk in range(ns):
                        a_k = acts[min(t + kk, T - 1)].copy()
                        a_k[:3] = np.clip(a_k[:3] + rho * u, -1, 1)
                        ob2, *_ = env.step(a_k)
                        o_new = cat_obs(extract_obs(ob2))
                    dv = o_new[POS] - center[min(int(round((t+ns)/max(T-1,1)*(PKNOTS-1))), PKNOTS-1)]
                    d = np.linalg.norm(dv) / sigma[p_idx]
                    nout = dv / (np.linalg.norm(dv) + 1e-12)
                    dv_full = o_new[POS] - o_now[POS]
                    dist_m = np.linalg.norm(dv_full)
                    # ray profile: synthetic displacement of base window by f*dv_full
                    for tag, f in agents:
                        prof = [a_base[tag]]
                        for fr in (1/3, 2/3, 1.0):
                            w = [win_base[0].copy(), win_base[1].copy()]
                            for fm in w: fm[POS] = fm[POS] + fr * dv_full
                            prof.append(f(w))
                        a_off = prof[-1]
                        da = a_off - a_base[tag]
                        drv_b = float(that @ a_base[tag]); drv_o = float(that @ a_off)
                        ratio = drv_o / drv_b if abs(drv_b) > 0.05 else np.nan
                        sec = [float(np.linalg.norm(prof[i+1] - prof[i])) / (dist_m / 3 + 1e-9) for i in range(3)]
                        rows.append((tag, d, ratio, float(np.linalg.norm(da)),
                                     float(that @ da), -float(nout @ da),
                                     sec[0], sec[1], sec[2], dist_m))
                    env.sim.set_state(st); env.sim.forward()
        ob, *_ = env.step(acts[t])
        obs_hist.append(cat_obs(extract_obs(ob)))
        if len(obs_hist) > 3: obs_hist.pop(0)
    if ok_rep:
        used += 1; print(f"DEMO {k} done", flush=True)
hf.close()
import collections
by = collections.defaultdict(list)
for r in rows: by[r[0]].append(r[1:])
BANDS = [(1, 2), (2, 4), (4, 8), (8, 20)]
for tag, rr in by.items():
    A = np.array(rr, dtype=float)
    for lo, hi in BANDS:
        m = (A[:, 0] >= lo) & (A[:, 0] < hi)
        if m.sum() < 10: continue
        B = A[m]
        rat = B[:, 1][~np.isnan(B[:, 1])]
        print(f"OFFANCH {tag} band=[{lo},{hi}) n={m.sum()} "
              f"drive_ratio_p50={np.median(rat):+.3f} |da|_p50={np.median(B[:,2]):.4f} "
              f"tanΔ_p50={np.median(B[:,3]):+.4f} inwΔ_p50={np.median(B[:,4]):+.4f} "
              f"sec13={np.median(B[:,5]):.2f} sec23={np.median(B[:,6]):.2f} sec33={np.median(B[:,7]):.2f}")
print("OFFANCH-DONE")
