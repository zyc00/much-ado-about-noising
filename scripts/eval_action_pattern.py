"""Concrete action-PATTERN analysis: what do MIP vs MSE(clean) actions actually
DO during self-rollout, such that one converges and one diverges?

Per executed step of each policy's own closed-loop rollout, record physical/measurable
action statistics, binned by OOD-score (kNN dist of state to clean-expert cloud):
  - pos_norm   = ||a[:3]||           (commanded world-frame translation magnitude)
  - rot_norm   = ||a[3:6]||          (commanded rotation magnitude)
  - sat_frac   = mean(|a[:6]|>0.95)  (how bang-bang / saturated)
  - jerk       = ||a_t - a_{t-1}||   (temporal chatter)
  - goal_cos   = cos(a[:3], hole_center - frame_mount_pos)  (is it driving the
                 frame TOWARD the insertion hole? +1 toward, 0 sideways, -1 away)
  - grip       = a[6]
Compares MIP vs MSE in-distribution vs OOD -> reveals the concrete divergence pattern.

  MUJOCO_GL=egl python scripts/eval_action_pattern.py --n 30
"""
import os
os.environ["MUJOCO_GL"] = "egl"
import argparse
import numpy as np
import torch
import h5py
import robosuite
import sys
sys.path.insert(0, "scripts"); sys.path.insert(0, ".")
from collect_tool_hang_demos import ENV_KWARGS
from hydra import initialize_config_dir, compose
from omegaconf import OmegaConf
from scipy.spatial import cKDTree
from mip.datasets.robomimic_dataset import make_dataset
from mip.agent import TrainingAgent

OK = ["object", "robot0_eef_pos", "robot0_eef_quat", "robot0_gripper_qpos"]
KM = {"object": "object-state"}
DS = "data/tool_hang_full2ins_2000.hdf5"


def load(ckpt, loss):
    cfgdir = os.path.abspath("examples/configs")
    with initialize_config_dir(version_base=None, config_dir=cfgdir):
        cfg = compose(config_name="main", overrides=[
            "task=tool_hang_ph_state_delta_legacy", f"+task.dataset_path={os.path.abspath(DS)}",
            "network=chiunet", f"optimization.loss_type={loss}",
            "optimization.auto_resume=false", "log.wandb_mode=disabled"])
    OmegaConf.set_struct(cfg, False); cfg.task.obs_dim = 53
    ds = make_dataset(cfg.task)
    ag = TrainingAgent(cfg); ag.load(ckpt, load_optimizer=False); ag.eval()
    return cfg, ds, ag


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--demos", default="data/warmstart_demos.hdf5")
    ap.add_argument("--n", type=int, default=30); ap.add_argument("--manifold_n", type=int, default=30)
    ap.add_argument("--max_steps", type=int, default=500); ap.add_argument("--H", type=int, default=16)
    ap.add_argument("--models", default=None, help="semicolon list of name,ckpt,ds,loss (overrides default SPEC)")
    args = ap.parse_args()

    if args.models:
        SPEC = {}
        for item in args.models.split(";"):
            nm, ck, dsp, loss = item.split(",")
            SPEC[nm] = (ck, dsp, loss)
    else:
        SPEC = {
            "MIP":         ("logs/full_mip_2000/models/model_latest.pt",        "data/tool_hang_full2ins_2000.hdf5", "mip"),
            "MSE":         ("logs/full_regression_2000/models/model_latest.pt", "data/tool_hang_full2ins_2000.hdf5", "regression"),
            "MSE_dart":    ("logs/dart_full2ins_mse/models/model_latest.pt",    "data/tool_hang_dart_full2ins_2000.hdf5", "regression"),
            "MSE_allrec":  ("logs/allrecov_mse/models/model_latest.pt",         "data/tool_hang_allrecov_6000.hdf5", "regression"),
        }
    A = {}
    for nm, (ck, dsp, loss) in SPEC.items():
        global DS; DS = dsp
        cfg, dsx, ag = load(ck, loss)
        A[nm] = dict(ag=ag, ds=dsx, no=dsx.normalizer["obs"]["state"], na=dsx.normalizer["action"])
    dev = cfg.optimization.device; AS = cfg.task.act_steps; start = cfg.task.obs_steps - 1
    df = h5py.File(args.demos, "r"); allk = list(df["demos"].keys())
    eval_keys = allk[:args.n]; man_keys = allk[args.n:args.n + args.manifold_n]
    env = robosuite.make("ToolHang", horizon=4000, **ENV_KWARGS)

    def ov(o): return np.concatenate([o[KM.get(k, k)] for k in OK]).astype(np.float32)
    def gs(n): return env.sim.data.site_xpos[env.sim.model.site_name2id(n)].copy()
    def hole_center():
        hc = np.mean([env.sim.data.geom_xpos[env.sim.model.geom_name2id(f"stand_wall{i}")] for i in range(4)], axis=0)
        hc[2] = gs("stand_mount_site")[2]; return hc
    def pred(M, window):
        w = np.stack(window[-2:])[None]
        ot = {"state": torch.tensor(M["no"].normalize(w), device=dev, dtype=torch.float32)}
        with torch.no_grad():
            an = M["ag"].sample(act_0=torch.randn((1, args.H, 10), device=dev), obs=ot, use_ema=True)
        return M["ds"].undo_transform_action(M["na"].unnormalize(an.detach().cpu().numpy())[:, start:start + AS])[0]

    man = []
    for k in man_keys:
        d = df["demos/" + k]; sd = int(d.attrs["seed"]); acts = np.clip(d["actions"][:], -1, 1); s0 = d["state0"][:]
        np.random.seed(sd); env.reset(); env.sim.set_state_from_flattened(s0); env.sim.forward()
        arm = env.robots[0].composite_controller.part_controllers["right"]; arm.update(); arm.reset_goal()
        man.append(ov(env._get_observations(force_update=True)))
        for t in range(len(acts)):
            o, _, _, _ = env.step(acts[t]); man.append(ov(o))
    man = np.array(man); mu, sig = man.mean(0), man.std(0) + 1e-6
    tree = cKDTree((man - mu) / sig)
    def ood(v): return float(tree.query((v - mu) / sig)[0])

    def rollout(M):
        recs = []  # (ood, pos_norm, rot_norm, sat, jerk, goal_cos, grip)
        for k in eval_keys:
            d = df["demos/" + k]; sd = int(d.attrs["seed"])
            np.random.seed(sd); env.reset()
            arm = env.robots[0].composite_controller.part_controllers["right"]; arm.update(); arm.reset_goal()
            for _ in range(10):
                env.step(np.zeros(7))
            o = env._get_observations(force_update=True); hist = [ov(o), ov(o)]; steps = 0; asm = False; prev = None
            while steps < args.max_steps and not asm:
                for a in pred(M, hist):
                    fm = gs("frame_mount_site"); hc = hole_center(); dirv = hc - fm
                    dn = np.linalg.norm(dirv); pn = np.linalg.norm(a[:3])
                    gc = float(a[:3] @ dirv / (pn * dn)) if pn > 1e-6 and dn > 1e-6 else np.nan
                    jk = np.linalg.norm(a[:6] - prev[:6]) if prev is not None else np.nan
                    o, _, _, _ = env.step(a); steps += 1; v = ov(o); hist.append(v)
                    recs.append((ood(v), pn, np.linalg.norm(a[3:6]), float(np.mean(np.abs(a[:6]) > 0.95)), jk, gc, float(a[6])))
                    prev = a.copy()
                    if env._check_frame_assembled(): asm = True; break
                    if steps >= args.max_steps: break
        return np.array(recs)

    R = {nm: rollout(A[nm]) for nm in A}
    df.close()
    cols = ["pos_norm", "rot_norm", "sat_frac", "jerk", "goal_cos", "grip"]
    bins = [(0, 1), (1, 2), (2, 5), (5, 1e9)]
    for name, r in R.items():
        oods = r[:, 0]
        print(f"=== {name} (n={len(r)})  by OOD-score bin ===")
        print("  OOD-bin   n     " + "  ".join(f"{c:>8}" for c in cols))
        for lo, hi in bins:
            m = (oods >= lo) & (oods < hi)
            if not m.sum(): continue
            vals = [np.nanmean(r[m, 1 + i]) for i in range(6)]
            print(f"  [{lo:>2},{hi if hi < 1e8 else 'inf':>3}) {m.sum():5d}  " + "  ".join(f"{v:8.3f}" for v in vals))
    np.savez("analysis/recovery/action_pattern.npz", **R)
    print("saved analysis/recovery/action_pattern.npz")


if __name__ == "__main__":
    main()
