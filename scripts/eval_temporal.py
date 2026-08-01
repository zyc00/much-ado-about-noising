"""Temporal analysis: WHEN/WHERE along the rollout does each policy leave the
support space? Per-step OOD-score (kNN dist to clean-expert state cloud) + phase
tag (reach / transport / insert), for MSE vs MIP, from init.

Outputs:
  (a) OOD-score vs normalized progress (median) + %episodes off-support(>tau) vs progress
  (b) per-PHASE off-support rate (reach/transport/insert): which phase does it leave?
  (c) first-crossing (onset) time distribution: when does OOD first exceed tau?
  (d) recover-vs-diverge after onset.
Saves npz for plotting.

  MUJOCO_GL=egl python scripts/eval_temporal.py --n 40
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
    ap.add_argument("--n", type=int, default=40); ap.add_argument("--manifold_n", type=int, default=30)
    ap.add_argument("--max_steps", type=int, default=500); ap.add_argument("--H", type=int, default=16)
    ap.add_argument("--tau", type=float, default=5.0); ap.add_argument("--bins", type=int, default=25)
    args = ap.parse_args()

    cfg, ds, mse = load("logs/full_regression_2000/models/model_latest.pt", "regression")
    _, _, mip = load("logs/full_mip_2000/models/model_latest.pt", "mip")
    dev = cfg.optimization.device; AS = cfg.task.act_steps; start = cfg.task.obs_steps - 1
    no = ds.normalizer["obs"]["state"]; na = ds.normalizer["action"]
    df = h5py.File(args.demos, "r"); allk = list(df["demos"].keys())
    eval_keys = allk[:args.n]; man_keys = allk[args.n:args.n + args.manifold_n]
    env = robosuite.make("ToolHang", horizon=4000, **ENV_KWARGS)

    def ov(o): return np.concatenate([o[KM.get(k, k)] for k in OK]).astype(np.float32)
    def gs(n): return env.sim.data.site_xpos[env.sim.model.site_name2id(n)].copy()
    def grasped():
        g = env._check_grasp(gripper=env.robots[0].gripper["right"], object_geoms=env.frame.contact_geoms)
        fz = env.sim.data.site_xpos[env.sim.model.site_name2id("frame_mount_site")][2]
        return bool(g and fz > 0.86)
    def hole_xy():
        hc = np.mean([env.sim.data.geom_xpos[env.sim.model.geom_name2id(f"stand_wall{i}")] for i in range(4)], axis=0)
        return float(np.linalg.norm((gs("frame_mount_site") - hc)[:2]))
    def phase():
        if not grasped(): return 0      # reach
        return 2 if hole_xy() < 0.05 else 1   # 2=insert (near hole), 1=transport
    def pred(ag, hist):
        w = np.stack(hist[-2:])[None]
        ot = {"state": torch.tensor(no.normalize(w), device=dev, dtype=torch.float32)}
        with torch.no_grad():
            an = ag.sample(act_0=torch.randn((1, args.H, 10), device=dev), obs=ot, use_ema=True)
        return ds.undo_transform_action(na.unnormalize(an.detach().cpu().numpy())[:, start:start + AS])[0]

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

    def rollout(ag):
        eps = []  # per ep: dict(dist[], phase[], asm, onset)
        for k in eval_keys:
            d = df["demos/" + k]; sd = int(d.attrs["seed"])
            np.random.seed(sd); env.reset()
            arm = env.robots[0].composite_controller.part_controllers["right"]; arm.update(); arm.reset_goal()
            for _ in range(10):
                env.step(np.zeros(7))
            o = env._get_observations(force_update=True); hist = [ov(o), ov(o)]
            steps = 0; asm = False; dist = []; phs = []
            while steps < args.max_steps and not asm:
                for a in pred(ag, hist):
                    o, _, _, _ = env.step(a); steps += 1; v = ov(o); hist.append(v)
                    dist.append(ood(v)); phs.append(phase())
                    if env._check_frame_assembled(): asm = True; break
                    if steps >= args.max_steps: break
            dist = np.array(dist); phs = np.array(phs)
            onset = int(np.argmax(dist > args.tau)) if (dist > args.tau).any() else -1
            eps.append(dict(dist=dist, phase=phs, asm=asm, onset=onset))
        return eps

    R = {"MSE": rollout(mse), "MIP": rollout(mip)}
    df.close()
    B = args.bins
    def binned(c):
        return np.interp(np.linspace(0, 1, B), np.linspace(0, 1, len(c)), c) if len(c) >= 2 else np.full(B, np.nan)

    for nm, eps in R.items():
        curves = np.array([binned(e["dist"]) for e in eps])
        offfrac = np.array([binned((e["dist"] > args.tau).astype(float)) for e in eps]).mean(0) * 100
        med = np.nanmedian(curves, 0)
        # per-phase off-support rate (over all steps pooled)
        ph_off = {}
        for ph, label in [(0, "reach"), (1, "transport"), (2, "insert")]:
            allp = np.concatenate([e["phase"] for e in eps]); alld = np.concatenate([e["dist"] for e in eps])
            m = allp == ph
            ph_off[label] = (100 * np.mean(alld[m] > args.tau) if m.sum() else np.nan, int(m.sum()))
        onsets = [e["onset"] for e in eps if e["onset"] >= 0]
        nsucc = sum(e["asm"] for e in eps)
        print(f"=== {nm}  success={nsucc}/{len(eps)}  off-support steps escape-rate ===")
        print("  per-phase off-support%%: " + "  ".join(f"{k}={v[0]:.0f}%(n{v[1]})" for k, v in ph_off.items()))
        print(f"  onset(first OOD>{args.tau}): {len(onsets)}/{len(eps)} eps cross; median onset step={np.median(onsets) if onsets else 'NA'}")
        print("  median OOD by progress(25): " + " ".join(f"{x:.1f}" for x in med))
        print("  %%off by progress(25):      " + " ".join(f"{x:.0f}" for x in offfrac))
        R[nm] = dict(curves=curves, offfrac=offfrac, med=med, ph_off=ph_off,
                     onsets=np.array(onsets), nsucc=nsucc, n=len(eps))
    np.savez("analysis/recovery/temporal.npz",
             MSE_curves=R["MSE"]["curves"], MIP_curves=R["MIP"]["curves"],
             MSE_off=R["MSE"]["offfrac"], MIP_off=R["MIP"]["offfrac"],
             MSE_onsets=R["MSE"]["onsets"], MIP_onsets=R["MIP"]["onsets"])
    print("saved analysis/recovery/temporal.npz")


if __name__ == "__main__":
    main()
