"""Rigorous test of "MIP's advantage == the recovery DART data teaches".

At identical query states (obs windows from an MSE+clean rollout, which naturally
spans in-distribution early states -> OOD drift states), query several policies
and compare their PREDICTED ACTIONS pairwise, binned by OOD-score
(= z-scored kNN distance of the state to the clean-expert state cloud).

Policies: mse_clean, mip_clean, mse_dart, mse_allrecov (each with its OWN
training-data normalizer). All see the SAME obs window -> a clean counterfactual:
"faced with this (drifted) state, what does each policy do?"

Hypothesis: on OOD states, d(mse_dart, mip_clean) << d(mse_clean, mip_clean)
-- i.e. MIP+clean's OOD action matches the DART-trained recovery action, while
MSE+clean diverges. On in-dist states all agree. Would show MIP's inductive bias
reproduces the recovery behavior DART data supplies explicitly.

  MUJOCO_GL=egl python scripts/eval_action_compare.py --n 25
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


def load(ckpt, ds_path, loss):
    cfgdir = os.path.abspath("examples/configs")
    with initialize_config_dir(version_base=None, config_dir=cfgdir):
        cfg = compose(config_name="main", overrides=[
            "task=tool_hang_ph_state_delta_legacy", f"+task.dataset_path={os.path.abspath(ds_path)}",
            "network=chiunet", f"optimization.loss_type={loss}",
            "optimization.auto_resume=false", "log.wandb_mode=disabled"])
    OmegaConf.set_struct(cfg, False); cfg.task.obs_dim = 53
    ds = make_dataset(cfg.task)
    ag = TrainingAgent(cfg); ag.load(ckpt, load_optimizer=False); ag.eval()
    return cfg, ds, ag


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--demos", default="data/warmstart_demos.hdf5")
    ap.add_argument("--n", type=int, default=25)          # mse_clean rollouts -> query states
    ap.add_argument("--manifold_n", type=int, default=30) # clean demos for OOD-score kNN cloud
    ap.add_argument("--max_steps", type=int, default=500)
    ap.add_argument("--H", type=int, default=16)
    args = ap.parse_args()

    SPEC = {
        "mse_clean":    ("logs/full_regression_2000/models/model_latest.pt", "data/tool_hang_full2ins_2000.hdf5", "regression"),
        "mip_clean":    ("logs/full_mip_2000/models/model_latest.pt",        "data/tool_hang_full2ins_2000.hdf5", "mip"),
        "mse_dart":     ("logs/dart_full2ins_mse/models/model_latest.pt",    "data/tool_hang_dart_full2ins_2000.hdf5", "regression"),
        "mse_allrecov": ("logs/allrecov_mse/models/model_latest.pt",         "data/tool_hang_allrecov_6000.hdf5", "regression"),
    }
    A = {}
    for name, (ck, dsp, loss) in SPEC.items():
        cfg, ds, ag = load(ck, dsp, loss)
        A[name] = dict(ds=ds, ag=ag, no=ds.normalizer["obs"]["state"], na=ds.normalizer["action"])
    dev = cfg.optimization.device; AS = cfg.task.act_steps; start = cfg.task.obs_steps - 1

    df = h5py.File(args.demos, "r"); allk = list(df["demos"].keys())
    eval_keys = allk[:args.n]; man_keys = allk[args.n:args.n + args.manifold_n]
    env = robosuite.make("ToolHang", horizon=4000, **ENV_KWARGS)

    def ov(o): return np.concatenate([o[KM.get(k, k)] for k in OK]).astype(np.float32)

    def act_of(name, window):
        a = A[name]
        w = np.stack(window)[None]
        ot = {"state": torch.tensor(a["no"].normalize(w), device=dev, dtype=torch.float32)}
        with torch.no_grad():
            an = a["ag"].sample(act_0=torch.randn((1, args.H, 10), device=dev), obs=ot, use_ema=True)
        return a["ds"].undo_transform_action(a["na"].unnormalize(an.detach().cpu().numpy())[:, start:start + AS])[0]  # (AS,7) raw

    # clean-expert state cloud for OOD-score
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
    print(f"[cloud] {len(man)} clean states", flush=True)

    # roll out mse_clean, collect (window, ood) at each step; query all policies
    rows = []  # (ood, {name: chunk})
    for k in eval_keys:
        d = df["demos/" + k]; sd = int(d.attrs["seed"])
        np.random.seed(sd); env.reset()
        arm = env.robots[0].composite_controller.part_controllers["right"]; arm.update(); arm.reset_goal()
        for _ in range(10):
            env.step(np.zeros(7))
        o = env._get_observations(force_update=True); hist = [ov(o), ov(o)]; steps = 0; asm = False
        while steps < args.max_steps and not asm:
            window = [hist[-2].copy(), hist[-1].copy()]
            chunks = {name: act_of(name, window) for name in A}
            rows.append((ood(window[-1]), chunks))
            for a in chunks["mse_clean"]:                     # advance with mse_clean (the failing policy)
                o, _, _, _ = env.step(a); steps += 1; hist.append(ov(o))
                if env._check_frame_assembled(): asm = True; break
                if steps >= args.max_steps: break
    df.close()

    oods = np.array([r[0] for r in rows])
    def chunkdist(a, b): return float(np.sqrt(np.mean((a - b) ** 2)))   # RMSE over (AS,7) raw action chunk
    pairs = [("mse_clean", "mip_clean"), ("mse_dart", "mip_clean"), ("mse_allrecov", "mip_clean"),
             ("mse_clean", "mse_dart"), ("mse_dart", "mse_allrecov")]
    print(f"ACTIONCMP n_states={len(rows)} (RMSE of raw 8x7 action chunk, by OOD-score)")
    bins = [(0, 1), (1, 2), (2, 4), (4, 8), (8, 1e9)]
    hdr = "  OOD-bin      n   " + "  ".join(f"{x}-{y}" for x, y in pairs)
    print(hdr)
    for lo, hi in bins:
        m = (oods >= lo) & (oods < hi)
        if not m.sum(): continue
        idx = np.where(m)[0]
        vals = []
        for x, y in pairs:
            ds_ = np.mean([chunkdist(rows[i][1][x], rows[i][1][y]) for i in idx])
            vals.append(f"{ds_:.3f}")
        print(f"  [{lo:>2},{hi if hi < 1e8 else 'inf':>3}) {m.sum():5d}   " + "    ".join(vals))

    # ---- DEVIATION-DIRECTION analysis: do MIP and DART deviate from mse_clean
    # in the SAME direction (same recovery mechanism, maybe different magnitude)? ----
    def flat(c): return c.reshape(-1)
    def cos(a, b):
        na_, nb_ = np.linalg.norm(a), np.linalg.norm(b)
        return float(a @ b / (na_ * nb_)) if na_ > 1e-9 and nb_ > 1e-9 else np.nan
    print("\nDEVIATION DIRECTION (Δ_X = a_X - a_mse_clean):  cos(Δmip,Δdart) cos(Δmip,Δallrecov) cos(Δdart,Δallrecov) | |Δmip| |Δdart| |Δallrecov|")
    for lo, hi in bins:
        m = (oods >= lo) & (oods < hi)
        if not m.sum(): continue
        idx = np.where(m)[0]
        cmd, cma, cda, nmip, ndart, nall = [], [], [], [], [], []
        for i in idx:
            ch = rows[i][1]
            dm = flat(ch["mip_clean"] - ch["mse_clean"])
            dd = flat(ch["mse_dart"] - ch["mse_clean"])
            da = flat(ch["mse_allrecov"] - ch["mse_clean"])
            cmd.append(cos(dm, dd)); cma.append(cos(dm, da)); cda.append(cos(dd, da))
            nmip.append(np.linalg.norm(dm)); ndart.append(np.linalg.norm(dd)); nall.append(np.linalg.norm(da))
        print(f"  [{lo:>2},{hi if hi < 1e8 else 'inf':>3}) n={m.sum():4d}   "
              f"{np.nanmean(cmd):+.3f}          {np.nanmean(cma):+.3f}            {np.nanmean(cda):+.3f}        "
              f"| {np.mean(nmip):.3f}  {np.mean(ndart):.3f}  {np.mean(nall):.3f}")
    # save raw for reuse
    np.savez("analysis/recovery/actioncmp.npz",
             oods=oods,
             mse_clean=np.array([flat(r[1]["mse_clean"]) for r in rows]),
             mip_clean=np.array([flat(r[1]["mip_clean"]) for r in rows]),
             mse_dart=np.array([flat(r[1]["mse_dart"]) for r in rows]),
             mse_allrecov=np.array([flat(r[1]["mse_allrecov"]) for r in rows]))
    print("saved analysis/recovery/actioncmp.npz")


if __name__ == "__main__":
    main()
