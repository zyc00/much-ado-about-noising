"""How does the MIP-vs-MSE(clean) deviation ARISE? Teacher-forcing vs self-forcing.

TEACHER-FORCED (both policies see the IDENTICAL expert state sequence, replayed):
  per-step open-loop action error vs the expert chunk, for MIP and MSE, plus their
  mutual agreement. -> Are MIP & MSE different ON the expert path?
SELF-FORCED (each policy rolls out closed-loop from init, on ITS OWN states):
  state OOD-score (kNN dist to clean-expert state cloud) over normalized progress,
  for MIP and MSE. -> Does the deviation get BORN by trajectory divergence?

Expectation: TF shows MIP~MSE (small, maybe MIP slightly worse open-loop); SF shows
MSE state-trajectory diverging (compounding) while MIP stays bounded -> the gap is a
closed-loop compounding effect, invisible under teacher forcing.

  MUJOCO_GL=egl python scripts/eval_tf_sf.py --n 30
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
    ap.add_argument("--n", type=int, default=30)
    ap.add_argument("--manifold_n", type=int, default=30)
    ap.add_argument("--stride", type=int, default=4)
    ap.add_argument("--max_steps", type=int, default=500)
    ap.add_argument("--H", type=int, default=16); ap.add_argument("--bins", type=int, default=10)
    args = ap.parse_args()

    cfg, ds, mse = load("logs/full_regression_2000/models/model_latest.pt", "regression")
    _, _, mip = load("logs/full_mip_2000/models/model_latest.pt", "mip")
    dev = cfg.optimization.device; AS = cfg.task.act_steps; start = cfg.task.obs_steps - 1
    no = ds.normalizer["obs"]["state"]; na = ds.normalizer["action"]
    df = h5py.File(args.demos, "r"); allk = list(df["demos"].keys())
    eval_keys = allk[:args.n]; man_keys = allk[args.n:args.n + args.manifold_n]
    env = robosuite.make("ToolHang", horizon=4000, **ENV_KWARGS)

    def ov(o): return np.concatenate([o[KM.get(k, k)] for k in OK]).astype(np.float32)
    def pred(ag, window):
        w = np.stack(window[-2:])[None]
        ot = {"state": torch.tensor(no.normalize(w), device=dev, dtype=torch.float32)}
        with torch.no_grad():
            an = ag.sample(act_0=torch.randn((1, args.H, 10), device=dev), obs=ot, use_ema=True)
        return ds.undo_transform_action(na.unnormalize(an.detach().cpu().numpy())[:, start:start + AS])[0]

    # clean state cloud for SF OOD-score
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

    def rmse(a, b): return float(np.sqrt(np.mean((a - b) ** 2)))

    # ---- TEACHER-FORCED: replay expert, query both on expert states ----
    tf = {"grasp": {"mip_e": [], "mse_e": [], "mm": []}, "insert": {"mip_e": [], "mse_e": [], "mm": []}}
    for k in eval_keys:
        d = df["demos/" + k]; sd = int(d.attrs["seed"]); c1 = int(d.attrs["c1"])
        acts = np.clip(d["actions"][:], -1, 1); s0 = d["state0"][:]
        np.random.seed(sd); env.reset(); env.sim.set_state_from_flattened(s0); env.sim.forward()
        arm = env.robots[0].composite_controller.part_controllers["right"]; arm.update(); arm.reset_goal()
        o = env._get_observations(force_update=True); hist = [ov(o), ov(o)]; T = len(acts)
        for t in range(T):
            if t >= 1 and t % args.stride == 0 and t + AS <= T:
                am = pred(mip, hist); ar = pred(mse, hist); tgt = acts[t:t + AS]
                ph = "grasp" if t < c1 else "insert"
                tf[ph]["mip_e"].append(rmse(am, tgt)); tf[ph]["mse_e"].append(rmse(ar, tgt)); tf[ph]["mm"].append(rmse(am, ar))
            o, _, _, _ = env.step(acts[t]); hist.append(ov(o))

    # ---- SELF-FORCED: roll out each policy from init, record OOD-score over progress ----
    B = args.bins
    def selfroll(ag):
        curves = []
        for k in eval_keys:
            d = df["demos/" + k]; sd = int(d.attrs["seed"])
            np.random.seed(sd); env.reset()
            arm = env.robots[0].composite_controller.part_controllers["right"]; arm.update(); arm.reset_goal()
            for _ in range(10):
                env.step(np.zeros(7))
            o = env._get_observations(force_update=True); hist = [ov(o), ov(o)]; steps = 0; asm = False; dd = []
            while steps < args.max_steps and not asm:
                for a in pred(ag, hist):
                    o, _, _, _ = env.step(a); steps += 1; v = ov(o); hist.append(v); dd.append(ood(v))
                    if env._check_frame_assembled(): asm = True; break
                    if steps >= args.max_steps: break
            dd = np.array(dd)
            if len(dd) >= 2:
                curves.append(np.interp(np.linspace(0, 1, B), np.linspace(0, 1, len(dd)), dd))
        return np.array(curves)
    sf_mip = selfroll(mip); sf_mse = selfroll(mse)
    df.close()

    print("=== TEACHER-FORCED (open-loop, on EXPERT states): RMSE of action chunk ===")
    for ph in ["grasp", "insert"]:
        g = tf[ph]
        print(f"  {ph:6s} n={len(g['mm']):4d}  MIP-vs-expert={np.mean(g['mip_e']):.4f}  MSE-vs-expert={np.mean(g['mse_e']):.4f}  MIP-vs-MSE={np.mean(g['mm']):.4f}")
    print("\n=== SELF-FORCED (closed-loop, on OWN states): state OOD-score (median) over progress ===")
    xs = np.linspace(0, 1, B)
    print("  progress: " + " ".join(f"{x:.1f}" for x in xs))
    print("  MIP med : " + " ".join(f"{np.nanmedian(sf_mip[:, i]):.2f}" for i in range(B)))
    print("  MSE med : " + " ".join(f"{np.nanmedian(sf_mse[:, i]):.2f}" for i in range(B)))
    print(f"  MIP %off(>2) end={100*np.mean(sf_mip[:,-1]>2):.0f}%  MSE %off end={100*np.mean(sf_mse[:,-1]>2):.0f}%")
    np.savez("analysis/recovery/tf_sf.npz", tf=np.array([[np.mean(tf[p][k]) for k in ['mip_e','mse_e','mm']] for p in ['grasp','insert']]),
             sf_mip=sf_mip, sf_mse=sf_mse)
    print("saved analysis/recovery/tf_sf.npz")


if __name__ == "__main__":
    main()
