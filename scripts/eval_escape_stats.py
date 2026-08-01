"""Quantify how OFTEN each model reaches off-support (vs how bad it is once there).
Rolls out the 5 models on the same 40 seeds, records per-step OOD distance to the
clean-support cloud, and reports escape statistics. To avoid the bias that failed
runs linger at high distance (inflating 'fraction of off-support steps'), the main
incidence metric is PER-SEED reach-incidence (binary: did the trajectory EVER cross
threshold) and first-passage step, plus a success-only time-off-support."""
import os
os.environ["MUJOCO_GL"] = "egl"
import argparse
import numpy as np
import torch
import h5py
import sys
sys.path.insert(0, "scripts"); sys.path.insert(0, ".")
from collect_tool_hang_demos import ENV_KWARGS
from hydra import initialize_config_dir, compose
from omegaconf import OmegaConf
from scipy.spatial import cKDTree
import robosuite
from mip.datasets.robomimic_dataset import make_dataset
from mip.agent import TrainingAgent

OK = ["object", "robot0_eef_pos", "robot0_eef_quat", "robot0_gripper_qpos"]
KM = {"object": "object-state"}
DS_DECODE = "data/tool_hang_full2ins_2000.hdf5"


def build_cfg(loss):
    cfgdir = os.path.abspath("examples/configs")
    with initialize_config_dir(version_base=None, config_dir=cfgdir):
        cfg = compose(config_name="main", overrides=[
            "task=tool_hang_ph_state_delta_legacy", f"+task.dataset_path={os.path.abspath(DS_DECODE)}",
            "network=chiunet", f"optimization.loss_type={loss}",
            "optimization.auto_resume=false", "log.wandb_mode=disabled"])
    OmegaConf.set_struct(cfg, False); cfg.task.obs_dim = 53
    return cfg


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--demos", default="data/warmstart_demos.hdf5")
    ap.add_argument("--n", type=int, default=40); ap.add_argument("--manifold_n", type=int, default=30)
    ap.add_argument("--max_steps", type=int, default=500); ap.add_argument("--H", type=int, default=16)
    args = ap.parse_args()

    base_cfg = build_cfg("regression")
    ds = make_dataset(base_cfg.task)
    dev = base_cfg.optimization.device; AS = base_cfg.task.act_steps; start = base_cfg.task.obs_steps - 1
    n2 = ds.normalizer
    n20 = torch.load("analysis/recovery/norm_clean20k.pt", weights_only=False)
    nD = torch.load("analysis/recovery/norm_puredart6k.pt", weights_only=False)
    SPEC = [
        ("MSE-2k",  "logs/full_regression_2000/models/model_latest.pt",  "regression", n2),
        ("MSE-20k", "logs/full_regression_20000/models/model_latest.pt", "regression", n20),
        ("MIP-2k",  "logs/full_mip_2000/models/model_latest.pt",         "mip",        n2),
        ("MIP-20k", "logs/full_mip_20000/models/model_latest.pt",        "mip",        n20),
        ("DART-6k", "logs/puredart_mse/models/model_latest.pt",          "regression", nD),
    ]
    M = {}
    for lab, ck, loss, nrm in SPEC:
        ag = TrainingAgent(build_cfg(loss)); ag.load(ck, load_optimizer=False); ag.eval()
        M[lab] = dict(ag=ag, no=nrm["obs"]["state"], na=nrm["action"])

    df = h5py.File(args.demos, "r"); allk = list(df["demos"].keys())
    eval_keys = allk[:args.n]; man_keys = allk[args.n:args.n + args.manifold_n]
    env = robosuite.make("ToolHang", horizon=4000, **ENV_KWARGS)

    def ov(o): return np.concatenate([o[KM.get(k, k)] for k in OK]).astype(np.float32)

    def predict(m, hist):
        w = np.stack(hist[-2:])[None]
        ot = {"state": torch.tensor(m["no"].normalize(w), device=dev, dtype=torch.float32)}
        with torch.no_grad():
            an = m["ag"].sample(act_0=torch.randn((1, args.H, 10), device=dev), obs=ot, use_ema=True)
        return ds.undo_transform_action(m["na"].unnormalize(an.detach().cpu().numpy())[:, start:start + AS])[0]

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

    def rollout(m):
        seeds = []
        for k in eval_keys:
            d = df["demos/" + k]; sd = int(d.attrs["seed"])
            np.random.seed(sd); env.reset()
            arm = env.robots[0].composite_controller.part_controllers["right"]; arm.update(); arm.reset_goal()
            for _ in range(10):
                env.step(np.zeros(7))
            o = env._get_observations(force_update=True); hist = [ov(o), ov(o)]; steps = 0; asm = False; dd = []
            while steps < args.max_steps and not asm:
                for a in predict(m, hist):
                    o, _, _, _ = env.step(a); steps += 1; v = ov(o); hist.append(v); dd.append(ood(v))
                    if env._check_frame_assembled(): asm = True; break
                    if steps >= args.max_steps: break
            seeds.append((np.array(dd), asm))
        return seeds

    R = {lab: rollout(M[lab]) for lab in M}
    df.close()
    np.savez("analysis/recovery/escape_stats.npz",
             **{f"{lab}_d{j}": R[lab][j][0] for lab in M for j in range(len(eval_keys))},
             **{f"{lab}_asm": np.array([R[lab][j][1] for j in range(len(eval_keys))]) for lab in M})

    THR = [2.45, 3.0, 4.0]  # p99, braking, PNR
    print(f"40 seeds. p95 in-domain=1.79, p99=2.45.  thresholds = {THR}")
    print(f"\n{'model':9} {'SR':>5} | {'reach>2.45':>10} {'reach>3':>8} {'reach>PNR4':>10} | "
          f"{'1stPass>2.45':>12} | {'succ-only %time>1.79':>20} {'mean maxd':>10}")
    for lab in M:
        seeds = R[lab]; ns = len(seeds); sr = 100 * sum(a for _, a in seeds) / ns
        def reach(thr): return 100 * sum(1 for dd, _ in seeds if dd.size and dd.max() > thr) / ns
        fps = [int(np.argmax(dd > 2.45)) for dd, _ in seeds if dd.size and dd.max() > 2.45]
        fp = np.median(fps) if fps else float("nan")
        # success-only fraction of steps above in-domain p95 (failed runs excluded to avoid linger bias)
        succ_dd = [dd for dd, a in seeds if a and dd.size]
        if succ_dd:
            tof = 100 * np.mean([np.mean(dd > 1.79) for dd in succ_dd])
        else:
            tof = float("nan")
        maxd = np.mean([dd.max() if dd.size else 0 for dd, _ in seeds])
        print(f"{lab:9} {sr:>4.0f}% | {reach(2.45):>9.0f}% {reach(3):>7.0f}% {reach(4):>9.0f}% | "
              f"{fp:>12.0f} | {tof:>19.1f}% {maxd:>10.2f}")
    print("\nreach>thr = % of 40 seeds whose trajectory EVER crosses thr (escape incidence, unbiased by linger)")
    print("1stPass = median step of first crossing 2.45 among seeds that cross it")
    print("succ-only %time>1.79 = among SUCCESSFUL runs, mean fraction of steps above in-domain p95")
    print("saved analysis/recovery/escape_stats.npz")


if __name__ == "__main__":
    main()
