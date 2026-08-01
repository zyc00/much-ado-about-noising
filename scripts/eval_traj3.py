"""Per-seed OOD-distance trajectories for MSE-clean, MSE-dart, MIP (same 40 seeds,
same clean-support cloud). 8x5 grid of small subplots, 3 curves each."""
import os
os.environ["MUJOCO_GL"] = "egl"
import argparse
import math
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
import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
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
    ap.add_argument("--n", type=int, default=40); ap.add_argument("--manifold_n", type=int, default=30)
    ap.add_argument("--max_steps", type=int, default=500); ap.add_argument("--H", type=int, default=16)
    args = ap.parse_args()
    SPEC = {"MSEclean": ("logs/full_regression_2000/models/model_latest.pt", "data/tool_hang_full2ins_2000.hdf5", "regression"),
            "MSEdart": ("logs/dart_full2ins_mse/models/model_latest.pt", "data/tool_hang_dart_full2ins_2000.hdf5", "regression"),
            "MIP": ("logs/full_mip_2000/models/model_latest.pt", "data/tool_hang_full2ins_2000.hdf5", "mip")}
    A = {}
    for nm, (ck, dsp, loss) in SPEC.items():
        cfg, ds, ag = load(ck, dsp, loss); A[nm] = dict(ag=ag, ds=ds, no=ds.normalizer["obs"]["state"], na=ds.normalizer["action"])
    dev = cfg.optimization.device; AS = cfg.task.act_steps; start = cfg.task.obs_steps - 1
    df = h5py.File(args.demos, "r"); allk = list(df["demos"].keys())
    eval_keys = allk[:args.n]; man_keys = allk[args.n:args.n + args.manifold_n]
    env = robosuite.make("ToolHang", horizon=4000, **ENV_KWARGS)

    def ov(o): return np.concatenate([o[KM.get(k, k)] for k in OK]).astype(np.float32)
    def predict(M, hist):
        w = np.stack(hist[-2:])[None]
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
        out = []
        for k in eval_keys:
            d = df["demos/" + k]; sd = int(d.attrs["seed"])
            np.random.seed(sd); env.reset()
            arm = env.robots[0].composite_controller.part_controllers["right"]; arm.update(); arm.reset_goal()
            for _ in range(10):
                env.step(np.zeros(7))
            o = env._get_observations(force_update=True); hist = [ov(o), ov(o)]; steps = 0; asm = False; dd = []
            while steps < args.max_steps and not asm:
                for a in predict(M, hist):
                    o, _, _, _ = env.step(a); steps += 1; v = ov(o); hist.append(v); dd.append(ood(v))
                    if env._check_frame_assembled(): asm = True; break
                    if steps >= args.max_steps: break
            out.append((np.array(dd), asm))
        return out

    R = {nm: rollout(A[nm]) for nm in A}
    df.close()
    col = {"MSEclean": "tab:red", "MSEdart": "tab:green", "MIP": "tab:blue"}
    rows, cols = 8, 5
    fig, ax = plt.subplots(rows, cols, figsize=(4 * cols, 2.6 * rows), squeeze=False)
    for j, k in enumerate(eval_keys):
        a = ax[j // cols][j % cols]
        tag = []
        for nm in ["MSEclean", "MSEdart", "MIP"]:
            dd, asm = R[nm][j]
            a.plot(np.clip(dd, 1e-2, None), color=col[nm], lw=1.0, alpha=0.8)
            tag.append(f"{nm[:3] if nm!='MSEclean' else 'Cln'}{'✓' if asm else '✗'}")
        a.axhline(4, color="k", ls="--", lw=0.8); a.axhline(1, color="gray", ls=":", lw=0.6)
        a.set_yscale("log"); a.set_title(f"seed {21000 + j}: " + " ".join(tag), fontsize=7); a.grid(alpha=.2)
        a.tick_params(labelsize=6)
    sr = {nm: sum(x[1] for x in R[nm]) for nm in A}
    fig.suptitle(f"OOD trajectories per seed | red=MSE-clean({sr['MSEclean']}/40) green=MSE-dart({sr['MSEdart']}/40) blue=MIP({sr['MIP']}/40) | dashed=PNR(4)", fontsize=12)
    plt.tight_layout(rect=[0, 0, 1, 0.985]); plt.savefig("analysis/recovery/traj3_40.png", dpi=110)
    print("saved analysis/recovery/traj3_40.png  SR:", sr)


if __name__ == "__main__":
    main()
