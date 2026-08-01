"""Per-seed OOD-distance trajectories for 5 models (MSE-clean-2k, MSE-clean-20k,
MIP-2k, MIP-20k, DART-6k) on the same 40 seeds, same clean-support cloud. 8x5 grid
of small subplots, 5 curves each. Same format as traj3_40.png. Saves to
analysis/recovery/mip_saves_mse_fails.png.

Shares ONE dataset (full2ins_2000) for undo_transform_action + dims; each model uses
its own normalizer (override) and its own loss_type cfg (mip=2-step, regression=1-step)."""
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
import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
from mip.datasets.robomimic_dataset import make_dataset
from mip.agent import TrainingAgent

OK = ["object", "robot0_eef_pos", "robot0_eef_quat", "robot0_gripper_qpos"]
KM = {"object": "object-state"}
DS_DECODE = "data/tool_hang_full2ins_2000.hdf5"   # only for undo_transform_action + dims


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

    # shared decode dataset + base cfg
    base_cfg = build_cfg("regression")
    ds = make_dataset(base_cfg.task)
    dev = base_cfg.optimization.device; AS = base_cfg.task.act_steps; start = base_cfg.task.obs_steps - 1

    n2 = ds.normalizer            # 2k clean normalizer
    n20 = torch.load("analysis/recovery/norm_clean20k.pt", weights_only=False)
    nD = torch.load("analysis/recovery/norm_puredart6k.pt", weights_only=False)

    # (label, ckpt, loss, normalizer, color, linestyle)
    SPEC = [
        ("MSE-2k",  "logs/full_regression_2000/models/model_latest.pt",  "regression", n2,  "tab:red",   "-"),
        ("MSE-20k", "logs/full_regression_20000/models/model_latest.pt", "regression", n20, "darkred",   "--"),
        ("MIP-2k",  "logs/full_mip_2000/models/model_latest.pt",         "mip",        n2,  "tab:blue",  "-"),
        ("MIP-20k", "logs/full_mip_20000/models/model_latest.pt",        "mip",        n20, "navy",      "--"),
        ("DART-6k", "logs/puredart_mse/models/model_latest.pt",          "regression", nD,  "tab:green", "-"),
    ]
    M = {}
    for lab, ck, loss, nrm, col, ls in SPEC:
        cfg = build_cfg(loss)
        ag = TrainingAgent(cfg); ag.load(ck, load_optimizer=False); ag.eval()
        M[lab] = dict(ag=ag, no=nrm["obs"]["state"], na=nrm["action"], col=col, ls=ls)

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

    # clean-support cloud from manifold demos (env replay)
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
        out = []
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
            out.append((np.array(dd), asm))
        return out

    R = {lab: rollout(M[lab]) for lab in M}
    df.close()
    rows, cols = 8, 5
    fig, ax = plt.subplots(rows, cols, figsize=(4 * cols, 2.6 * rows), squeeze=False)
    for j, k in enumerate(eval_keys):
        a = ax[j // cols][j % cols]
        tag = []
        for lab in M:
            dd, asm = R[lab][j]
            a.plot(np.clip(dd, 1e-2, None), color=M[lab]["col"], ls=M[lab]["ls"], lw=1.1, alpha=0.85)
            tag.append(f"{lab}{'✓' if asm else '✗'}")
        a.axhline(4, color="k", ls="--", lw=0.8); a.axhline(1.79, color="gray", ls=":", lw=0.6)
        a.set_yscale("log"); a.set_title(f"seed {21000 + j}", fontsize=7); a.grid(alpha=.2)
        a.tick_params(labelsize=6)
    sr = {lab: sum(x[1] for x in R[lab]) for lab in M}
    handles = [plt.Line2D([0], [0], color=M[lab]["col"], ls=M[lab]["ls"], lw=2,
                          label=f"{lab} ({sr[lab]}/{args.n})") for lab in M]
    fig.legend(handles=handles, loc="upper center", ncol=5, fontsize=11, bbox_to_anchor=(0.5, 1.0))
    fig.suptitle("OOD-distance per seed (5 models) | dashed=PNR(4), dotted=p95 in-domain(1.79)", fontsize=12, y=0.985)
    plt.tight_layout(rect=[0, 0, 1, 0.965]); plt.savefig("analysis/recovery/mip_saves_mse_fails.png", dpi=110)
    print("saved analysis/recovery/mip_saves_mse_fails.png  SR:", sr)


if __name__ == "__main__":
    main()
