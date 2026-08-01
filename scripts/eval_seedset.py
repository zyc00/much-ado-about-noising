"""Isolate the official-vs-standalone eval gap. Reproduces the EXACT placement
seeding the official wrapper uses (np.random.seed(s); env.reset(); settle), so
seeds 0..19 == official eval's first batch (== training placements, since the
scripted collector used seeds 0..19999). Lets you toggle:
  --seed_lo/--seed_hi : placement seed range (train 0.. vs held-out 21000..)
  --H                 : act_0 horizon length (10 = training/official, 16 = old standalone bug)
Success = _check_frame_assembled (== official mean_assembled).

  MUJOCO_GL=egl python scripts/eval_seedset.py --ckpt ... --dataset ... --loss regression \
     --seed_lo 0 --seed_hi 50 --H 10 --tag train_H10
"""
import os
os.environ["MUJOCO_GL"] = "egl"
import argparse
import numpy as np
import torch
import robosuite
import sys
sys.path.insert(0, "scripts"); sys.path.insert(0, ".")
from collect_tool_hang_demos import ENV_KWARGS
from hydra import initialize_config_dir, compose
from omegaconf import OmegaConf
from mip.datasets.robomimic_dataset import make_dataset
from mip.agent import TrainingAgent

OK = ["object", "robot0_eef_pos", "robot0_eef_quat", "robot0_gripper_qpos"]
KM = {"object": "object-state"}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--dataset", required=True)
    ap.add_argument("--loss", default="regression")
    ap.add_argument("--seed_lo", type=int, default=0)
    ap.add_argument("--seed_hi", type=int, default=50)
    ap.add_argument("--H", type=int, default=10)
    ap.add_argument("--settle", type=int, default=10)
    ap.add_argument("--max_steps", type=int, default=700)
    ap.add_argument("--norm_pt", default=None, help="override normalizer with a saved torch dict (exact normalizer from a remote-trained dataset)")
    ap.add_argument("--tag", default="")
    ap.add_argument("--abs", action="store_true", help="use absolute-input OSC controller env")
    ap.add_argument("--track_dist", action="store_true", help="track support distance (40-anchor cloud) per step; report cross2/cross4 and SR conditioned on crossing")
    ap.add_argument("--net", default="", help="extra hydra overrides, semicolon-separated, e.g. 'network.model_dim=256;network.dim_mult=[1,2,4]'")
    args = ap.parse_args()

    extra = [o for o in args.net.split(";") if o]
    cfgdir = os.path.abspath("examples/configs")
    with initialize_config_dir(version_base=None, config_dir=cfgdir):
        cfg = compose(config_name="main", overrides=[
            "task=tool_hang_ph_state_delta_legacy", f"+task.dataset_path={os.path.abspath(args.dataset)}",
            "network=chiunet", f"optimization.loss_type={args.loss}",
            "optimization.auto_resume=false", "log.wandb_mode=disabled"] + extra)
    OmegaConf.set_struct(cfg, False); cfg.task.obs_dim = 53
    dev = cfg.optimization.device
    AS = cfg.task.act_steps; start = cfg.task.obs_steps - 1
    ds = make_dataset(cfg.task)
    agent = TrainingAgent(cfg); agent.load(args.ckpt, load_optimizer=False); agent.eval()
    if args.norm_pt:
        _nrm = torch.load(args.norm_pt, weights_only=False)
        no = _nrm["obs"]["state"]; na = _nrm["action"]
        print(f"[norm] using exact normalizer from {args.norm_pt}")
    else:
        no = ds.normalizer["obs"]["state"]; na = ds.normalizer["action"]
    import copy as _copy
    _ek = _copy.deepcopy(ENV_KWARGS)
    if args.abs:
        _ek["controller_configs"]["body_parts"]["right"]["input_type"] = "absolute"
    env = robosuite.make("ToolHang", horizon=4000, **_ek)

    def ov(o):
        return np.concatenate([o[KM.get(k, k)] for k in OK]).astype(np.float32)

    tree = None
    if args.track_dist:
        import h5py
        from scipy.spatial import cKDTree
        hf = h5py.File(args.dataset, "r"); g = "data" if "data" in hf else "demos"
        cl = []
        for i in range(40):
            oo = hf[f"{g}/demo_{i}/obs"]
            cl.append(np.concatenate([np.asarray(oo[k]) for k in OK], axis=1).astype(np.float32))
        hf.close()
        cl = np.concatenate(cl, 0)
        mu_c, sig_c = cl.mean(0), cl.std(0) + 1e-6
        tree = cKDTree((cl - mu_c) / sig_c)

    def chunk(hist):
        w = np.stack(hist[-2:])[None]
        ot = {"state": torch.tensor(no.normalize(w), device=dev, dtype=torch.float32)}
        with torch.no_grad():
            an = agent.sample(act_0=torch.randn((1, args.H, 10), device=dev), obs=ot, use_ema=True)
        return ds.undo_transform_action(na.unnormalize(an.detach().cpu().numpy())[:, start:start + AS])[0]

    succ = 0; N = 0; ep = []
    for sd in range(args.seed_lo, args.seed_hi):
        # EXACT official wrapper placement seeding
        np.random.seed(sd); env.reset()
        arm = env.robots[0].composite_controller.part_controllers["right"]; arm.update(); arm.reset_goal()
        for _ in range(args.settle):
            if args.abs:
                # absolute mode: zeros would command the origin; HOLD current pose instead
                import robosuite.utils.transform_utils as _T
                _o = env._get_observations(force_update=True)
                _hold = np.concatenate([_o["robot0_eef_pos"],
                                        _T.quat2axisangle(_o["robot0_eef_quat"]), [0.0]])
                env.step(_hold.astype(np.float32))
            else:
                env.step(np.zeros(7))
        o = env._get_observations(force_update=True); hist = [ov(o), ov(o)]
        steps = 0; asm = False; maxd = 0.0
        dser = []
        while steps < args.max_steps and not asm:
            for a in chunk(hist):
                o, _, _, _ = env.step(a); steps += 1; hist.append(ov(o))
                if tree is not None:
                    d, _ = tree.query((hist[-1] - mu_c) / sig_c)
                    maxd = max(maxd, float(d)); dser.append(float(d))
                if env._check_frame_assembled():
                    asm = True; break
                if steps >= args.max_steps:
                    break
        succ += int(asm); N += 1
        fc4 = next((i for i, dv in enumerate(dser) if dv >= 4.0), -1)
        dd = np.diff(dser) if len(dser) > 1 else np.array([0.0])
        db = np.array(dser[:-1]) if len(dser) > 1 else np.array([0.0])
        drift_lo = float(dd[db < 2.0].mean()) if (db < 2.0).any() else float("nan")
        drift_hi = float(dd[(db >= 2.0) & (db < 4.0)].mean()) if ((db >= 2.0) & (db < 4.0)).any() else float("nan")
        ep.append((int(asm), maxd, int(maxd >= 2.0), int(maxd >= 4.0), fc4, drift_lo, drift_hi))
    extra = ""
    if tree is not None:
        ep = np.array(ep); c2 = ep[:, 2].astype(bool); c4 = ep[:, 3].astype(bool)
        sr_c4 = 100 * ep[c4, 0].mean() if c4.any() else float("nan")
        sr_n4 = 100 * ep[~c4, 0].mean() if (~c4).any() else float("nan")
        fct = ep[ep[:, 4] >= 0, 4]
        extra = (f" | cross2={c2.mean():.2f} cross4={c4.mean():.2f} "
                 f"SR|cross4={sr_c4:.0f} SR|stay={sr_n4:.0f} "
                 f"maxd_p50/90/95={np.percentile(ep[:,1],50):.2f}/{np.percentile(ep[:,1],90):.2f}/{np.percentile(ep[:,1],95):.2f} "
                 f"firstcross4_p50={np.median(fct) if len(fct) else float('nan'):.0f} "
                 f"drift_d<2={np.nanmean(ep[:,5]):+.4f} drift_[2,4)={np.nanmean(ep[:,6]):+.4f}")
    print(f"SEEDSET {args.tag} seeds[{args.seed_lo},{args.seed_hi}) H={args.H} "
          f"assembled={succ}/{N} = {100*succ/N:.1f}%{extra}")


if __name__ == "__main__":
    main()
