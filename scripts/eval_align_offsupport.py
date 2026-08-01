"""Do FAILURES misalign BECAUSE they leave support DURING the align/carry phase?
Roll out a policy on held-out seeds tracking per step:
  d       = 1-NN distance to clean support (calibrated ruler)
  phase   = oracle phase: 0 pre-lift, 1 lifted/carrying+aligning, 2 aligned-near-stand
  axy     = alignment error: xy dist of frame_mount_site to stand-wall center
  grasped = still holding the frame
Per-episode analysis (fail vs success):
  - escape onset step (first d>2.45 that never returns below 1.79) and ITS PHASE
  - best alignment achieved (min axy while lifted); step of best alignment
  - temporal order: escape onset vs best-alignment step
  - d during the align phase (phase==1) before any alignment is reached
If failures' escape onset lies in phase 1 (while carrying/aligning, before aligned) and
their best-ever alignment stays poor, the chain 'off-support during align -> misalign ->
fail' is confirmed; if they align fine on-support and fail later, it is refuted."""
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


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", default="logs/full_regression_2000/models/model_latest.pt")
    ap.add_argument("--dataset", default="data/tool_hang_full2ins_2000.hdf5")
    ap.add_argument("--loss", default="regression")
    ap.add_argument("--seed_lo", type=int, default=21000); ap.add_argument("--seed_hi", type=int, default=21080)
    ap.add_argument("--H", type=int, default=16); ap.add_argument("--settle", type=int, default=10)
    ap.add_argument("--max_steps", type=int, default=700)
    ap.add_argument("--out", default="analysis/recovery/align_offsupport_mse.npz")
    ap.add_argument("--norm_pt", default=None, help="exact normalizer override (e.g. DART-6k)")
    args = ap.parse_args()
    cfgdir = os.path.abspath("examples/configs")
    with initialize_config_dir(version_base=None, config_dir=cfgdir):
        cfg = compose(config_name="main", overrides=["task=tool_hang_ph_state_delta_legacy",
            f"+task.dataset_path={os.path.abspath(args.dataset)}", "network=chiunet",
            f"optimization.loss_type={args.loss}", "optimization.auto_resume=false", "log.wandb_mode=disabled"])
    OmegaConf.set_struct(cfg, False); cfg.task.obs_dim = 53
    dev = cfg.optimization.device; AS = cfg.task.act_steps; start = cfg.task.obs_steps - 1
    ds = make_dataset(cfg.task)
    ag = TrainingAgent(cfg); ag.load(args.ckpt, load_optimizer=False); ag.eval()
    if args.norm_pt:
        _n = torch.load(args.norm_pt, weights_only=False)
        no = _n["obs"]["state"]; na = _n["action"]
        print(f"[norm] exact normalizer from {args.norm_pt}")
    else:
        no = ds.normalizer["obs"]["state"]; na = ds.normalizer["action"]

    # clean-support cloud from clean training demos' stored obs (same ruler family as before)
    h = h5py.File(args.dataset, "r"); g = "data"
    cl = np.concatenate([np.concatenate([np.asarray(h[f"{g}/demo_{i}/obs"][k]) for k in OK], axis=1)
                         for i in range(40)], 0).astype(np.float32)
    h.close()
    mu, sig = cl.mean(0), cl.std(0) + 1e-6
    tree = cKDTree((cl - mu) / sig)
    def dist(v): return float(tree.query((v - mu) / sig)[0])

    env = robosuite.make("ToolHang", horizon=4000, **ENV_KWARGS)
    def ov(o): return np.concatenate([o[KM.get(k, k)] for k in OK]).astype(np.float32)

    def phase_align():
        sim = env.sim  # re-read every call: env.reset() rebuilds sim
        fz = sim.data.site_xpos[sim.model.site_name2id("frame_mount_site")]
        hc = np.mean([sim.data.geom_xpos[sim.model.geom_name2id(f"stand_wall{i}")] for i in range(4)], axis=0)
        axy = float(np.linalg.norm((fz - hc)[:2]))
        lifted = fz[2] > 0.86; near = axy < 0.05
        ph = 2 if (lifted and near) else (1 if lifted else 0)
        return ph, axy, float(fz[2])

    def predict(hist):
        w = np.stack(hist[-2:])[None]
        ot = {"state": torch.tensor(no.normalize(w), device=dev, dtype=torch.float32)}
        with torch.no_grad():
            an = ag.sample(act_0=torch.randn((1, args.H, 10), device=dev), obs=ot, use_ema=True)
        return ds.undo_transform_action(na.unnormalize(an.detach().cpu().numpy())[:, start:start + AS])[0]

    episodes = []
    for sd in range(args.seed_lo, args.seed_hi):
        np.random.seed(sd); env.reset()
        arm = env.robots[0].composite_controller.part_controllers["right"]; arm.update(); arm.reset_goal()
        for _ in range(args.settle):
            env.step(np.zeros(7))
        o = env._get_observations(force_update=True); hist = [ov(o), ov(o)]
        D, PH, AXY = [], [], []
        steps = 0; asm = False
        while steps < args.max_steps and not asm:
            for a in predict(hist):
                o, _, _, _ = env.step(a); steps += 1; v = ov(o); hist.append(v)
                ph, axy, _ = phase_align()
                D.append(dist(v)); PH.append(ph); AXY.append(axy)
                if env._check_frame_assembled(): asm = True; break
                if steps >= args.max_steps: break
        episodes.append((sd, asm, np.array(D), np.array(PH, dtype=np.int8), np.array(AXY)))
        print(f"  seed {sd}: {'OK ' if asm else 'FAIL'} steps={len(D)} maxd={max(D):.1f} minaxy={min(AXY):.3f}", flush=True)

    np.savez(args.out,
             seeds=np.array([e[0] for e in episodes]), asm=np.array([e[1] for e in episodes]),
             **{f"D{i}": e[2] for i, e in enumerate(episodes)},
             **{f"PH{i}": e[3] for i, e in enumerate(episodes)},
             **{f"AXY{i}": e[4] for i, e in enumerate(episodes)})

    # ---------- analysis ----------
    def escape_onset(D):
        # first index where d>2.45 and d never returns below 1.79 afterwards
        below = D < 1.79
        last_ok = -1
        for t in range(len(D)):
            if below[t]: last_ok = t
        for t in range(len(D)):
            if D[t] > 2.45 and t > last_ok:
                return t
        return None

    fails = [e for e in episodes if not e[1]]; succs = [e for e in episodes if e[1]]
    print(f"\n===== {len(fails)} FAILURES / {len(succs)} successes =====")
    ph_at_onset = {0: 0, 1: 0, 2: 0, None: 0}
    onset_before_align = 0; aligned_ever_f = 0
    for sd, asm, D, PH, AXY in fails:
        t0 = escape_onset(D)
        ph = PH[t0] if t0 is not None else None
        ph_at_onset[ph if ph is None else int(ph)] += 1
        aligned = np.where(PH == 2)[0]
        aligned_ever_f += int(len(aligned) > 0)
        if t0 is not None and (len(aligned) == 0 or t0 < aligned[0]):
            onset_before_align += 1
    print(f"escape-onset phase distribution (failures): {ph_at_onset}  (0=pre-lift,1=carry/align,2=aligned)")
    print(f"failures whose escape onset is BEFORE ever reaching aligned: {onset_before_align}/{len(fails)}")
    print(f"failures that EVER reach aligned (phase2): {aligned_ever_f}/{len(fails)}")
    ba_f = [min(AXY[PH >= 1]) if (PH >= 1).any() else np.inf for _, _, D, PH, AXY in fails]
    ba_s = [min(AXY[PH >= 1]) if (PH >= 1).any() else np.inf for _, _, D, PH, AXY in succs]
    print(f"best alignment error while lifted (m): failures median {np.median(ba_f):.3f} vs successes {np.median(ba_s):.3f}")
    # off-support during align phase BEFORE first aligned moment
    def d_align_pre(D, PH):
        aligned = np.where(PH == 2)[0]; end = aligned[0] if len(aligned) else len(D)
        m = (PH[:end] == 1)
        return D[:end][m].mean() if m.any() else np.nan
    da_f = [d_align_pre(D, PH) for _, _, D, PH, _ in fails]
    da_s = [d_align_pre(D, PH) for _, _, D, PH, _ in succs]
    print(f"mean d during carry/align (pre-aligned): failures {np.nanmedian(da_f):.2f} vs successes {np.nanmedian(da_s):.2f}")
    print(f"saved {args.out}")


if __name__ == "__main__":
    main()
