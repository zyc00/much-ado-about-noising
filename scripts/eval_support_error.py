"""KEYSTONE evidence: open-loop GROUND-TRUTH action error & predicted-chunk jerk
vs distance-to-clean-training-support, on HELD-OUT DART recovery trajectories.

Why this is the tight test:
  - Test states come from SUCCESSFUL scripted DART recovery demos -> they are OFF the
    clean path but VALID & physically fine (controls the "state is just broken" confound).
  - Each state has a GROUND-TRUTH action (the scripted recovery command) -> real error.
  - dist-to-clean-support = kNN distance of the state to the CLEAN training-state cloud
    (the support MSE_clean/MIP_clean were trained on).
Models: MSE_clean, MIP_clean (the small/clean comparison) + MSE_dart (control: it COVERS
these recovery states, so if its error stays low at the same states, error≠difficulty,
error==off-clean-support i.e. extrapolation).

Prediction if "it's extrapolation": all ~0 near clean support; off support MSE_clean
error&jerk rise most, MIP_clean rises less (better extrapolation), MSE_dart stays low.
No env needed — reads stored obs/actions.

  python scripts/eval_support_error.py
"""
import os
os.environ["MUJOCO_GL"] = "egl"
import argparse
import numpy as np
import torch
import h5py
import sys
sys.path.insert(0, "scripts"); sys.path.insert(0, ".")
from hydra import initialize_config_dir, compose
from omegaconf import OmegaConf
from scipy.spatial import cKDTree
from mip.datasets.robomimic_dataset import make_dataset
from mip.agent import TrainingAgent

OK = ["object", "robot0_eef_pos", "robot0_eef_quat", "robot0_gripper_qpos"]


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


def read_obs_actions(path, nmax=None):
    """Return list of (ov[T,53], actions[T,7]) per demo from stored obs."""
    h = h5py.File(path, "r"); g = "data" if "data" in h else "demos"
    ks = list(h[g].keys())
    if nmax: ks = ks[:nmax]
    out = []
    for k in ks:
        o = h[f"{g}/{k}/obs"]
        ov = np.concatenate([np.asarray(o[key]) for key in OK], axis=1).astype(np.float32)  # (T,53)
        a = np.clip(np.asarray(h[f"{g}/{k}/actions"]), -1, 1).astype(np.float32)
        out.append((ov, a))
    h.close()
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--clean", default="data/tool_hang_full2ins_2000.hdf5")
    ap.add_argument("--test", default="data/tool_hang_darttest_full2ins.hdf5")
    ap.add_argument("--clean_n", type=int, default=40); ap.add_argument("--stride", type=int, default=2)
    ap.add_argument("--H", type=int, default=16)
    args = ap.parse_args()

    SPEC = {
        "MSE_clean": ("logs/full_regression_2000/models/model_latest.pt", "data/tool_hang_full2ins_2000.hdf5", "regression"),
        "MIP_clean": ("logs/full_mip_2000/models/model_latest.pt",        "data/tool_hang_full2ins_2000.hdf5", "mip"),
        "MSE_dart":  ("logs/dart_full2ins_mse/models/model_latest.pt",    "data/tool_hang_dart_full2ins_2000.hdf5", "regression"),
    }
    A = {}
    for nm, (ck, dsp, loss) in SPEC.items():
        cfg, ds, ag = load(ck, dsp, loss)
        A[nm] = dict(ag=ag, ds=ds, no=ds.normalizer["obs"]["state"], na=ds.normalizer["action"])
    dev = cfg.optimization.device; AS = cfg.task.act_steps; start = cfg.task.obs_steps - 1

    # clean support cloud (z-scored kNN)
    clean = read_obs_actions(args.clean, args.clean_n)
    cl = np.concatenate([ov for ov, _ in clean], axis=0)
    mu, sig = cl.mean(0), cl.std(0) + 1e-6
    tree = cKDTree((cl - mu) / sig)
    def dist(v): return float(tree.query((v - mu) / sig)[0])
    print(f"[clean support] {len(cl)} states from {len(clean)} demos", flush=True)

    test = read_obs_actions(args.test)
    print(f"[test] {sum(len(a) for _, a in test)} states from {len(test)} held-out DART demos", flush=True)

    def predict(M, window):
        w = np.stack(window)[None]
        ot = {"state": torch.tensor(M["no"].normalize(w), device=dev, dtype=torch.float32)}
        with torch.no_grad():
            an = M["ag"].sample(act_0=torch.randn((1, args.H, 10), device=dev), obs=ot, use_ema=True)
        return M["ds"].undo_transform_action(M["na"].unnormalize(an.detach().cpu().numpy())[:, start:start + AS])[0]

    def rmse(a, b): return float(np.sqrt(np.mean((a - b) ** 2)))
    def jerk(ch): return float(np.mean(np.linalg.norm(np.diff(ch[:, :6], axis=0), axis=1)))  # intra-chunk jerk (pos+rot)

    rows = []  # (dist, {nm:(err,jerk)})
    for ov, acts in test:
        T = len(acts)
        for t in range(1, T):
            if t % args.stride or t + AS > T: continue
            window = [ov[t - 1], ov[t]]; gt = acts[t:t + AS]
            res = {}
            for nm in A:
                ch = predict(A[nm], window)
                res[nm] = (rmse(ch, gt), jerk(ch))
            rows.append((dist(ov[t]), res))

    dists = np.array([r[0] for r in rows])
    bins = [(0, 1), (1, 2), (2, 4), (4, 8), (8, 1e9)]
    print(f"\nERR/JERK vs dist-to-CLEAN-support  (n_states={len(rows)})  [ground-truth = scripted recovery action]")
    print("  supp-dist   n  | " + " | ".join(f"{nm}: err  jerk" for nm in A))
    for lo, hi in bins:
        m = [i for i in range(len(rows)) if lo <= rows[i][0] < hi]
        if not m: continue
        cells = []
        for nm in A:
            e = np.mean([rows[i][1][nm][0] for i in m]); j = np.mean([rows[i][1][nm][1] for i in m])
            cells.append(f"{e:.3f} {j:.3f}")
        print(f"  [{lo:>2},{hi if hi < 1e8 else 'inf':>3}) {len(m):5d} | " + " | ".join(cells))
    np.savez("analysis/recovery/support_error.npz", dists=dists,
             **{nm: np.array([[rows[i][1][nm][0], rows[i][1][nm][1]] for i in range(len(rows))]) for nm in A})
    print("saved analysis/recovery/support_error.npz")


if __name__ == "__main__":
    main()
