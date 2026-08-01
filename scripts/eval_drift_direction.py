"""WHY does MIP drift in the RIGHT direction off-support? Decompose each policy's
prediction drift against the GROUND-TRUTH corrective change.
For each held-out test state s with clean 1-NN neighbor s0 (demo i, time t0):
  GT change      da* = a*(s) - a*(s0)      (recovery action at s minus expert action at s0)
  policy drift   dA  = a_hat(s) - a_hat(s0)
Metrics per distance bin and per PERTURBATION TYPE (eef-pose-dominant vs object-dominant
deviation): cos(dA, da*), parallel/orthogonal components, residual after projection.
Tests the delta-action isomorphism hypothesis: the denoising head's 'subtract the
residual' operator matches the corrective structure of recovery actions -- and should
work mainly when the deviation lives in eef-pose dims (which have action-space mirrors),
not object dims. No env."""
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
from mip.samplers import regression_sampler, mip_step1_only_sampler

OK = ["object", "robot0_eef_pos", "robot0_eef_quat", "robot0_gripper_qpos"]
EEF = slice(44, 51)   # eef_pos(3)+eef_quat(4) within the 53-dim obs


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


def read(path, nmax=None, with_actions=True):
    h = h5py.File(path, "r"); g = "data" if "data" in h else "demos"; ks = list(h[g].keys())
    if nmax: ks = ks[:nmax]
    out = []
    for k in ks:
        o = h[f"{g}/{k}/obs"]
        ov = np.concatenate([np.asarray(o[key]) for key in OK], axis=1).astype(np.float32)
        a = np.clip(np.asarray(h[f"{g}/{k}/actions"]), -1, 1).astype(np.float32) if with_actions else None
        out.append((ov, a))
    h.close(); return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--clean", default="data/tool_hang_full2ins_2000.hdf5")
    ap.add_argument("--test", default="data/dart_test_huge_full2ins.hdf5")
    ap.add_argument("--clean_n", type=int, default=40); ap.add_argument("--stride", type=int, default=4)
    ap.add_argument("--H", type=int, default=16)
    args = ap.parse_args()
    cfg, ds, mse = load("logs/full_regression_2000/models/model_latest.pt", args.clean, "regression")
    _, _, mip = load("logs/full_mip_2000/models/model_latest.pt", args.clean, "mip")
    dev = cfg.optimization.device; AS = cfg.task.act_steps; start = cfg.task.obs_steps - 1
    no = ds.normalizer["obs"]["state"]; na = ds.normalizer["action"]; optim = cfg.optimization

    clean = read(args.clean, args.clean_n)
    cl = np.concatenate([ov for ov, _ in clean], 0)
    owner = np.concatenate([[(i, t) for t in range(len(ov))] for i, (ov, _) in enumerate(clean)], 0)
    mu, sig = cl.mean(0), cl.std(0) + 1e-6
    tree = cKDTree((cl - mu) / sig)

    def window(tr, t): return [tr[max(t - 1, 0)], tr[t]]

    def predict(ag, sampler, win):
        w = no.normalize(np.stack(win)[None])
        x = torch.tensor(w, device=dev, dtype=torch.float32)
        with torch.no_grad():
            with ag._inference_mode():
                an = sampler(optim, ag.flow_map_ema, ag.encoder_ema, torch.zeros((1, args.H, 10), device=dev), {"state": x})
        return ds.undo_transform_action(na.unnormalize(an.detach().cpu().numpy())[:, start:start + AS])[0][0, :6]

    nb = {}
    def neighbor(idx):
        if idx not in nb:
            i, t0 = map(int, owner[idx])
            ov, acts = clean[i]
            t_act = min(t0, len(acts) - 1)
            nb[idx] = (predict(mse, regression_sampler, window(ov, t0)),
                       predict(mip, mip_step1_only_sampler, window(ov, t0)),
                       acts[t_act, :6])
        return nb[idx]

    test = read(args.test)
    rows = []
    for ov, acts in test:
        T = len(acts)
        for t in range(1, T - AS):
            if t % args.stride: continue
            z = (ov[t] - mu) / sig
            d, idx = tree.query(z)
            aM = predict(mse, regression_sampler, window(ov, t))
            aP = predict(mip, mip_step1_only_sampler, window(ov, t))
            aM0, aP0, a0 = neighbor(int(idx))
            gt = acts[t, :6]
            da_gt = gt - a0
            dz = z - (cl[int(idx)] - mu) / sig
            eef_share = float(np.sum(dz[EEF] ** 2) / max(np.sum(dz ** 2), 1e-9))
            rows.append((float(d), eef_share, da_gt, aM - aM0, aP - aP0))
    D = np.array([r[0] for r in rows]); ES = np.array([r[1] for r in rows])
    GT = np.stack([r[2] for r in rows]); DM = np.stack([r[3] for r in rows]); DP = np.stack([r[4] for r in rows])
    np.savez("analysis/recovery/drift_direction.npz", d=D, eef=ES, gt=GT, dM=DM, dP=DP)

    def cosv(A, B):
        na_ = np.linalg.norm(A, axis=1); nb_ = np.linalg.norm(B, axis=1)
        ok = (na_ > 0.05) & (nb_ > 0.05)
        return (np.sum(A * B, axis=1) / (na_ * nb_ + 1e-9)), ok

    print(f"{len(rows)} pairs.  cos(policy drift, GT corrective change)  |GT change| floor 0.05")
    print(f"{'bin':8} {'N':>6} {'|da*|':>6} | {'cos MSE':>8} {'cos MIP':>8} | {'res_MSE':>8} {'res_MIP':>8}")
    for lo, hi in [(1, 2), (2, 3), (3, 4), (4, 6)]:
        m = (D >= lo) & (D < hi)
        cM, okM = cosv(DM[m], GT[m]); cP, okP = cosv(DP[m], GT[m])
        rM = np.linalg.norm(DM[m] - GT[m], axis=1).mean(); rP = np.linalg.norm(DP[m] - GT[m], axis=1).mean()
        print(f"[{lo},{hi})  {int(m.sum()):>6} {np.linalg.norm(GT[m],axis=1).mean():>6.2f} | "
              f"{cM[okM].mean():>8.2f} {cP[okP].mean():>8.2f} | {rM:>8.3f} {rP:>8.3f}")
    print("\nsplit by perturbation type (off-support d in [2,4)):")
    off = (D >= 2) & (D < 4)
    for name, msk in [("eef-dominant (share>0.5)", off & (ES > 0.5)), ("object-dominant (share<0.2)", off & (ES < 0.2))]:
        if msk.sum() < 20:
            print(f"  {name}: N={int(msk.sum())} -- few"); continue
        cM, okM = cosv(DM[msk], GT[msk]); cP, okP = cosv(DP[msk], GT[msk])
        rM = np.linalg.norm(DM[msk] - GT[msk], axis=1).mean(); rP = np.linalg.norm(DP[msk] - GT[msk], axis=1).mean()
        print(f"  {name}: N={int(msk.sum())}  cosMSE {cM[okM].mean():.2f}  cosMIP {cP[okP].mean():.2f}  "
              f"resMSE {rM:.3f}  resMIP {rP:.3f}")
    print("saved analysis/recovery/drift_direction.npz")


if __name__ == "__main__":
    main()
