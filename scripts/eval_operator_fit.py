"""What IS the extrapolation rule, functionally? Fit linear operators on off-support pairs:
  GT:   da*  ~ G_gt  . dz     (the data-side extension law; scripted recovery)
  MSE:  daM  ~ G_mse . dz     (the learned extrapolation of the regressor)
  MIP:  daP  ~ G_mip . dz     (the learned extrapolation of the denoise-cotrained net)
  ON:   tangent 'script-advance' operator G_on from consecutive CLEAN states
dz = z-scored 53-dim state deviation from the clean 1-NN; da = executed 6-dim action diff.
Reports: linearity R^2 (5-fold), operator similarity via per-pair prediction cosine,
negative-feedback structure of the eef-pose block, and whether MSE's residual operator
matches the tangent operator (script-continuation hypothesis). No env."""
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
POS = slice(44, 47)   # eef_pos dims in obs


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


def read(path, nmax=None):
    h = h5py.File(path, "r"); g = "data" if "data" in h else "demos"; ks = list(h[g].keys())
    if nmax: ks = ks[:nmax]
    out = []
    for k in ks:
        o = h[f"{g}/{k}/obs"]
        ov = np.concatenate([np.asarray(o[key]) for key in OK], axis=1).astype(np.float32)
        a = np.clip(np.asarray(h[f"{g}/{k}/actions"]), -1, 1).astype(np.float32)
        out.append((ov, a))
    h.close(); return out


def ridge(X, Y, lam=1e-2):
    # Y (N,6), X (N,53) -> G (6,53)
    A = X.T @ X + lam * len(X) * np.eye(X.shape[1])
    return np.linalg.solve(A, X.T @ Y).T


def r2_cv(X, Y, lam=1e-2, k=5, seed=0):
    rng = np.random.RandomState(seed); idx = rng.permutation(len(X))
    fold = np.array_split(idx, k); ss_res = 0.0; ss_tot = 0.0
    for i in range(k):
        te = fold[i]; tr = np.concatenate([fold[j] for j in range(k) if j != i])
        G = ridge(X[tr], Y[tr], lam)
        P = X[te] @ G.T
        ss_res += np.sum((Y[te] - P) ** 2); ss_tot += np.sum((Y[te] - Y[tr].mean(0)) ** 2)
    return 1 - ss_res / ss_tot


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--clean", default="data/tool_hang_full2ins_2000.hdf5")
    ap.add_argument("--test", default="data/dart_test_huge_full2ins.hdf5")
    ap.add_argument("--clean_n", type=int, default=40); ap.add_argument("--stride", type=int, default=4)
    ap.add_argument("--H", type=int, default=16)
    ap.add_argument("--lo", type=float, default=2.0); ap.add_argument("--hi", type=float, default=4.0)
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

    nbc = {}
    def neighbor(idx):
        if idx not in nbc:
            i, t0 = map(int, owner[idx])
            ov, acts = clean[i]
            nbc[idx] = (predict(mse, regression_sampler, window(ov, t0)),
                        predict(mip, mip_step1_only_sampler, window(ov, t0)),
                        acts[min(t0, len(acts) - 1), :6])
        return nbc[idx]

    test = read(args.test)
    DZ, GT, DM, DP, DD = [], [], [], [], []
    for ov, acts in test:
        T = len(acts)
        for t in range(1, T - AS):
            if t % args.stride: continue
            z = (ov[t] - mu) / sig
            d, idx = tree.query(z)
            if not (args.lo <= d < args.hi): continue
            aM = predict(mse, regression_sampler, window(ov, t))
            aP = predict(mip, mip_step1_only_sampler, window(ov, t))
            aM0, aP0, a0 = neighbor(int(idx))
            DZ.append(z - (cl[int(idx)] - mu) / sig)
            GT.append(acts[t, :6] - a0); DM.append(aM - aM0); DP.append(aP - aP0); DD.append(d)
    DZ = np.stack(DZ); GT = np.stack(GT); DM = np.stack(DM); DP = np.stack(DP); DD = np.array(DD)
    print(f"off-support pairs in [{args.lo},{args.hi}): {len(DZ)}")

    # tangent 'script-advance' operator from clean consecutive frames
    XT, YT = [], []
    for ov, acts in clean:
        z = (ov - mu) / sig
        for t in range(1, len(acts) - 1, 3):
            XT.append(z[t + 1] - z[t]); YT.append(acts[t + 1, :6] - acts[t, :6])
    XT = np.stack(XT); YT = np.stack(YT)

    G_gt = ridge(DZ, GT); G_m = ridge(DZ, DM); G_p = ridge(DZ, DP); G_on = ridge(XT, YT)
    print("\n== linearity (5-fold R^2 on off-support pairs) ==")
    print(f"  GT  da* ~ G dz : R2 = {r2_cv(DZ, GT):.3f}   (is the data-side extension a linear law?)")
    print(f"  MSE drift      : R2 = {r2_cv(DZ, DM):.3f}")
    print(f"  MIP drift      : R2 = {r2_cv(DZ, DP):.3f}")

    def pred_cos(Ga, Gb):
        Pa = DZ @ Ga.T; Pb = DZ @ Gb.T
        c = np.sum(Pa * Pb, 1) / (np.linalg.norm(Pa, 1e-9 + 1) if False else (np.linalg.norm(Pa, axis=1) * np.linalg.norm(Pb, axis=1) + 1e-9))
        return c.mean()
    print("\n== operator similarity (mean per-pair prediction cosine on off-support dz) ==")
    print(f"  cos(G_MIP dz, G_GT dz) = {pred_cos(G_p, G_gt):.2f}")
    print(f"  cos(G_MSE dz, G_GT dz) = {pred_cos(G_m, G_gt):.2f}")
    print(f"  cos(G_MSE dz, G_ON dz) = {pred_cos(G_m, G_on):.2f}   (script-continuation hypothesis)")
    print(f"  cos(G_MIP dz, G_ON dz) = {pred_cos(G_p, G_on):.2f}")
    resid = G_m - G_gt
    print(f"  cos((G_MSE - G_GT) dz, G_ON dz) = {pred_cos(resid, G_on):.2f}   (is MSE's ERROR = tangent rule?)")

    # negative-feedback structure: eef_pos block (obs pos dims -> action pos dims)
    def posblock(G):
        B = G[0:3, POS] * sig[POS][None, :]   # de-zscore input for physical units
        return B
    for name, G in [("GT", G_gt), ("MIP", G_p), ("MSE", G_m)]:
        B = posblock(G)
        w = np.linalg.eigvals((B + B.T) / 2)
        print(f"  {name} eef-pos block: diag {np.round(np.diag(B),3)}  sym-eig {np.round(np.real(w),3)}  "
              f"({'NEGATIVE feedback' if np.all(np.real(w) < 0) else 'not neg-definite'})")

    # spectrum / rank
    for name, G in [("G_GT", G_gt), ("G_MIP", G_p), ("G_MSE", G_m), ("G_ON", G_on)]:
        s = np.linalg.svd(G, compute_uv=False)
        print(f"  {name}: top5 sv {np.round(s[:5],3)}  eff-rank(90%) {int(np.searchsorted(np.cumsum(s**2)/np.sum(s**2), 0.9)+1)}")
    np.savez("analysis/recovery/operator_fit.npz", DZ=DZ, GT=GT, DM=DM, DP=DP, d=DD,
             G_gt=G_gt, G_m=G_m, G_p=G_p, G_on=G_on)
    print("saved analysis/recovery/operator_fit.npz")


if __name__ == "__main__":
    main()
