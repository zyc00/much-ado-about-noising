"""EXP 2: VIEW-INVARIANT SUBSPACE TEST.
For a model, extract phi0 = phi(t=0, 0, s) and phitau = phi(tau, 0, s) on clean states.
Subspaces in feature space:
  U_servo : top-2 left singular dirs of cross-cov of dphi (off-support pairs) with G_GT dz
  U_bad   : orthonormalized {W e_bad1, W e_rotamp} — readout-weight combos that produce
            MSE's bad pose direction / rotation-amplifier output direction
Measure:
  1. view-difference ratios ||P(phi0-phitau)|| / ||P(phi0-mean)|| in each subspace
  2. CKA between views restricted to each subspace
  3. clean-action decodability (ridge readout R2) from P_servo phi vs P_bad phi
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
POS = slice(44, 47)
H = 16


def read(path, nmax=None):
    h = h5py.File(path, "r"); g = "data" if "data" in h else "demos"
    n = min(nmax, len(h[g])) if nmax else len(h[g])
    out = []
    for i in range(n):
        o = h[f"{g}/demo_{i}/obs"]
        ov = np.concatenate([np.asarray(o[key]) for key in OK], axis=1).astype(np.float32)
        a = np.clip(np.asarray(h[f"{g}/demo_{i}/actions"]), -1, 1).astype(np.float32)
        out.append((ov, a))
    h.close(); return out


def cka(X, Y):
    Xc = X - X.mean(0); Yc = Y - Y.mean(0)
    return float(np.linalg.norm(Xc.T @ Yc, "fro") ** 2 /
                 (np.linalg.norm(Xc.T @ Xc, "fro") * np.linalg.norm(Yc.T @ Yc, "fro") + 1e-12))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", required=True); ap.add_argument("--loss", required=True)
    ap.add_argument("--random_init", action="store_true")
    ap.add_argument("--tag", default="")
    ap.add_argument("--clean", default="data/tool_hang_full2ins_2000.hdf5")
    ap.add_argument("--test", default="data/tool_hang_puredart_full2ins_2000.hdf5")
    ap.add_argument("--ref", default="analysis/recovery/badmode_ref.npz")
    ap.add_argument("--train_demos", type=int, default=100); ap.add_argument("--test_n", type=int, default=300)
    args = ap.parse_args()
    cfgdir = os.path.abspath("examples/configs")
    with initialize_config_dir(version_base=None, config_dir=cfgdir):
        cfg = compose(config_name="main", overrides=["task=tool_hang_ph_state_delta_legacy",
            f"+task.dataset_path={os.path.abspath(args.clean)}", "network=chiunet",
            f"optimization.loss_type={args.loss}", "optimization.auto_resume=false", "log.wandb_mode=disabled"])
    OmegaConf.set_struct(cfg, False); cfg.task.obs_dim = 53
    ds = make_dataset(cfg.task)
    if args.random_init:
        torch.manual_seed(0)
    ag = TrainingAgent(cfg)
    if not args.random_init:
        ag.load(args.ckpt, load_optimizer=False)
    ag.eval()
    dev = cfg.optimization.device; TAU = float(cfg.optimization.t_two_step)
    no = ds.normalizer["obs"]["state"]

    feats = {}
    net = None
    for name, m in ag.flow_map_ema.named_modules():
        if name.endswith("final_conv"):
            net = m
    list(net.children())[-1].register_forward_pre_hook(
        lambda mod, inp: feats.__setitem__("x", inp[0].detach()))

    def phi(win, tval):
        x = torch.tensor(no.normalize(np.stack(win)[None]), device=dev, dtype=torch.float32)
        with torch.no_grad():
            with ag._inference_mode():
                emb = ag.encoder_ema({"state": x}, None)
                ag.flow_map_ema.get_velocity(torch.full((1,), tval, device=dev),
                                             torch.zeros((1, H, 10), device=dev), emb)
        return feats["x"].reshape(-1).cpu().numpy()

    clean = read(args.clean, max(args.train_demos, 40))
    wins, Y = [], []
    for i, (ov, acts) in enumerate(clean[:args.train_demos]):
        for t in range(1, len(acts) - H, 8):
            wins.append([ov[t - 1], ov[t]]); Y.append(acts[t, :6])
    Y = np.stack(Y).astype(np.float64)
    P0 = np.stack([phi(w, 0.0) for w in wins]).astype(np.float64)
    PT = np.stack([phi(w, TAU) for w in wins]).astype(np.float64)
    mu0 = P0.mean(0)

    # ridge head W (t0 view) for bad-mode feature dirs
    Xc = P0 - mu0; Yc = Y - Y.mean(0)
    W = np.linalg.solve(Xc.T @ Xc + 1e-3 * len(Xc) * np.eye(Xc.shape[1]), Xc.T @ Yc)

    # ---- servo subspace from off-support pairs (t0 view) ----
    anchors40 = clean[:40]
    cl = np.concatenate([ov for ov, _ in anchors40], 0)
    owner = np.concatenate([[(i, t) for t in range(len(ov))] for i, (ov, _) in enumerate(anchors40)], 0)
    mu, sig = cl.mean(0), cl.std(0) + 1e-6
    tree = cKDTree((cl - mu) / sig)
    test = read(args.test, args.test_n)
    F_s, F_0, DZ, GT = [], [], [], []
    f0c = {}
    for ov, acts in test:
        for t in range(1, len(acts) - H, 4):
            z = (ov[t] - mu) / sig
            d, idx = tree.query(z)
            if not (2.0 <= d < 4.0): continue
            i0, t0 = map(int, owner[int(idx)])
            if int(idx) not in f0c:
                f0c[int(idx)] = phi([anchors40[i0][0][max(t0 - 1, 0)], anchors40[i0][0][t0]], 0.0)
            F_0.append(f0c[int(idx)]); F_s.append(phi([ov[t - 1], ov[t]], 0.0))
            DZ.append(z - (cl[int(idx)] - mu) / sig)
            GT.append(acts[t, :6] - anchors40[i0][1][min(t0, len(anchors40[i0][1]) - 1), :6])
    F_s = np.stack(F_s); F_0 = np.stack(F_0); DZ = np.stack(DZ); GT = np.stack(GT)

    def ridge6(X, Yv, l=1e-2):
        Aa = X.T @ X + l * len(X) * np.eye(X.shape[1]); return np.linalg.solve(Aa, X.T @ Yv).T
    G_gt = ridge6(DZ, GT)
    M = (F_s - F_0).T @ (DZ @ G_gt.T)
    U_servo = np.linalg.svd(M, full_matrices=False)[0][:, :2]

    # ---- bad-mode feature dirs ----
    ref = np.load(args.ref)
    def posblk(Gx):
        Bx = Gx[0:3, POS] * ref["sig"][POS][None, :]
        return (Bx + Bx.T) / 2
    wm, Vm = np.linalg.eigh(posblk(ref["G_m"]))
    bad1 = Vm[:, np.argmax(wm)]
    rout = np.linalg.svd(ref["G_m"][3:6, :] * ref["sig"][None, :])[0][:, 0]
    e1 = np.zeros(6); e1[0:3] = bad1
    e2 = np.zeros(6); e2[3:6] = rout
    U_bad = np.linalg.qr(np.stack([W @ e1, W @ e2], 1))[0]

    # ---- measurements ----
    D = P0 - PT
    for name, Uk in [("servo", U_servo), ("bad", U_bad)]:
        num = np.linalg.norm(D @ Uk, "fro")
        den = np.linalg.norm((P0 - mu0) @ Uk, "fro") + 1e-12
        c = cka(P0 @ Uk, PT @ Uk)
        # decodability from the 2-d projection alone
        Zp = (P0 - mu0) @ Uk
        Wp = np.linalg.solve(Zp.T @ Zp + 1e-3 * len(Zp) * np.eye(2), Zp.T @ Yc)
        r2 = 1 - np.sum((Zp @ Wp - Yc) ** 2) / np.sum(Yc ** 2)
        print(f"VIEWINV {args.tag} sub={name} viewdiff_ratio={num/den:.3f} "
              f"viewCKA={c:.3f} decodeR2_2d={r2:.3f}", flush=True)
    # whole-space reference
    print(f"VIEWINV {args.tag} sub=all viewdiff_ratio="
          f"{np.linalg.norm(D,'fro')/(np.linalg.norm(P0-mu0,'fro')+1e-12):.3f} "
          f"viewCKA={cka(P0, PT):.3f}")


if __name__ == "__main__":
    main()
