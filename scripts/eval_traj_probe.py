"""CHECK 3: training-trajectory probe for one checkpoint.
Penultimate feature-ridge (t=0, zeros view) -> induced operator on the standard pairs:
  cos(G_phi, G_GT), pos-block eigs, rot-block top-sv,
  bad-mode projections (gain@bad1, rot@MSE-amp from badmode_ref.npz),
  principal angles of the ckpt's 2-d servo feature subspace vs the RANDOM-INIT one
  (U_rand from logs/urand_ref.npz, produced by eval_cka_drift.py).
Prints one TRAJPROBE line per call.
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


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", required=True); ap.add_argument("--loss", required=True)
    ap.add_argument("--tag", default="")
    ap.add_argument("--clean", default="data/tool_hang_full2ins_2000.hdf5")
    ap.add_argument("--test", default="data/tool_hang_puredart_full2ins_2000.hdf5")
    ap.add_argument("--ref", default="analysis/recovery/badmode_ref.npz")
    ap.add_argument("--urand", default="logs/urand_ref.npz")
    ap.add_argument("--train_demos", type=int, default=100); ap.add_argument("--test_n", type=int, default=300)
    args = ap.parse_args()
    cfgdir = os.path.abspath("examples/configs")
    with initialize_config_dir(version_base=None, config_dir=cfgdir):
        cfg = compose(config_name="main", overrides=["task=tool_hang_ph_state_delta_legacy",
            f"+task.dataset_path={os.path.abspath(args.clean)}", "network=chiunet",
            f"optimization.loss_type={args.loss}", "optimization.auto_resume=false", "log.wandb_mode=disabled"])
    OmegaConf.set_struct(cfg, False); cfg.task.obs_dim = 53
    ds = make_dataset(cfg.task)
    ag = TrainingAgent(cfg); ag.load(args.ckpt, load_optimizer=False); ag.eval()
    dev = cfg.optimization.device
    no = ds.normalizer["obs"]["state"]

    feats = {}
    net = None
    for name, m in ag.flow_map_ema.named_modules():
        if name.endswith("final_conv"):
            net = m
    list(net.children())[-1].register_forward_pre_hook(
        lambda mod, inp: feats.__setitem__("x", inp[0].detach()))

    def phi(win):
        x = torch.tensor(no.normalize(np.stack(win)[None]), device=dev, dtype=torch.float32)
        with torch.no_grad():
            with ag._inference_mode():
                emb = ag.encoder_ema({"state": x}, None)
                ag.flow_map_ema.get_velocity(torch.zeros(1, device=dev),
                                             torch.zeros((1, H, 10), device=dev), emb)
        return feats["x"].reshape(-1).cpu().numpy()

    clean = read(args.clean, max(args.train_demos, 40))
    Xtr, Ytr = [], []
    for i, (ov, acts) in enumerate(clean[:args.train_demos]):
        for t in range(1, len(acts) - H, 6):
            Xtr.append(phi([ov[t - 1], ov[t]])); Ytr.append(acts[t, :6])
    Xtr = np.stack(Xtr).astype(np.float64); Ytr = np.stack(Ytr).astype(np.float64)
    mu_f = Xtr.mean(0); Xc = Xtr - mu_f
    mu_y = Ytr.mean(0); Yc = Ytr - mu_y
    W = np.linalg.solve(Xc.T @ Xc + 1e-3 * len(Xc) * np.eye(Xc.shape[1]), Xc.T @ Yc)

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
                f0c[int(idx)] = phi([anchors40[i0][0][max(t0 - 1, 0)], anchors40[i0][0][t0]])
            F_0.append(f0c[int(idx)]); F_s.append(phi([ov[t - 1], ov[t]]))
            DZ.append(z - (cl[int(idx)] - mu) / sig)
            GT.append(acts[t, :6] - anchors40[i0][1][min(t0, len(anchors40[i0][1]) - 1), :6])
    F_s = np.stack(F_s).astype(np.float64); F_0 = np.stack(F_0).astype(np.float64)
    DZ = np.stack(DZ); GT = np.stack(GT)

    def ridge6(X, Y, l=1e-2):
        Aa = X.T @ X + l * len(X) * np.eye(X.shape[1]); return np.linalg.solve(Aa, X.T @ Y).T
    G_gt = ridge6(DZ, GT)
    DR = (F_s - mu_f) @ W - (F_0 - mu_f) @ W
    G = ridge6(DZ, DR)
    Pa = DZ @ G.T; Pb = DZ @ G_gt.T
    pcos = float(np.mean(np.sum(Pa * Pb, 1) /
                         (np.linalg.norm(Pa, axis=1) * np.linalg.norm(Pb, axis=1) + 1e-9)))
    B = G[0:3, POS] * sig[POS][None, :]
    w = np.linalg.eigvalsh((B + B.T) / 2)
    R = G[3:6, :] * sig[None, :]
    rsv = float(np.linalg.svd(R, compute_uv=False)[0])

    ref = np.load(args.ref)
    def posblk(Gx):
        Bx = Gx[0:3, POS] * ref["sig"][POS][None, :]
        return (Bx + Bx.T) / 2
    wm, Vm = np.linalg.eigh(posblk(ref["G_m"]))
    bad1 = Vm[:, np.argmax(wm)]
    vrot = np.linalg.svd(ref["G_m"][3:6, :] * ref["sig"][None, :])[2][0]
    S = posblk(G)
    gb = float(bad1 @ S @ bad1)
    ra = float(np.linalg.norm((G[3:6, :] * ref["sig"][None, :]) @ vrot))

    ang_str = "NA"
    if os.path.exists(args.urand):
        ur = np.load(args.urand)["U_rand"]
        tgt = DZ @ G_gt.T
        M = (F_s - F_0).T @ tgt
        U = np.linalg.svd(M, full_matrices=False)[0][:, :2]
        s = np.linalg.svd(ur.T @ U, compute_uv=False)
        ang_str = str(np.round(np.degrees(np.arccos(np.clip(s, -1, 1))), 1).tolist())

    # ---- CKA to seed-0 random init: all dims / servo subspace / bad-mode subspace ----
    torch.manual_seed(0)
    ag_r = TrainingAgent(cfg); ag_r.eval()
    feats_r = {}
    net_r = None
    for name, m in ag_r.flow_map_ema.named_modules():
        if name.endswith("final_conv"):
            net_r = m
    list(net_r.children())[-1].register_forward_pre_hook(
        lambda mod, inp: feats_r.__setitem__("x", inp[0].detach()))
    def phi_r(win):
        x = torch.tensor(no.normalize(np.stack(win)[None]), device=dev, dtype=torch.float32)
        with torch.no_grad():
            with ag_r._inference_mode():
                emb = ag_r.encoder_ema({"state": x}, None)
                ag_r.flow_map_ema.get_velocity(torch.zeros(1, device=dev),
                                               torch.zeros((1, H, 10), device=dev), emb)
        return feats_r["x"].reshape(-1).cpu().numpy()
    # shared subsample of on-support states (reuse Xtr rows' generating windows is gone;
    # recompute on a stride-12 subset of clean demos)
    wins = []
    for i, (ov, acts) in enumerate(clean[:args.train_demos]):
        for t in range(1, len(acts) - H, 12):
            wins.append([ov[t - 1], ov[t]])
    PH = np.stack([phi(wn) for wn in wins]).astype(np.float64)
    PR = np.stack([phi_r(wn) for wn in wins]).astype(np.float64)
    def cka(X, Y):
        Xc2 = X - X.mean(0); Yc2 = Y - Y.mean(0)
        return float(np.linalg.norm(Xc2.T @ Yc2, "fro") ** 2 /
                     (np.linalg.norm(Xc2.T @ Xc2, "fro") * np.linalg.norm(Yc2.T @ Yc2, "fro")))
    c_all = cka(PH, PR)
    c_srv = c_bad = float("nan")
    if os.path.exists(args.urand):
        ur = np.load(args.urand)["U_rand"]
        c_srv = cka(PH @ ur, PR @ ur)
        # bad-mode feature dirs: readout-weight combos producing bad1 (pose) and MSE
        # rot-amp output direction
        e_bad1 = np.zeros(6); e_bad1[0:3] = bad1
        rout = np.linalg.svd(ref["G_m"][3:6, :] * ref["sig"][None, :])[0][:, 0]
        e_rot = np.zeros(6); e_rot[3:6] = rout
        Ub = np.linalg.qr(np.stack([W @ e_bad1, W @ e_rot], 1))[0]
        c_bad = cka(PH @ Ub, PR @ Ub)
    print(f"TRAJPROBE {args.tag} cosGT={pcos:.2f} poseig={np.round(w,3).tolist()} "
          f"negdef={bool(np.all(w<0))} rotsv={rsv:.3f} gain@bad1={gb:+.4f} "
          f"rot@mseamp={ra:.3f} angles_vs_Urand={ang_str} "
          f"CKArand_all={c_all:.3f} CKArand_servo={c_srv:.3f} CKArand_bad={c_bad:.3f}")


if __name__ == "__main__":
    main()
