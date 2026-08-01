"""EXP1 static part: ridge-head lambda sweep on frozen penultimate features (t=0, zeros).
For each lambda: fit readout (with intercept) on clean states; compute on the standard
pairs protocol (bands [1,4), anchors = 40 clean demos):
  - operator G_lam (ridge of DR on DZ), opR2, pred-cos vs G_GT, pos-block eigs, negdef
  - rot-block top singular value (de-z-scored)
  - bias bins: mean |first-executed-action pred - GT recovery action| in d bins
Saves all G_lam to one npz for local bad-mode projection analysis.
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
LAMS = [1e-8, 1e-6, 1e-4, 1e-2, 1.0, 1e2]


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
    ap.add_argument("--train_demos", type=int, default=200); ap.add_argument("--test_n", type=int, default=400)
    ap.add_argument("--random_init", action="store_true")
    ap.add_argument("--view", default="t0", choices=["t0", "tau"])
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
    TAUV = 0.0 if args.view == "t0" else float(cfg.optimization.t_two_step)
    dev = cfg.optimization.device
    no = ds.normalizer["obs"]["state"]; na = ds.normalizer["action"]
    A10 = ds.replay_buffer["action"][:]
    ends = np.asarray(ds.replay_buffer.episode_ends[:]); stt = np.concatenate([[0], ends[:-1]])

    feats = {}
    net = None
    for name, m in ag.flow_map_ema.named_modules():
        if name.endswith("final_conv"):
            net = m
    last = list(net.children())[-1]
    last.register_forward_pre_hook(lambda mod, inp: feats.__setitem__("x", inp[0].detach()))

    def phi(win):
        x = torch.tensor(no.normalize(np.stack(win)[None]), device=dev, dtype=torch.float32)
        with torch.no_grad():
            with ag._inference_mode():
                emb = ag.encoder_ema({"state": x}, None)
                ag.flow_map_ema.get_velocity(torch.full((1,), TAUV, device=dev),
                                             torch.zeros((1, H, 10), device=dev), emb)
        return feats["x"].reshape(-1).cpu().numpy()

    clean = read(args.clean, args.train_demos)
    # readout training set: target = first executed action (6d, raw units)
    Xtr, Ytr = [], []
    for i, (ov, acts) in enumerate(clean):
        for t in range(1, len(acts) - H, 6):
            Xtr.append(phi([ov[t - 1], ov[t]])); Ytr.append(acts[t, :6])
    Xtr = np.stack(Xtr).astype(np.float64); Ytr = np.stack(Ytr).astype(np.float64)
    mu_f = Xtr.mean(0); Xc = Xtr - mu_f
    mu_y = Ytr.mean(0); Yc = Ytr - mu_y
    XtX = Xc.T @ Xc; XtY = Xc.T @ Yc

    # pairs, band [1,4) (binned later)
    anchors40 = clean[:40]
    cl = np.concatenate([ov for ov, _ in anchors40], 0)
    owner = np.concatenate([[(i, t) for t in range(len(ov))] for i, (ov, _) in enumerate(anchors40)], 0)
    mu, sig = cl.mean(0), cl.std(0) + 1e-6
    tree = cKDTree((cl - mu) / sig)
    test = read(args.test, args.test_n)
    F_s, F_0, DZ, GT_d, GT_abs, DBIN = [], [], [], [], [], []
    f0cache = {}
    for ov, acts in test:
        for t in range(1, len(acts) - H, 4):
            z = (ov[t] - mu) / sig
            d, idx = tree.query(z)
            if not (1.0 <= d < 4.0): continue
            i0, t0 = map(int, owner[int(idx)])
            if int(idx) not in f0cache:
                f0cache[int(idx)] = phi([anchors40[i0][0][max(t0 - 1, 0)], anchors40[i0][0][t0]])
            F_0.append(f0cache[int(idx)])
            F_s.append(phi([ov[t - 1], ov[t]]))
            DZ.append(z - (cl[int(idx)] - mu) / sig)
            a_anchor = anchors40[i0][1][min(t0, len(anchors40[i0][1]) - 1), :6]
            GT_d.append(acts[t, :6] - a_anchor)
            GT_abs.append(acts[t, :6])
            DBIN.append(d)
    F_s = np.stack(F_s).astype(np.float64); F_0 = np.stack(F_0).astype(np.float64)
    DZ = np.stack(DZ); GT_d = np.stack(GT_d); GT_abs = np.stack(GT_abs); DBIN = np.array(DBIN)
    m24 = DBIN >= 2.0   # operator band [2,4)

    def ridge6(X, Y, l=1e-2):
        Aa = X.T @ X + l * len(X) * np.eye(X.shape[1]); return np.linalg.solve(Aa, X.T @ Y).T
    G_gt = ridge6(DZ[m24], GT_d[m24])
    out = {"G_gt": G_gt, "sig": sig, "lams": np.array(LAMS)}
    for lam in LAMS:
        W = np.linalg.solve(XtX + lam * len(Xc) * np.eye(Xc.shape[1]), XtY)
        r2tr = 1 - np.sum((Xc @ W - Yc) ** 2) / np.sum(Yc ** 2)
        P_s = (F_s - mu_f) @ W + mu_y; P_0 = (F_0 - mu_f) @ W + mu_y
        DR = P_s - P_0
        G = ridge6(DZ[m24], DR[m24])
        Pa = DZ[m24] @ G.T; Pb = DZ[m24] @ G_gt.T
        pcos = float(np.mean(np.sum(Pa * Pb, 1) /
                             (np.linalg.norm(Pa, axis=1) * np.linalg.norm(Pb, axis=1) + 1e-9)))
        r2op = 1 - np.sum((DR[m24] - Pa) ** 2) / np.sum((DR[m24] - DR[m24].mean(0)) ** 2)
        B = G[0:3, POS] * sig[POS][None, :]
        w = np.linalg.eigvalsh((B + B.T) / 2)
        rsv = np.linalg.svd(G[3:6, :] * sig[None, :], compute_uv=False)[0]
        # bias bins: |pred - GT recovery action|
        bins = []
        for lo, hi in [(1, 2), (2, 3), (3, 4)]:
            m = (DBIN >= lo) & (DBIN < hi)
            bins.append(float(np.mean(np.linalg.norm(P_s[m] - GT_abs[m], axis=1))) if m.sum() else float("nan"))
        print(f"RIDGESWEEP {args.tag} lam={lam:g} trainR2={r2tr:.3f} opR2={r2op:.3f} "
              f"cosGT={pcos:.2f} poseig={np.round(w,3).tolist()} negdef={bool(np.all(w<0))} "
              f"rotsv={rsv:.3f} bias[1-2/2-3/3-4]={np.round(bins,3).tolist()}", flush=True)
        out[f"G_lam{lam:g}"] = G
    np.savez(os.path.join(os.path.dirname(args.ckpt), f"ridgesweep_{args.tag}.npz"), **out)


if __name__ == "__main__":
    main()
