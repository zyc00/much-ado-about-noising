"""EXP-A2: SERVO_SUBSPACE_ABLATION on MIP features.
Estimate the servo feature subspace U from off-support pairs: the left singular vectors of
the cross-covariance  M = sum_pairs  dphi (G_GT dz)^T  (2048 x 6, rank <= 6).
Ablate top-k directions at inference: phi' = phi - U_k U_k^T (phi - mu_phi).
Deploy a clean-data ridge head (refit is NOT allowed to see the ablation: fit on original
features, deploy on ablated ones — measures causal use of those directions).
Also reports, per k: on-support readout R2 (on ablated features), operator cos/eigs,
then closed-loop SR with PNR-crossing stats for the chosen --k.
Modes: --k -1 = static sweep over all k (no rollouts); --k >= 0 = closed-loop for that k.
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
KM = {"object": "object-state"}
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
    ap.add_argument("--k", type=int, default=-1, help="-1: static sweep; >=0: closed-loop SR with this k")
    ap.add_argument("--null_dir", action="store_true", help="specificity control: ablate the k LOWEST of the 6 cross-cov directions (near-zero servo content) instead of the top-k")
    ap.add_argument("--refit", action="store_true", help="refit the ridge heads ON THE ABLATED features (clean information-removal test; avoids head/feature mismatch)")
    ap.add_argument("--tag", default="")
    ap.add_argument("--clean", default="data/tool_hang_full2ins_2000.hdf5")
    ap.add_argument("--test", default="data/tool_hang_puredart_full2ins_2000.hdf5")
    ap.add_argument("--train_demos", type=int, default=200); ap.add_argument("--test_n", type=int, default=400)
    ap.add_argument("--seed_lo", type=int, default=21000); ap.add_argument("--seed_hi", type=int, default=21100)
    ap.add_argument("--settle", type=int, default=10); ap.add_argument("--max_steps", type=int, default=700)
    args = ap.parse_args()
    cfgdir = os.path.abspath("examples/configs")
    with initialize_config_dir(version_base=None, config_dir=cfgdir):
        cfg = compose(config_name="main", overrides=["task=tool_hang_ph_state_delta_legacy",
            f"+task.dataset_path={os.path.abspath(args.clean)}", "network=chiunet",
            f"optimization.loss_type={args.loss}", "optimization.auto_resume=false", "log.wandb_mode=disabled"])
    OmegaConf.set_struct(cfg, False); cfg.task.obs_dim = 53
    ds = make_dataset(cfg.task)
    ag = TrainingAgent(cfg); ag.load(args.ckpt, load_optimizer=False); ag.eval()
    dev = cfg.optimization.device; start = cfg.task.obs_steps - 1; AS = cfg.task.act_steps
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
                ag.flow_map_ema.get_velocity(torch.zeros(1, device=dev),
                                             torch.zeros((1, H, 10), device=dev), emb)
        return feats["x"].reshape(-1).cpu().numpy()

    clean = read(args.clean, args.train_demos)
    # ---- ridge chunk-head on ORIGINAL features (deployment head) ----
    Xtr, Ychunk, Yfirst = [], [], []
    for i, (ov, acts) in enumerate(clean):
        T = ends[i] - stt[i]
        for t in range(1, T - (H - 1), 6):
            Xtr.append(phi([ov[t - 1], ov[t]]))
            Ychunk.append(na.normalize(A10[stt[i] + t - 1: stt[i] + t - 1 + H])[start:start + AS].reshape(-1))
            Yfirst.append(acts[t, :6])
    Xtr = np.stack(Xtr).astype(np.float64)
    Ychunk = np.stack(Ychunk).astype(np.float64); Yfirst = np.stack(Yfirst).astype(np.float64)
    mu_f = Xtr.mean(0); Xc = Xtr - mu_f
    lam = 1e-4
    A = Xc.T @ Xc + lam * len(Xc) * np.eye(Xc.shape[1])
    mu_c = Ychunk.mean(0); Wc = np.linalg.solve(A, Xc.T @ (Ychunk - mu_c))
    mu_1 = Yfirst.mean(0); W1 = np.linalg.solve(A, Xc.T @ (Yfirst - mu_1))

    # ---- pairs & servo subspace ----
    anchors40 = clean[:40]
    cl = np.concatenate([ov for ov, _ in anchors40], 0)
    owner = np.concatenate([[(i, t) for t in range(len(ov))] for i, (ov, _) in enumerate(anchors40)], 0)
    mu, sig = cl.mean(0), cl.std(0) + 1e-6
    tree = cKDTree((cl - mu) / sig)
    test = read(args.test, args.test_n)
    F_s, F_0, DZ, GT, DD = [], [], [], [], []
    f0c = {}
    for ov, acts in test:
        for t in range(1, len(acts) - H, 4):
            z = (ov[t] - mu) / sig
            d, idx = tree.query(z)
            if not (1.0 <= d < 6.0): continue
            i0, t0 = map(int, owner[int(idx)])
            if int(idx) not in f0c:
                f0c[int(idx)] = phi([anchors40[i0][0][max(t0 - 1, 0)], anchors40[i0][0][t0]])
            F_0.append(f0c[int(idx)]); F_s.append(phi([ov[t - 1], ov[t]]))
            DZ.append(z - (cl[int(idx)] - mu) / sig)
            GT.append(acts[t, :6] - anchors40[i0][1][min(t0, len(anchors40[i0][1]) - 1), :6])
            DD.append(d)
    F_s = np.stack(F_s).astype(np.float64); F_0 = np.stack(F_0).astype(np.float64)
    DZ = np.stack(DZ); GT = np.stack(GT); DD = np.array(DD)
    m12 = DD < 2.0; m24 = (DD >= 2.0) & (DD < 4.0); m46 = DD >= 4.0

    def ridge6(X, Y, l=1e-2):
        Aa = X.T @ X + l * len(X) * np.eye(X.shape[1]); return np.linalg.solve(Aa, X.T @ Y).T
    # U and G_gt from the standard [2,4) band (protocol unchanged)
    G_gt = ridge6(DZ[m24], GT[m24])
    G_gt12 = ridge6(DZ[m12], GT[m12])
    G_gt46 = ridge6(DZ[m46], GT[m46]) if m46.sum() > 60 else None
    tgt = DZ[m24] @ G_gt.T                  # servo-predicted action deltas, [2,4)
    dphi = F_s[m24] - F_0[m24]
    M = dphi.T @ tgt                        # (2048, 6)
    U, sv, _ = np.linalg.svd(M, full_matrices=False)   # U: (2048, 6)
    print(f"SERVOABL {args.tag} subspace svals={np.round(sv/sv[0],3).tolist()} "
          f"N12={int(m12.sum())} N24={int(m24.sum())}", flush=True)

    def ablate(F, k):
        if k == 0: return F
        Uk = U[:, 6 - k:] if args.null_dir else U[:, :k]
        return F - (F - mu_f) @ Uk @ Uk.T

    ks = [0, 1, 2, 3, 5, 6] if args.k < 0 else [args.k]
    for k in ks:
        Fs_a = ablate(F_s, k); F0_a = ablate(F_0, k); Xa = ablate(Xtr, k)
        if args.refit and k > 0:
            # refit both heads on the ablated features (information-removal test)
            Xac = Xa - mu_f
            Aa_ = Xac.T @ Xac + lam * len(Xac) * np.eye(Xac.shape[1])
            Wc = np.linalg.solve(Aa_, Xac.T @ (Ychunk - mu_c))
            W1 = np.linalg.solve(Aa_, Xac.T @ (Yfirst - mu_1))
        r2on = 1 - np.sum(((Xa - mu_f) @ W1 - (Yfirst - mu_1)) ** 2) / np.sum((Yfirst - mu_1) ** 2)
        r2ch = 1 - np.sum(((Xa - mu_f) @ Wc - (Ychunk - mu_c)) ** 2) / np.sum((Ychunk - mu_c) ** 2)
        featshift = float(np.linalg.norm(Xa - Xtr, "fro") / (np.linalg.norm(Xtr - mu_f, "fro") + 1e-12))
        outshift = float(np.mean(np.linalg.norm((Xa - mu_f) @ Wc - (Xtr - mu_f) @ Wc, axis=1)))
        chunkerr = float(np.mean(np.linalg.norm((Xa - mu_f) @ Wc - (Ychunk - mu_c), axis=1)))
        print(f"SERVOABL {args.tag} k={k} featshift={featshift:.4f} outshift={outshift:.4f} "
              f"chunkerr={chunkerr:.4f}", flush=True)
        DR = ((Fs_a - mu_f) @ W1) - ((F0_a - mu_f) @ W1)
        bands = [("12", m12, G_gt12), ("24", m24, G_gt)]
        if G_gt46 is not None:
            bands.append(("46", m46, G_gt46))
        for bname, mm, Gg in bands:
            G = ridge6(DZ[mm], DR[mm])
            Pa = DZ[mm] @ G.T; Pb = DZ[mm] @ Gg.T
            pcos = float(np.mean(np.sum(Pa * Pb, 1) /
                                 (np.linalg.norm(Pa, axis=1) * np.linalg.norm(Pb, axis=1) + 1e-9)))
            B = G[0:3, POS] * sig[POS][None, :]
            w = np.linalg.eigvalsh((B + B.T) / 2)
            print(f"SERVOABL {args.tag} k={k} band[{bname}] onR2={r2on:.3f} chunkR2={r2ch:.3f} "
                  f"cosGT={pcos:.2f} poseig={np.round(w,3).tolist()} "
                  f"negdef={bool(np.all(w<0))} refit={args.refit}", flush=True)

    if args.k < 0:
        return
    # ---- closed loop with ablated features + chunk ridge head ----
    import robosuite
    from collect_tool_hang_demos import ENV_KWARGS
    k = args.k
    anch = [ov for ov, _ in anchors40]
    env = robosuite.make("ToolHang", horizon=4000, **ENV_KWARGS)

    def ov_(o):
        return np.concatenate([o[KM.get(kk, kk)] for kk in OK]).astype(np.float32)

    def chunk(hist):
        f = ablate(phi(np.stack(hist[-2:]))[None], k)[0]
        an = ((f - mu_f) @ Wc + mu_c).reshape(1, AS, 10)
        return ds.undo_transform_action(na.unnormalize(an))[0]

    anc_first = np.concatenate([a[:len(o)][:, :6] if len(a) >= len(o) else
                                np.pad(a, ((0, len(o) - len(a)), (0, 0)), "edge")[:, :6]
                                for o, a in [(ov_, ac_) for ov_, ac_ in anchors40]], 0)
    succ = 0; N = 0; ep = []
    E_lo, E_hi, DD_lo, DD_hi, exc = [], [], [], [], []
    for sd in range(args.seed_lo, args.seed_hi):
        np.random.seed(sd); env.reset()
        arm = env.robots[0].composite_controller.part_controllers["right"]; arm.update(); arm.reset_goal()
        for _ in range(args.settle):
            env.step(np.zeros(7))
        o = env._get_observations(force_update=True); hist = [ov_(o), ov_(o)]
        steps = 0; asm = False; maxd = 0.0
        dser = []; eser = []
        while steps < args.max_steps and not asm:
            for a in chunk(hist):
                z = (hist[-1] - mu) / sig
                d0, idx0 = tree.query(z)
                agt = np.clip(anc_first[int(idx0)] + G_gt @ (z - (cl[int(idx0)] - mu) / sig), -1, 1)
                eser.append(float(np.linalg.norm(a[:6] - agt)))
                o, _, _, _ = env.step(a); steps += 1; hist.append(ov_(o))
                d, _ = tree.query((hist[-1] - mu) / sig)
                maxd = max(maxd, float(d)); dser.append(float(d))
                if env._check_frame_assembled():
                    asm = True; break
                if steps >= args.max_steps:
                    break
        succ += int(asm); N += 1
        dsa = np.array(dser); esa = np.array(eser[:len(dsa)])
        dda = np.diff(dsa)
        blo = dsa[:-1] < 2.0; bhi = (dsa[:-1] >= 2.0) & (dsa[:-1] < 4.0)
        if blo.any(): E_lo.append(esa[:-1][blo]); DD_lo.append(dda[blo])
        if bhi.any(): E_hi.append(esa[:-1][bhi]); DD_hi.append(dda[bhi])
        t2 = 1
        while t2 < len(dsa):
            if dsa[t2] >= 2.0 and dsa[t2 - 1] < 2.0:
                exc.append((int((dsa[t2:t2+10] < 2.0).any()), int((dsa[t2:t2+20] < 2.0).any()),
                            int((dsa[t2:t2+10] >= 4.0).any()), int((dsa[t2:t2+20] >= 4.0).any())))
                nxt = t2 + 1
                while nxt < len(dsa) and dsa[nxt] >= 2.0: nxt += 1
                t2 = nxt
            t2 += 1
        fc4 = next((i for i, dv in enumerate(dser) if dv >= 4.0), -1)
        dd = np.diff(dser) if len(dser) > 1 else np.array([0.0])
        db = np.array(dser[:-1]) if len(dser) > 1 else np.array([0.0])
        dl = float(dd[db < 2.0].mean()) if (db < 2.0).any() else float("nan")
        dh = float(dd[(db >= 2.0) & (db < 4.0)].mean()) if ((db >= 2.0) & (db < 4.0)).any() else float("nan")
        ep.append((int(asm), maxd, int(maxd >= 4.0), fc4, dl, dh))
    ep = np.array(ep); c4 = ep[:, 2].astype(bool)
    sr_c4 = 100 * ep[c4, 0].mean() if c4.any() else float("nan")
    sr_n4 = 100 * ep[~c4, 0].mean() if (~c4).any() else float("nan")
    fct = ep[ep[:, 3] >= 0, 3]
    print(f"SERVOABL {args.tag} k={k} SR seeds[{args.seed_lo},{args.seed_hi}) "
          f"assembled={succ}/{N} = {100*succ/N:.1f}% cross4={c4.mean():.2f} "
          f"SR|cross4={sr_c4:.0f} SR|stay={sr_n4:.0f} "
          f"maxd_p50/90={np.percentile(ep[:,1],50):.2f}/{np.percentile(ep[:,1],90):.2f} "
          f"firstcross4_p50={np.median(fct) if len(fct) else float('nan'):.0f} "
          f"drift_d<2={np.nanmean(ep[:,4]):+.4f} drift_[2,4)={np.nanmean(ep[:,5]):+.4f}")
    for nm2, E2, D2 in [("d<2", E_lo, DD_lo), ("[2,4)", E_hi, DD_hi)]:
        if not E2: continue
        E2 = np.concatenate(E2); D2 = np.concatenate(D2)
        print(f"SERVOABL {args.tag} k={k} rollband={nm2} N={len(E2)} "
              f"err_mean/p50/p90/p95={E2.mean():.4f}/{np.percentile(E2,50):.4f}/{np.percentile(E2,90):.4f}/{np.percentile(E2,95):.4f} "
              f"dd_mean/p90={D2.mean():+.4f}/{np.percentile(D2,90):+.4f} frac_dd>0={(D2>0).mean():.2f}", flush=True)
    if exc:
        X2 = np.array(exc)
        print(f"SERVOABL {args.tag} k={k} excursions N={len(X2)} return10/20={X2[:,0].mean():.2f}/{X2[:,1].mean():.2f} "
              f"cross4_10/20={X2[:,2].mean():.2f}/{X2[:,3].mean():.2f}")


if __name__ == "__main__":
    main()
