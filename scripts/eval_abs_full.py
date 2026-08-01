"""EXP-1: ABS_FEATURE_RIDGE_AND_IMPLIED_DELTA.
For an abs-representation model:
 1. raw abs operator: physical-unit responses d(goal_pos), d(goal_rot axis-angle) ~ G_abs dz
    vs analytic G_GT_abs (goal_pos = eef_pos + 0.05 a_pos; R_goal = R(0.5 a_rot) R(eef)).
 2. implied delta: a_delta_implied = [(goal_pos_hat - eef_pos)/0.05, axisangle(R_hat R(eef)^-1)/0.5]
    responses ~ G_delta_implied dz vs G_GT_delta (action units) + badmode projections.
 3. residual form: sym-eigs & cos of (G_abs_posblock - I) vs G_GT_delta posblock.
 4. feature ridge (phi0 / phitau, --view): frozen penultimate + clean ridge -> both operators.
 5. target-error tails on DART states: |pred_abs - GT_abs| pos/rot p50/90/95/99, overall and
    insertion stage (anchor progress > 0.75).
"""
import os
os.environ["MUJOCO_GL"] = "egl"
import argparse
import numpy as np
import torch
import h5py
import sys
sys.path.insert(0, "scripts"); sys.path.insert(0, ".")
import robosuite.utils.transform_utils as T
from hydra import initialize_config_dir, compose
from omegaconf import OmegaConf
from scipy.spatial import cKDTree
from mip.datasets.robomimic_dataset import make_dataset
from mip.agent import TrainingAgent

OK = ["object", "robot0_eef_pos", "robot0_eef_quat", "robot0_gripper_qpos"]
POS = slice(44, 47); QUAT = slice(47, 51)
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


def goal_of(obs_row, act_delta):
    p = obs_row[POS] + 0.05 * act_delta[:3]
    Rd = T.quat2mat(T.axisangle2quat(0.5 * act_delta[3:6]))
    return p, Rd @ T.quat2mat(obs_row[QUAT])


def rotdiff(Ra, Rb):
    return T.quat2axisangle(T.mat2quat(Ra @ Rb.T))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", required=True); ap.add_argument("--loss", required=True)
    ap.add_argument("--tag", default="")
    ap.add_argument("--absdata", default="data/tool_hang_full2ins_abs_2000.hdf5")
    ap.add_argument("--clean", default="data/tool_hang_full2ins_2000.hdf5")
    ap.add_argument("--test", default="data/tool_hang_puredart_full2ins_2000.hdf5")
    ap.add_argument("--ref", default="analysis/recovery/badmode_ref.npz")
    ap.add_argument("--view", default="t0", choices=["t0", "tau"])
    ap.add_argument("--random_init", action="store_true")
    ap.add_argument("--train_demos", type=int, default=150); ap.add_argument("--test_n", type=int, default=400)
    args = ap.parse_args()
    cfgdir = os.path.abspath("examples/configs")
    with initialize_config_dir(version_base=None, config_dir=cfgdir):
        cfg = compose(config_name="main", overrides=["task=tool_hang_ph_state_delta_legacy",
            f"+task.dataset_path={os.path.abspath(args.absdata)}", "network=chiunet",
            f"optimization.loss_type={args.loss}", "optimization.auto_resume=false", "log.wandb_mode=disabled"])
    OmegaConf.set_struct(cfg, False); cfg.task.obs_dim = 53
    ds = make_dataset(cfg.task)
    if args.random_init:
        torch.manual_seed(0)
    ag = TrainingAgent(cfg)
    if not args.random_init:
        ag.load(args.ckpt, load_optimizer=False)
    ag.eval()
    dev = cfg.optimization.device; start = cfg.task.obs_steps - 1; AS = cfg.task.act_steps
    no = ds.normalizer["obs"]["state"]; na = ds.normalizer["action"]
    TAUV = 0.0 if args.view == "t0" else float(cfg.optimization.t_two_step)

    feats = {}
    net = None
    for name, m in ag.flow_map_ema.named_modules():
        if name.endswith("final_conv"):
            net = m
    list(net.children())[-1].register_forward_pre_hook(
        lambda mod, inp: feats.__setitem__("x", inp[0].detach()))

    def fwd(win):
        x = torch.tensor(no.normalize(np.stack(win)[None]), device=dev, dtype=torch.float32)
        with torch.no_grad():
            with ag._inference_mode():
                emb = ag.encoder_ema({"state": x}, None)
                an = ag.flow_map_ema.get_velocity(torch.full((1,), TAUV, device=dev),
                                                  torch.zeros((1, H, 10), device=dev), emb)
        a7 = ds.undo_transform_action(na.unnormalize(an.detach().cpu().numpy())[:, start:start + AS])[0][0]
        return a7, feats["x"].reshape(-1).cpu().numpy()

    # ---- pairs (obs from DELTA files; GT converted analytically) ----
    clean = read(args.clean, 40)
    cl = np.concatenate([ov for ov, _ in clean], 0)
    owner = np.concatenate([[(i, t) for t in range(len(ov))] for i, (ov, _) in enumerate(clean)], 0)
    lens = {i: len(ov) for i, (ov, _) in enumerate(clean)}
    mu, sig = cl.mean(0), cl.std(0) + 1e-6
    tree = cKDTree((cl - mu) / sig)
    test = read(args.test, args.test_n)
    DZ, GTA, GTD, RA, RD, F_s, F_0, PROG = [], [], [], [], [], [], [], []
    ERR_pos, ERR_rot, ERR_prog = [], [], []
    f0c = {}
    for ov, acts in test:
        for t in range(1, len(acts) - H, 4):
            z = (ov[t] - mu) / sig
            d, idx = tree.query(z)
            if not (2.0 <= d < 4.0): continue
            i0, t0 = map(int, owner[int(idx)])
            a0 = clean[i0][1][min(t0, len(clean[i0][1]) - 1)]
            gt = acts[t]
            gp_s, Rg_s = goal_of(ov[t], gt); gp_0, Rg_0 = goal_of(clean[i0][0][t0], a0)
            GTA.append(np.concatenate([gp_s - gp_0, rotdiff(Rg_s, Rg_0)]))
            GTD.append(np.concatenate([gt[:3] - a0[:3],
                       rotdiff(T.quat2mat(T.axisangle2quat(0.5 * gt[3:6])),
                               T.quat2mat(T.axisangle2quat(0.5 * a0[3:6]))) / 0.5]))
            key = int(idx)
            if key not in f0c:
                a7_0, f0 = fwd([clean[i0][0][max(t0 - 1, 0)], clean[i0][0][t0]])
                f0c[key] = (a7_0, f0, clean[i0][0][t0])
            a7_s, fs = fwd([ov[t - 1], ov[t]])
            a7_0, f0, obs0 = f0c[key]
            # model outputs interpreted as ABS goal pose: [pos(3), axisangle(3), grip]
            Rg_hat_s = T.quat2mat(T.axisangle2quat(a7_s[3:6]))
            Rg_hat_0 = T.quat2mat(T.axisangle2quat(a7_0[3:6]))
            RA.append(np.concatenate([a7_s[:3] - a7_0[:3], rotdiff(Rg_hat_s, Rg_hat_0)]))
            # implied delta (action units)
            di_s = np.concatenate([(a7_s[:3] - ov[t][POS]) / 0.05,
                                   T.quat2axisangle(T.mat2quat(Rg_hat_s @ T.quat2mat(ov[t][QUAT]).T)) / 0.5])
            di_0 = np.concatenate([(a7_0[:3] - obs0[POS]) / 0.05,
                                   T.quat2axisangle(T.mat2quat(Rg_hat_0 @ T.quat2mat(obs0[QUAT]).T)) / 0.5])
            RD.append(di_s - di_0)
            F_s.append(fs); F_0.append(f0)
            DZ.append(z - (cl[key] - mu) / sig)
            pg = t0 / max(lens[i0] - 1, 1); PROG.append(pg)
            # target error tails (model abs pred vs analytic GT abs on this state)
            epos = np.linalg.norm(a7_s[:3] - gp_s); erot = np.linalg.norm(rotdiff(Rg_hat_s, Rg_s))
            ERR_pos.append(epos); ERR_rot.append(erot); ERR_prog.append(pg)
    DZ = np.stack(DZ); GTA = np.stack(GTA); GTD = np.stack(GTD)
    RA = np.stack(RA); RD = np.stack(RD)
    F_s = np.stack(F_s).astype(np.float64); F_0 = np.stack(F_0).astype(np.float64)
    ERR_pos = np.array(ERR_pos); ERR_rot = np.array(ERR_rot); ERR_prog = np.array(ERR_prog)

    def ridge6(X, Y, l=1e-2):
        Aa = X.T @ X + l * len(X) * np.eye(X.shape[1]); return np.linalg.solve(Aa, X.T @ Y).T
    def r2of(G, X, Y):
        P = X @ G.T
        return 1 - np.sum((Y - P) ** 2) / np.sum((Y - Y.mean(0)) ** 2)
    def pcos(G, Gg, X):
        Pa = X @ G.T; Pb = X @ Gg.T
        return float(np.mean(np.sum(Pa * Pb, 1) /
                             (np.linalg.norm(Pa, axis=1) * np.linalg.norm(Pb, axis=1) + 1e-9)))
    def blocks(G, unit=1.0):
        B = G[0:3, POS] * sig[POS][None, :] * unit
        w = np.linalg.eigvalsh((B + B.T) / 2)
        rsv = float(np.linalg.svd(G[3:6, :] * sig[None, :], compute_uv=False)[0])
        return w, rsv, B

    G_gt_a = ridge6(DZ, GTA); G_gt_d = ridge6(DZ, GTD)
    G_a = ridge6(DZ, RA); G_d = ridge6(DZ, RD)
    ref = np.load(args.ref)
    def badproj(G):
        Bx = G[0:3, POS] * ref["sig"][POS][None, :]
        S = (Bx + Bx.T) / 2
        wm, Vm = np.linalg.eigh((ref["G_m"][0:3, POS] * ref["sig"][POS][None, :] +
                                 (ref["G_m"][0:3, POS] * ref["sig"][POS][None, :]).T) / 2)
        bad1 = Vm[:, np.argmax(wm)]
        vrot = np.linalg.svd(ref["G_m"][3:6, :] * ref["sig"][None, :])[2][0]
        return float(bad1 @ S @ bad1), float(np.linalg.norm((G[3:6, :] * ref["sig"][None, :]) @ vrot))

    w, rsv, _ = blocks(G_a)
    print(f"ABSFULL {args.tag} RAW_ABS R2={r2of(G_a, DZ, RA):.3f} cosGTabs={pcos(G_a, G_gt_a, DZ):.2f} "
          f"poseig={np.round(w,4).tolist()} rotsv={rsv:.3f}", flush=True)
    w, rsv, _ = blocks(G_d)
    gb, ra = badproj(G_d)
    print(f"ABSFULL {args.tag} IMPLIED_DELTA R2={r2of(G_d, DZ, RD):.3f} cosGTdelta={pcos(G_d, G_gt_d, DZ):.2f} "
          f"poseig={np.round(w,4).tolist()} rotsv={rsv:.3f} gain@bad1={gb:+.4f} rot@mseamp={ra:.3f}", flush=True)
    # residual form: (G_abs - dEef/dz) pos block vs GT delta pos block
    Iblk = np.zeros((3, 53)); Iblk[:, POS] = np.eye(3) / sig[POS][None, :] * sig[POS][None, :]
    B_res = (G_a[0:3, :] - np.pad(np.eye(3) / sig[POS][None, :], ((0, 0), (POS.start, 53 - POS.stop))) * 0)  # placeholder
    # compute directly: residual response = RA_pos - dz_eefpos(physical)
    RES = RA[:, :3] - DZ[:, POS] * sig[POS][None, :]
    G_res = ridge6(DZ, RES)
    Br = G_res[0:3, POS] * sig[POS][None, :]
    wr = np.linalg.eigvalsh((Br + Br.T) / 2)
    # cos vs GT delta pos part (physical: 0.05 * action units)
    G_gt_d_pos = G_gt_d[:3] * 0.05
    Pa = DZ @ G_res.T; Pb = DZ @ G_gt_d_pos.T
    c_res = float(np.mean(np.sum(Pa * Pb, 1) /
                          (np.linalg.norm(Pa, axis=1) * np.linalg.norm(Pb, axis=1) + 1e-9)))
    print(f"ABSFULL {args.tag} RESIDUAL(G_abs-I) cosGTdelta_pos={c_res:.2f} sym_eigs={np.round(wr,4).tolist()}", flush=True)
    # feature ridge
    mu_f = F_s.mean(0)  # use pair features' mean; readout fit on clean train states
    # clean readout set
    ds_clean = read(args.absdata, args.train_demos)
    Xtr, Ytr = [], []
    for i, (ov, acts) in enumerate(ds_clean):
        for t in range(1, len(acts) - H, 8):
            _, f = fwd([ov[t - 1], ov[t]])
            Xtr.append(f); Ytr.append(acts[t, :6])
    Xtr = np.stack(Xtr).astype(np.float64); Ytr = np.stack(Ytr).astype(np.float64)
    muf = Xtr.mean(0); Xc = Xtr - muf
    muy = Ytr.mean(0)
    W = np.linalg.solve(Xc.T @ Xc + 1e-3 * len(Xc) * np.eye(Xc.shape[1]), Xc.T @ (Ytr - muy))
    r2tr = 1 - np.sum((Xc @ W - (Ytr - muy)) ** 2) / np.sum((Ytr - muy) ** 2)
    DRf = (F_s - muf) @ W - (F_0 - muf) @ W   # abs-action-unit responses [pos_abs(3), aa(3)]
    Gf = ridge6(DZ, DRf)
    # abs cos: convert GTA to same units (pos in m == abs action pos units; rot axisangle)
    print(f"ABSFULL {args.tag} FEATRIDGE view={args.view} trainR2={r2tr:.3f} "
          f"opR2={r2of(Gf, DZ, DRf):.3f} cosGTabs={pcos(Gf, G_gt_a, DZ):.2f}", flush=True)
    # target error tails
    ins = ERR_prog > 0.75
    def tails(x):
        return "/".join(f"{np.percentile(x, q):.4f}" for q in [50, 90, 95, 99])
    print(f"ABSFULL {args.tag} TAILS pos_all={tails(ERR_pos)} rot_all={tails(ERR_rot)} "
          f"pos_ins={tails(ERR_pos[ins]) if ins.any() else 'NA'} rot_ins={tails(ERR_rot[ins]) if ins.any() else 'NA'} N_ins={int(ins.sum())}")


if __name__ == "__main__":
    main()
