"""EXP 9B: SUFFICIENCY BY ADAPTER DISTILLATION.
Fit a linear adapter T on clean states: T phi_MSE(s) ~ phi_MIP(s) (ridge).
Then fit the standard chunk ridge head on T phi_MSE and deploy closed-loop.
If MIP's representation is (linearly) transferable, MSE-features-through-T should
recover MIP-level operator and SR. Also reports the adapted operator on pairs.
"""
import os
os.environ["MUJOCO_GL"] = "egl"
import argparse
import numpy as np
import torch
import h5py
import sys
sys.path.insert(0, "scripts"); sys.path.insert(0, ".")
import robosuite
from collect_tool_hang_demos import ENV_KWARGS
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
    ap.add_argument("--student", default="logs/full_regression_2000_s1/models/model_latest.pt")
    ap.add_argument("--student_loss", default="regression")
    ap.add_argument("--teacher", default="logs/full_mip_2000_s1/models/model_latest.pt")
    ap.add_argument("--teacher_loss", default="mip")
    ap.add_argument("--tag", default="ADAPT")
    ap.add_argument("--clean", default="data/tool_hang_full2ins_2000.hdf5")
    ap.add_argument("--test", default="data/tool_hang_puredart_full2ins_2000.hdf5")
    ap.add_argument("--train_demos", type=int, default=200); ap.add_argument("--test_n", type=int, default=400)
    ap.add_argument("--seed_lo", type=int, default=21000); ap.add_argument("--seed_hi", type=int, default=21100)
    ap.add_argument("--settle", type=int, default=10); ap.add_argument("--max_steps", type=int, default=700)
    args = ap.parse_args()
    cfgdir = os.path.abspath("examples/configs")

    def make(ckpt, loss):
        with initialize_config_dir(version_base=None, config_dir=cfgdir):
            cfg = compose(config_name="main", overrides=["task=tool_hang_ph_state_delta_legacy",
                f"+task.dataset_path={os.path.abspath(args.clean)}", "network=chiunet",
                f"optimization.loss_type={loss}", "optimization.auto_resume=false", "log.wandb_mode=disabled"])
        OmegaConf.set_struct(cfg, False); cfg.task.obs_dim = 53
        ag = TrainingAgent(cfg); ag.load(ckpt, load_optimizer=False); ag.eval()
        feats = {}
        net = None
        for name, m in ag.flow_map_ema.named_modules():
            if name.endswith("final_conv"):
                net = m
        list(net.children())[-1].register_forward_pre_hook(
            lambda mod, inp: feats.__setitem__("x", inp[0].detach()))
        return cfg, ag, feats

    cfgS, agS, fS = make(args.student, args.student_loss)
    cfgT, agT, fT = make(args.teacher, args.teacher_loss)
    dev = cfgS.optimization.device; start = cfgS.task.obs_steps - 1; AS = cfgS.task.act_steps
    ds = make_dataset(cfgS.task)
    no = ds.normalizer["obs"]["state"]; na = ds.normalizer["action"]
    A10 = ds.replay_buffer["action"][:]
    ends = np.asarray(ds.replay_buffer.episode_ends[:]); stt = np.concatenate([[0], ends[:-1]])

    def phi_of(ag, feats, win):
        x = torch.tensor(no.normalize(np.stack(win)[None]), device=dev, dtype=torch.float32)
        with torch.no_grad():
            with ag._inference_mode():
                emb = ag.encoder_ema({"state": x}, None)
                ag.flow_map_ema.get_velocity(torch.zeros(1, device=dev),
                                             torch.zeros((1, H, 10), device=dev), emb)
        return feats["x"].reshape(-1).cpu().numpy()

    phiS = lambda w: phi_of(agS, fS, w)
    phiT = lambda w: phi_of(agT, fT, w)

    clean = read(args.clean, args.train_demos)
    XS, XT, Ychunk = [], [], []
    for i, (ov, acts) in enumerate(clean):
        T = ends[i] - stt[i]
        for t in range(1, T - (H - 1), 6):
            w = [ov[t - 1], ov[t]]
            XS.append(phiS(w)); XT.append(phiT(w))
            Ychunk.append(na.normalize(A10[stt[i] + t - 1: stt[i] + t - 1 + H])[start:start + AS].reshape(-1))
    XS = np.stack(XS).astype(np.float64); XT = np.stack(XT).astype(np.float64)
    Ychunk = np.stack(Ychunk).astype(np.float64)
    muS = XS.mean(0); muT = XT.mean(0)
    lam = 1e-4
    A_ = (XS - muS).T @ (XS - muS) + lam * len(XS) * np.eye(XS.shape[1])
    Tmap = np.linalg.solve(A_, (XS - muS).T @ (XT - muT))       # 2048x2048 adapter
    r2T = 1 - np.sum(((XS - muS) @ Tmap - (XT - muT)) ** 2) / np.sum((XT - muT) ** 2)
    XA = (XS - muS) @ Tmap                                       # adapted features (centered)
    mu_c = Ychunk.mean(0)
    A2 = XA.T @ XA + lam * len(XA) * np.eye(XA.shape[1])
    Wc = np.linalg.solve(A2, XA.T @ (Ychunk - mu_c))
    r2c = 1 - np.sum((XA @ Wc - (Ychunk - mu_c)) ** 2) / np.sum((Ychunk - mu_c) ** 2)
    print(f"ADAPT {args.tag} adapterR2={r2T:.3f} chunkR2={r2c:.3f}", flush=True)

    # ---- operator on pairs (adapted student features, first-action readout) ----
    Yfirst = np.stack([clean[i][1][t, :6]
                       for i, (ov, acts) in enumerate(clean)
                       for t in range(1, (ends[i] - stt[i]) - (H - 1), 6)]).astype(np.float64)
    mu1 = Yfirst.mean(0)
    W1 = np.linalg.solve(A2, XA.T @ (Yfirst - mu1))
    anchors40 = clean[:40]
    cl = np.concatenate([ov for ov, _ in anchors40], 0)
    owner = np.concatenate([[(i, t) for t in range(len(ov))] for i, (ov, _) in enumerate(anchors40)], 0)
    mu, sig = cl.mean(0), cl.std(0) + 1e-6
    tree = cKDTree((cl - mu) / sig)
    test = read(args.test, args.test_n)
    DZ, DR, GT = [], [], []
    f0c = {}
    for ov, acts in test:
        for t in range(1, len(acts) - H, 4):
            z = (ov[t] - mu) / sig
            d, idx = tree.query(z)
            if not (2.0 <= d < 4.0): continue
            i0, t0 = map(int, owner[int(idx)])
            if int(idx) not in f0c:
                f0c[int(idx)] = ((phiS([anchors40[i0][0][max(t0 - 1, 0)], anchors40[i0][0][t0]]) - muS) @ Tmap) @ W1
            DZ.append(z - (cl[int(idx)] - mu) / sig)
            DR.append(((phiS([ov[t - 1], ov[t]]) - muS) @ Tmap) @ W1 - f0c[int(idx)])
            GT.append(acts[t, :6] - anchors40[i0][1][min(t0, len(anchors40[i0][1]) - 1), :6])
    DZ = np.stack(DZ); DR = np.stack(DR); GT = np.stack(GT)
    def ridge6(X, Y, l=1e-2):
        Aa = X.T @ X + l * len(X) * np.eye(X.shape[1]); return np.linalg.solve(Aa, X.T @ Y).T
    G = ridge6(DZ, DR); G_gt = ridge6(DZ, GT)
    Pa = DZ @ G.T; Pb = DZ @ G_gt.T
    pcos = float(np.mean(np.sum(Pa * Pb, 1) /
                         (np.linalg.norm(Pa, axis=1) * np.linalg.norm(Pb, axis=1) + 1e-9)))
    B = G[0:3, POS] * sig[POS][None, :]
    w = np.linalg.eigvalsh((B + B.T) / 2)
    rsv = float(np.linalg.svd(G[3:6, :] * sig[None, :], compute_uv=False)[0])
    print(f"ADAPT {args.tag} operator cosGT={pcos:.2f} poseig={np.round(w,3).tolist()} "
          f"negdef={bool(np.all(w<0))} rotsv={rsv:.3f}", flush=True)

    # ---- closed loop ----
    env = robosuite.make("ToolHang", horizon=4000, **ENV_KWARGS)
    def ov_(o):
        return np.concatenate([o[KM.get(k, k)] for k in OK]).astype(np.float32)
    def chunk(hist):
        f = (phiS(np.stack(hist[-2:])) - muS) @ Tmap
        an = (f @ Wc + mu_c).reshape(1, AS, 10)
        return ds.undo_transform_action(na.unnormalize(an))[0]
    succ = 0; N = 0; ep = []
    for sd in range(args.seed_lo, args.seed_hi):
        np.random.seed(sd); env.reset()
        arm = env.robots[0].composite_controller.part_controllers["right"]; arm.update(); arm.reset_goal()
        for _ in range(args.settle):
            env.step(np.zeros(7))
        o = env._get_observations(force_update=True); hist = [ov_(o), ov_(o)]
        steps = 0; asm = False; maxd = 0.0
        while steps < args.max_steps and not asm:
            for a in chunk(hist):
                o, _, _, _ = env.step(a); steps += 1; hist.append(ov_(o))
                d, _ = tree.query((hist[-1] - mu) / sig)
                maxd = max(maxd, float(d))
                if env._check_frame_assembled():
                    asm = True; break
                if steps >= args.max_steps:
                    break
        succ += int(asm); N += 1
        ep.append((int(asm), maxd, int(maxd >= 4.0)))
    ep = np.array(ep); c4 = ep[:, 2].astype(bool)
    sr_c4 = 100 * ep[c4, 0].mean() if c4.any() else float("nan")
    sr_n4 = 100 * ep[~c4, 0].mean() if (~c4).any() else float("nan")
    print(f"ADAPT {args.tag} SR seeds[{args.seed_lo},{args.seed_hi}) "
          f"assembled={succ}/{N} = {100*succ/N:.1f}% cross4={c4.mean():.2f} "
          f"SR|cross4={sr_c4:.0f} SR|stay={sr_n4:.0f}")


if __name__ == "__main__":
    main()
