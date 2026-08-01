"""EXP-A6: POLICY_VISITED_OPERATOR. Fit the model's off-support response operator on
states the POLICY ITSELF visits (rollouts, d in [2,4) vs the 40-demo clean anchor cloud),
instead of DART-generated test states. Compare direction with the DART-fit G_GT
(transported reference from badmode_ref.npz) and report bad-mode projections.
Procedure: roll the policy on held-out seeds, collect (s_{t-1}, s_t) windows with
d(s_t) in [2,4); model response DR = f(s) - f(anchor); fit G on (dz, DR);
report R2, cos vs G_GT (applied to the SAME visited dz), pos-block eigs, rot metrics.
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
    ap.add_argument("--ckpt", required=True); ap.add_argument("--loss", required=True)
    ap.add_argument("--tag", default="")
    ap.add_argument("--clean", default="data/tool_hang_full2ins_2000.hdf5")
    ap.add_argument("--ref", default="analysis/recovery/badmode_ref.npz")
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

    clean = read(args.clean, 40)
    cl = np.concatenate([ov for ov, _ in clean], 0)
    owner = np.concatenate([[(i, t) for t in range(len(ov))] for i, (ov, _) in enumerate(clean)], 0)
    mu, sig = cl.mean(0), cl.std(0) + 1e-6
    tree = cKDTree((cl - mu) / sig)
    ref = np.load(args.ref)
    G_gt = ref["G_gt"]

    def respond(w0, w1):
        x = torch.tensor(no.normalize(np.stack([w0, w1])[None]), device=dev, dtype=torch.float32)
        with torch.no_grad():
            with ag._inference_mode():
                emb = ag.encoder_ema({"state": x}, None)
                an = ag.flow_map_ema.get_velocity(torch.zeros(1, device=dev),
                                                  torch.zeros((1, H, 10), device=dev), emb)
        return ds.undo_transform_action(na.unnormalize(an.detach().cpu().numpy())[:, start:start + AS])[0][0, :6]

    def chunk(hist):
        w = np.stack(hist[-2:])[None]
        ot = {"state": torch.tensor(no.normalize(w), device=dev, dtype=torch.float32)}
        with torch.no_grad():
            an = ag.sample(act_0=torch.randn((1, H, 10), device=dev), obs=ot, use_ema=True)
        return ds.undo_transform_action(na.unnormalize(an.detach().cpu().numpy())[:, start:start + AS])[0]

    def ov_(o):
        return np.concatenate([o[KM.get(k, k)] for k in OK]).astype(np.float32)

    # ---- rollouts: collect visited off-support windows ----
    env = robosuite.make("ToolHang", horizon=4000, **ENV_KWARGS)
    visited = []   # (w_prev, w_cur, d, idx)
    succ = 0; N = 0
    for sd in range(args.seed_lo, args.seed_hi):
        np.random.seed(sd); env.reset()
        arm = env.robots[0].composite_controller.part_controllers["right"]; arm.update(); arm.reset_goal()
        for _ in range(args.settle):
            env.step(np.zeros(7))
        o = env._get_observations(force_update=True); hist = [ov_(o), ov_(o)]
        steps = 0; asm = False
        while steps < args.max_steps and not asm:
            for a in chunk(hist):
                o, _, _, _ = env.step(a); steps += 1; hist.append(ov_(o))
                z = (hist[-1] - mu) / sig
                d, idx = tree.query(z)
                if 2.0 <= d < 4.0:
                    visited.append((hist[-2].copy(), hist[-1].copy(), float(d), int(idx)))
                if env._check_frame_assembled():
                    asm = True; break
                if steps >= args.max_steps:
                    break
        succ += int(asm); N += 1
    print(f"VISITED {args.tag} SR={succ}/{N} n_offsupport_windows={len(visited)}", flush=True)

    # subsample for probe cost
    rng = np.random.RandomState(0)
    if len(visited) > 6000:
        visited = [visited[i] for i in rng.choice(len(visited), 6000, replace=False)]

    DZ, DR = [], []
    nb = {}
    for w0, w1, d, idx in visited:
        i0, t0 = map(int, owner[idx])
        if idx not in nb:
            nb[idx] = respond(clean[i0][0][max(t0 - 1, 0)], clean[i0][0][t0])
        DZ.append((w1 - mu) / sig - (cl[idx] - mu) / sig)
        DR.append(respond(w0, w1) - nb[idx])
    DZ = np.stack(DZ); DR = np.stack(DR)

    def ridge(X, Y, l=1e-2):
        A = X.T @ X + l * len(X) * np.eye(X.shape[1]); return np.linalg.solve(A, X.T @ Y).T
    G = ridge(DZ, DR)
    Pa = DZ @ G.T; Pb = DZ @ G_gt.T
    pcos = float(np.mean(np.sum(Pa * Pb, 1) /
                         (np.linalg.norm(Pa, axis=1) * np.linalg.norm(Pb, axis=1) + 1e-9)))
    r2 = 1 - np.sum((DR - Pa) ** 2) / np.sum((DR - DR.mean(0)) ** 2)
    B = G[0:3, POS] * sig[POS][None, :]
    w = np.linalg.eigvalsh((B + B.T) / 2)
    # bad-mode projections vs reference MSE modes
    def posblk(Gx):
        Bx = Gx[0:3, POS] * ref["sig"][POS][None, :]
        return (Bx + Bx.T) / 2
    S_mse = posblk(ref["G_m"])
    wm, Vm = np.linalg.eigh(S_mse)
    bad1 = Vm[:, np.argmax(wm)]
    R_mse = ref["G_m"][3:6, :] * ref["sig"][None, :]
    vrot = np.linalg.svd(R_mse)[2][0]
    S = posblk(G); R = G[3:6, :] * ref["sig"][None, :]
    print(f"VISITED {args.tag} N={len(DZ)} R2={r2:.3f} cosGT={pcos:.2f} "
          f"poseig={np.round(w,3).tolist()} negdef={bool(np.all(w<0))} "
          f"gain@bad1={float(bad1 @ S @ bad1):+.4f} rot@mseamp={float(np.linalg.norm(R @ vrot)):.3f} "
          f"rotsv={float(np.linalg.svd(R, compute_uv=False)[0]):.3f}")
    np.savez(os.path.join(os.path.dirname(args.ckpt), f"visited_{args.tag}.npz"), DZ=DZ, DR=DR, G=G)


if __name__ == "__main__":
    main()
