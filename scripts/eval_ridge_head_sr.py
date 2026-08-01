"""Replace the trained final head with a clean-data ridge readout on FROZEN penultimate
features (view t=0, zeros action input) and deploy it closed-loop.
Tests whether MSE's escape pathology lives in the head or in the features: the feature-ridge
probe showed MSE-phi0 linearly decodes a weakly corrective NEGATIVE-definite operator
(cos 0.56) while the full MSE model's operator is indefinite (+0.021 escape direction).
Ridge target = the 8 executed normalized 10-d actions (slots start:start+AS of the chunk),
so deployment reuses the exact unnormalize/undo_transform pipeline of eval_seedset."""
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
from mip.datasets.robomimic_dataset import make_dataset
from mip.agent import TrainingAgent

OK = ["object", "robot0_eef_pos", "robot0_eef_quat", "robot0_gripper_qpos"]
KM = {"object": "object-state"}
H = 16


def read_obs(path, nmax):
    # replay-buffer order: demo_{i} for i in range(N) (numeric, NOT alphabetical h5 order)
    h = h5py.File(path, "r"); g = "data" if "data" in h else "demos"
    out = []
    for i in range(min(nmax, len(h[g]))):
        o = h[f"{g}/demo_{i}/obs"]
        out.append(np.concatenate([np.asarray(o[key]) for key in OK], axis=1).astype(np.float32))
    h.close(); return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", required=True); ap.add_argument("--loss", required=True)
    ap.add_argument("--dataset", default="data/tool_hang_full2ins_2000.hdf5")
    ap.add_argument("--train_demos", type=int, default=200)
    ap.add_argument("--stride", type=int, default=6)
    ap.add_argument("--lam", type=float, default=1e-3)
    ap.add_argument("--seed_lo", type=int, default=21000); ap.add_argument("--seed_hi", type=int, default=21100)
    ap.add_argument("--settle", type=int, default=10); ap.add_argument("--max_steps", type=int, default=700)
    ap.add_argument("--tag", default="")
    ap.add_argument("--random_init", action="store_true",
                    help="deploy ridge head on UNTRAINED network features (random-kernel closed-loop control)")
    ap.add_argument("--view", default="t0", choices=["t0", "tau"], help="feature-extraction view")
    args = ap.parse_args()
    cfgdir = os.path.abspath("examples/configs")
    with initialize_config_dir(version_base=None, config_dir=cfgdir):
        cfg = compose(config_name="main", overrides=["task=tool_hang_ph_state_delta_legacy",
            f"+task.dataset_path={os.path.abspath(args.dataset)}", "network=chiunet",
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
    def hook(mod, inp):
        feats["x"] = inp[0].detach()
    last.register_forward_pre_hook(hook)

    def phi(win):  # win: (2, 53) raw obs window [prev, cur]
        x = torch.tensor(no.normalize(win[None]), device=dev, dtype=torch.float32)
        with torch.no_grad():
            with ag._inference_mode():
                emb = ag.encoder_ema({"state": x}, None)
                ag.flow_map_ema.get_velocity(torch.full((1,), TAUV, device=dev),
                                             torch.zeros((1, H, 10), device=dev), emb)
        return feats["x"].reshape(-1).cpu().numpy()

    # ---- fit ridge head: phi(t=0, zeros, s) -> executed normalized action slots ----
    obs_demos = read_obs(args.dataset, args.train_demos)
    Xtr, Ytr = [], []
    for i, ov in enumerate(obs_demos):
        T = ends[i] - stt[i]
        assert T == len(ov), f"episode order mismatch at {i}: replay {T} vs h5 {len(ov)}"
        for t in range(1, T - (H - 1), args.stride):
            # chunk covering times [t-1 .. t+14]; executed = slots [start:start+AS] = times [t..t+7]
            seg = na.normalize(A10[stt[i] + t - 1: stt[i] + t - 1 + H])[start:start + AS]
            Xtr.append(phi(np.stack([ov[t - 1], ov[t]]))); Ytr.append(seg.reshape(-1))
    Xtr = np.stack(Xtr).astype(np.float64); Ytr = np.stack(Ytr).astype(np.float64)
    mu_f = Xtr.mean(0); Xc = Xtr - mu_f
    mu_y = Ytr.mean(0); Yc = Ytr - mu_y   # intercept: fit centered, add mu_y at deploy
    A = Xc.T @ Xc + args.lam * len(Xc) * np.eye(Xc.shape[1])
    Wr = np.linalg.solve(A, Xc.T @ Yc)
    r2 = 1 - np.sum((Xc @ Wr - Yc) ** 2) / np.sum(Yc ** 2)
    print(f"RIDGEHEAD {args.tag} fit N={len(Xtr)} trainR2={r2:.3f}", flush=True)

    # ---- support-distance tracker (standard 40-anchor cloud) for PNR crossing ----
    from scipy.spatial import cKDTree
    anch = obs_demos[:40]
    cl = np.concatenate(anch, 0)
    mu_c, sig_c = cl.mean(0), cl.std(0) + 1e-6
    tree = cKDTree((cl - mu_c) / sig_c)

    # ---- closed-loop deployment with the ridge head ----
    env = robosuite.make("ToolHang", horizon=4000, **ENV_KWARGS)

    def ov_(o):
        return np.concatenate([o[KM.get(k, k)] for k in OK]).astype(np.float32)

    def chunk(hist):
        f = phi(np.stack(hist[-2:]))
        an = ((f - mu_f) @ Wr + mu_y).reshape(1, AS, 10)
        return ds.undo_transform_action(na.unnormalize(an))[0]

    succ = 0; N = 0; ep = []   # (success, maxd, crossed2, crossed4)
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
                d, _ = tree.query((hist[-1] - mu_c) / sig_c)
                maxd = max(maxd, float(d))
                if env._check_frame_assembled():
                    asm = True; break
                if steps >= args.max_steps:
                    break
        succ += int(asm); N += 1
        ep.append((int(asm), maxd, int(maxd >= 2.0), int(maxd >= 4.0)))
    ep = np.array(ep)
    c2, c4 = ep[:, 2].astype(bool), ep[:, 3].astype(bool)
    sr_c4 = 100 * ep[c4, 0].mean() if c4.any() else float("nan")
    sr_n4 = 100 * ep[~c4, 0].mean() if (~c4).any() else float("nan")
    print(f"RIDGEHEAD {args.tag} seeds[{args.seed_lo},{args.seed_hi}) "
          f"assembled={succ}/{N} = {100*succ/N:.1f}% | cross2={c2.mean():.2f} "
          f"cross4={c4.mean():.2f} SR|cross4={sr_c4:.0f} SR|stay={sr_n4:.0f} "
          f"maxd_p50={np.median(ep[:,1]):.2f}")


if __name__ == "__main__":
    main()
