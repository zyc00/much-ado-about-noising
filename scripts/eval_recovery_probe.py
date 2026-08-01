"""Step A — OOD recovery + action-prior probe (tests WHY MIP recovers, MSE doesn't).

Natural closed-loop rollout from init. At every policy prediction step record:
  (1) obs_dist  = z-scored cKDTree NN distance of the CURRENT obs to the expert
                  OBS manifold (how far off-manifold the state has drifted)
  (2) act_dist  = z-scored cKDTree NN distance of the PREDICTED action chunk to the
                  expert ACTION manifold (is the policy's output a valid expert-like
                  action, or did it drift to a nonsense action?)
plus per-episode obs_dist trajectory + assembled outcome.

Two questions:
  H1 (action prior): does MIP keep act_dist LOW even when obs_dist is HIGH (OOD)?
     i.e. is MIP's output pulled onto the action manifold (generative prior),
     while MSE's action drifts with the obs? -> plot act_dist vs obs_dist bins.
  Recovery: among steps where obs_dist crosses tau_hi, what fraction RETURN below
     tau_lo (recover) vs run away to tau_div (diverge)? -> recovery rate.

Saves npz of (obs_dist, act_dist) pairs + per-episode curves for plotting.

  MUJOCO_GL=egl python scripts/eval_recovery_probe.py --ckpt ... --dataset ... --loss mip \
     --demos data/warmstart_demos.hdf5 --n 50 --manifold_n 40 --out_npz mip.npz --tag MIP
"""
import os
os.environ["MUJOCO_GL"] = "egl"
import argparse
import numpy as np
import torch
import h5py
import robosuite
import sys
sys.path.insert(0, "scripts"); sys.path.insert(0, ".")
from collect_tool_hang_demos import ENV_KWARGS
from hydra import initialize_config_dir, compose
from omegaconf import OmegaConf
from scipy.spatial import cKDTree
from mip.datasets.robomimic_dataset import make_dataset
from mip.agent import TrainingAgent

OK = ["object", "robot0_eef_pos", "robot0_eef_quat", "robot0_gripper_qpos"]
KM = {"object": "object-state"}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--dataset", required=True)
    ap.add_argument("--loss", default="regression")
    ap.add_argument("--demos", default="data/warmstart_demos.hdf5")
    ap.add_argument("--n", type=int, default=50)
    ap.add_argument("--manifold_n", type=int, default=40)
    ap.add_argument("--max_steps", type=int, default=700)
    ap.add_argument("--H", type=int, default=16)
    ap.add_argument("--out_npz", default=None)
    ap.add_argument("--tag", default="")
    args = ap.parse_args()

    cfgdir = os.path.abspath("examples/configs")
    with initialize_config_dir(version_base=None, config_dir=cfgdir):
        cfg = compose(config_name="main", overrides=[
            "task=tool_hang_ph_state_delta_legacy", f"+task.dataset_path={os.path.abspath(args.dataset)}",
            "network=chiunet", f"optimization.loss_type={args.loss}",
            "optimization.auto_resume=false", "log.wandb_mode=disabled"])
    OmegaConf.set_struct(cfg, False); cfg.task.obs_dim = 53
    dev = cfg.optimization.device
    AS = cfg.task.act_steps; start = cfg.task.obs_steps - 1
    ds = make_dataset(cfg.task)
    agent = TrainingAgent(cfg); agent.load(args.ckpt, load_optimizer=False); agent.eval()
    no = ds.normalizer["obs"]["state"]; na = ds.normalizer["action"]
    df = h5py.File(args.demos, "r")
    allk = list(df["demos"].keys())
    eval_keys = allk[:args.n]
    man_keys = allk[args.n:args.n + args.manifold_n] or allk[:args.manifold_n]
    env = robosuite.make("ToolHang", horizon=4000, **ENV_KWARGS)

    def ov(o):
        return np.concatenate([o[KM.get(k, k)] for k in OK]).astype(np.float32)

    def chunk(hist):
        w = np.stack(hist[-2:])[None]
        ot = {"state": torch.tensor(no.normalize(w), device=dev, dtype=torch.float32)}
        with torch.no_grad():
            an = agent.sample(act_0=torch.randn((1, args.H, 10), device=dev), obs=ot, use_ema=True)
        return ds.undo_transform_action(na.unnormalize(an.detach().cpu().numpy())[:, start:start + AS])[0]

    # ---- build expert OBS manifold + ACTION-chunk manifold (held-out replays) ----
    man_obs, man_act = [], []
    for k in man_keys:
        d = df["demos/" + k]; sd = int(d.attrs["seed"]); acts = np.clip(d["actions"][:], -1, 1)
        s0 = d["state0"][:]
        np.random.seed(sd); env.reset(); env.sim.set_state_from_flattened(s0); env.sim.forward()
        arm = env.robots[0].composite_controller.part_controllers["right"]; arm.update(); arm.reset_goal()
        man_obs.append(ov(env._get_observations(force_update=True)))
        T = len(acts)
        for t in range(T):
            o, _, _, _ = env.step(acts[t]); man_obs.append(ov(o))
            if t + AS <= T:
                man_act.append(acts[t:t + AS].reshape(-1))   # (AS*7,) raw expert chunk
    man_obs = np.array(man_obs); man_act = np.array(man_act)
    omu, osig = man_obs.mean(0), man_obs.std(0) + 1e-6
    amu, asig = man_act.mean(0), man_act.std(0) + 1e-6
    otree = cKDTree((man_obs - omu) / osig)
    atree = cKDTree((man_act - amu) / asig)
    print(f"[manifold] obs={len(man_obs)} act={len(man_act)} (chunk dim {man_act.shape[1]})", flush=True)

    pairs = []          # (obs_dist, act_dist) at each prediction step
    ep_curves = []      # per-episode obs_dist trajectory + success
    n_succ = 0
    for k in eval_keys:
        d = df["demos/" + k]; sd = int(d.attrs["seed"])
        np.random.seed(sd); env.reset()
        arm = env.robots[0].composite_controller.part_controllers["right"]; arm.update(); arm.reset_goal()
        for _ in range(10):
            env.step(np.zeros(7))
        o = env._get_observations(force_update=True); hist = [ov(o), ov(o)]
        dists = []; steps = 0; asm = False
        while steps < args.max_steps and not asm:
            cur = hist[-1]
            od, _ = otree.query((cur - omu) / osig)
            ch = chunk(hist)                                   # (AS,7) raw predicted
            adst, _ = atree.query((ch.reshape(-1) - amu) / asig)
            pairs.append((float(od), float(adst)))
            for a in ch:
                o, _, _, _ = env.step(a); steps += 1
                v = ov(o); hist.append(v)
                dists.append(float(otree.query((v - omu) / osig)[0]))
                if env._check_frame_assembled():
                    asm = True; break
                if steps >= args.max_steps:
                    break
        ep_curves.append((np.array(dists), bool(asm)))
        n_succ += int(asm)
    df.close()
    pairs = np.array(pairs)

    # ---- H1: action-prior — act_dist binned by obs_dist ----
    od_all, ad_all = pairs[:, 0], pairs[:, 1]
    print(f"RECOVERY {args.tag} n_ep={len(ep_curves)} success={n_succ}/{len(ep_curves)} n_steps={len(pairs)}")
    edges = [0, 1, 2, 3, 5, 10, 1e9]
    print("  act_dist | obs_dist bin:")
    for lo, hi in zip(edges[:-1], edges[1:]):
        m = (od_all >= lo) & (od_all < hi)
        if m.sum():
            print(f"    obs_dist[{lo:>4},{hi if hi < 1e8 else 'inf':>4}) n={m.sum():5d}  act_dist mean={ad_all[m].mean():.2f} med={np.median(ad_all[m]):.2f}")

    # ---- Recovery: among drift-onset crossings, recover vs diverge ----
    TAU_HI, TAU_LO, TAU_DIV, W = 2.0, 1.2, 5.0, 12
    n_onset = n_recover = n_diverge = 0
    for curve, _ in ep_curves:
        i = 0
        while i < len(curve):
            if curve[i] >= TAU_HI:
                n_onset += 1
                win = curve[i:i + W]
                if win.min() < TAU_LO:
                    n_recover += 1; i += int(np.argmin(win)) + 1
                elif win.max() > TAU_DIV:
                    n_diverge += 1; i += W
                else:
                    i += W
            else:
                i += 1
    rr = 100 * n_recover / max(n_onset, 1)
    print(f"  DRIFT-ONSET(>{TAU_HI}) n={n_onset}  RECOVER(<{TAU_LO} within {W})={n_recover} ({rr:.0f}%)  "
          f"DIVERGE(>{TAU_DIV})={n_diverge}")
    if args.out_npz:
        np.savez(args.out_npz, pairs=pairs, n_succ=n_succ, n_ep=len(ep_curves),
                 onset=n_onset, recover=n_recover, diverge=n_diverge, tag=args.tag,
                 curves=np.array([c for c, _ in ep_curves], dtype=object),
                 succ=np.array([s for _, s in ep_curves]))
        print(f"  saved {args.out_npz}")


if __name__ == "__main__":
    main()
