"""ONE-STEP ACTION-PERTURBATION EQUIVALENCE TEST (dynamics-mediated denoising transfer).
For clean anchors (s0, a0): execute a0+eps (pos-dim perturbation) from the EXACT recorded
state -> next state s_eps = s0 + B eps. Measure:
  c_a = dec(g(a0_chunk + eps_chunk, w0)) - dec(g(a0_chunk, w0))      [action-fiber]
  c_s = dec(g(a0_chunk, w_eps)) - dec(g(a0_chunk, w0))               [state-fiber]
and for policies: c_s^MSE, c_s^MIP1 (their drift under the same real deviation).
Predictions (virtual-recovery-supervision):
  (1) c_a ~ -eps (trained removal; residual form)
  (2) c_s DIRECTION ~ -eps (the corrective direction transferred to the s-pathway)
  (3) c_s aligns with the GT-law prediction G_GT * dz(s_eps)
  gain: |c_s|/|eps| tells whether the transfer imprints the label gain (~1) or the
  true servo gain (~K B ~ 0.14).
Also measures B (one-step tracking fraction alpha) directly. Uses env (local)."""
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
from mip.datasets.robomimic_dataset import make_dataset
from mip.agent import TrainingAgent
from mip.samplers import regression_sampler, mip_step1_only_sampler

OK = ["object", "robot0_eef_pos", "robot0_eef_quat", "robot0_gripper_qpos"]
KM = {"object": "object-state"}
H = 16


def load(ckpt, loss):
    cfgdir = os.path.abspath("examples/configs")
    with initialize_config_dir(version_base=None, config_dir=cfgdir):
        cfg = compose(config_name="main", overrides=["task=tool_hang_ph_state_delta_legacy",
            "+task.dataset_path=" + os.path.abspath("data/tool_hang_full2ins_2000.hdf5"),
            "network=chiunet", f"optimization.loss_type={loss}",
            "optimization.auto_resume=false", "log.wandb_mode=disabled"])
    OmegaConf.set_struct(cfg, False); cfg.task.obs_dim = 53
    ds = make_dataset(cfg.task)
    ag = TrainingAgent(cfg); ag.load(ckpt, load_optimizer=False); ag.eval()
    return cfg, ds, ag


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n_anchor", type=int, default=150)
    ap.add_argument("--eps", type=float, default=0.4, help="pos-dim perturbation scale (normalized action units)")
    ap.add_argument("--ksteps", type=int, default=12, help="sustain the perturbation for k control steps (DART-style)")
    args = ap.parse_args()
    cfg, ds, den = load("logs/denoise_only_2k/models/model_latest.pt", "denoise_only")
    _, _, mse = load("logs/full_regression_2000/models/model_latest.pt", "regression")
    _, _, mip = load("logs/full_mip_2000/models/model_latest.pt", "mip")
    dev = cfg.optimization.device; start = cfg.task.obs_steps - 1; AS = cfg.task.act_steps
    no = ds.normalizer["obs"]["state"]; na = ds.normalizer["action"]; optim = cfg.optimization
    TAU = float(optim.t_two_step)
    A10 = ds.replay_buffer["action"][:]
    ends = np.asarray(ds.replay_buffer.episode_ends[:]); stt = np.concatenate([[0], ends[:-1]])

    h = h5py.File("data/tool_hang_full2ins_2000.hdf5", "r")
    demos = []
    for i in range(40):
        d = h[f"data/demo_{i}"]
        ov = np.concatenate([np.asarray(d["obs"][k]) for k in OK], axis=1).astype(np.float32)
        demos.append((ov, np.clip(np.asarray(d["actions"]), -1, 1), np.asarray(d["states"])))
    h.close()
    cl = np.concatenate([ov for ov, _, _ in demos], 0)
    mu, sig = cl.mean(0), cl.std(0) + 1e-6

    # G_GT (normalized-action / z-input) from the saved operator fit for prediction (3)
    OF = np.load("analysis/recovery/operator_fit_b24.npz")
    G_gt = OF["G_gt"]

    env = robosuite.make("ToolHang", horizon=4000, **ENV_KWARGS)
    def ov_of(o): return np.concatenate([o[KM.get(k, k)] for k in OK]).astype(np.float32)

    def chunk_n(i, t0):
        seg = A10[stt[i] + t0: stt[i] + t0 + H]
        return None if len(seg) < H else torch.tensor(na.normalize(seg)[None], device=dev, dtype=torch.float32)

    def win_t(f0, f1):
        return torch.tensor(no.normalize(np.stack([f0, f1])[None]), device=dev, dtype=torch.float32)

    def g_den(a_chunk, x):
        with torch.no_grad():
            with den._inference_mode():
                emb = den.encoder_ema({"state": x}, None)
                return den.flow_map_ema.get_velocity(torch.full((1,), TAU, device=dev), a_chunk, emb)

    def pol(ag, sampler, x):
        with torch.no_grad():
            with ag._inference_mode():
                return sampler(optim, ag.flow_map_ema, ag.encoder_ema, torch.zeros((1, H, 10), device=dev), {"state": x})

    def dec(an):
        return ds.undo_transform_action(na.unnormalize(an.detach().cpu().numpy())[:, start:start + AS])[0][0, :6]

    rng = np.random.RandomState(0)
    CA, CS, CSm, CSp, EPS, DPOSE, GTP, DZN = [], [], [], [], [], [], [], []
    n_done = 0
    while n_done < args.n_anchor:
        i = rng.randint(0, 40); ovi, acts, states = demos[i]
        t0 = rng.randint(2, len(acts) - H - 2)
        c0 = chunk_n(i, t0)
        if c0 is None: continue
        a0 = acts[t0].copy()
        eps3 = args.eps * rng.randn(3)
        K = args.ksteps
        if t0 + K + 1 >= len(acts) - H: continue
        # --- env: sustain the perturbation for K steps (DART-style) ---
        env.reset()
        env.sim.set_state_from_flattened(states[t0]); env.sim.forward()
        arm = env.robots[0].composite_controller.part_controllers["right"]; arm.update(); arm.reset_goal()
        traj_p = []
        for j in range(K):
            aj = acts[t0 + j].copy(); aj[:3] = np.clip(aj[:3] + eps3, -1, 1)
            o, _, _, _ = env.step(aj); traj_p.append(ov_of(o))
        s_eps_prev, s_eps = traj_p[-2], traj_p[-1]
        # nominal rollout for baseline
        env.reset(); env.sim.set_state_from_flattened(states[t0]); env.sim.forward()
        arm = env.robots[0].composite_controller.part_controllers["right"]; arm.update(); arm.reset_goal()
        traj_n = []
        for j in range(K):
            o2, _, _, _ = env.step(acts[t0 + j]); traj_n.append(ov_of(o2))
        s_nom_prev, s_nom = traj_n[-2], traj_n[-1]
        eps_eff = eps3
        dpose = (s_eps[44:47] - s_nom[44:47])
        # anchor chunk at the phase-matched nominal time t0+K
        c0 = chunk_n(i, t0 + K)
        if c0 is None: continue
        a0 = acts[t0 + K].copy()
        # --- probes ---
        w0 = win_t(s_nom_prev, s_nom)
        w_eps = win_t(s_eps_prev, s_eps)
        w_nom = w0
        # action-fiber: perturb first executed action's pos dims in chunk space
        c_pert = c0.clone()
        a_pert = a0.copy(); a_pert[:3] = np.clip(a_pert[:3] + eps3, -1, 1)
        pn = na.normalize(np.concatenate([a_pert[:3], A10[stt[i] + t0 + K][3:]])[None])[0]
        c_pert[0, start, 0:3] = torch.tensor(pn[0:3], device=dev)
        ca = dec(g_den(c_pert, w0)) - dec(g_den(c0, w0))
        cs = dec(g_den(c0, w_eps)) - dec(g_den(c0, w_nom))
        csm = dec(pol(mse, regression_sampler, w_eps)) - dec(pol(mse, regression_sampler, w_nom))
        csp = dec(pol(mip, mip_step1_only_sampler, w_eps)) - dec(pol(mip, mip_step1_only_sampler, w_nom))
        dz = (s_eps - mu) / sig - (s_nom - mu) / sig
        DPOSE[-1:] = DPOSE[-1:]  # noop
        gtp = (G_gt @ dz)                                   # GT-law predicted correction (normalized 6d)
        CA.append(ca); CS.append(cs); CSm.append(csm); CSp.append(csp)
        EPS.append(eps_eff); DPOSE.append(dpose); GTP.append(gtp); DZN.append(float(np.linalg.norm(dz)))
        n_done += 1
    CA = np.stack(CA); CS = np.stack(CS); CSm = np.stack(CSm); CSp = np.stack(CSp)
    EPS = np.stack(EPS); DPOSE = np.stack(DPOSE); GTP = np.stack(GTP); DZN = np.array(DZN)

    def cosw(A, B):
        return float(np.mean(np.sum(A * B, 1) / (np.linalg.norm(A, axis=1) * np.linalg.norm(B, axis=1) + 1e-9)))
    E6 = np.concatenate([EPS, np.zeros_like(EPS)], 1)      # eps embedded in 6-dim action space
    print(f"anchors: {len(CA)}   eps {args.eps} sustained {args.ksteps} steps")
    print(f"achieved pose deviation |dpose| mean = {np.linalg.norm(DPOSE,axis=1).mean()*100:.1f} cm")
    print(f"B (per-step tracking fraction over {args.ksteps} steps): "
          f"{np.mean(np.linalg.norm(DPOSE,axis=1)/(args.ksteps*0.05*np.linalg.norm(EPS,axis=1)+1e-9)):.2f}")
    print(f"\n(1) action-fiber removal: cos(c_a, -eps) = {cosw(CA, -E6):.2f}   "
          f"|c_a|/|eps| = {np.mean(np.linalg.norm(CA,axis=1)/np.linalg.norm(EPS,axis=1)):.2f}")
    print(f"(2) state-fiber correction, denoiser: cos(c_s, -eps) = {cosw(CS, -E6):.2f}   "
          f"|c_s|/|eps| = {np.mean(np.linalg.norm(CS,axis=1)/np.linalg.norm(EPS,axis=1)):.2f}")
    print(f"(3) cos(c_s, GT-law prediction)      = {cosw(CS, GTP):.2f}   "
          f"|c_s|/|GT-pred| = {np.mean(np.linalg.norm(CS,axis=1)/(np.linalg.norm(GTP,axis=1)+1e-9)):.2f}")
    print(f"\ncontrols under the SAME real deviation:")
    print(f"  MIP-step1: cos(c_s,-eps) = {cosw(CSp, -E6):.2f}  cos(c_s,GT) = {cosw(CSp, GTP):.2f}  |c|/|eps| = {np.mean(np.linalg.norm(CSp,axis=1)/np.linalg.norm(EPS,axis=1)):.2f}")
    print(f"  MSE      : cos(c_s,-eps) = {cosw(CSm, -E6):.2f}  cos(c_s,GT) = {cosw(CSm, GTP):.2f}  |c|/|eps| = {np.mean(np.linalg.norm(CSm,axis=1)/np.linalg.norm(EPS,axis=1)):.2f}")
    print(f"\nachieved |dz|: p25/p50/p75 = {np.percentile(DZN,[25,50,75])}")
    def cosr(A,B,m):
        return float(np.mean(np.sum(A[m]*B[m],1)/(np.linalg.norm(A[m],axis=1)*np.linalg.norm(B[m],axis=1)+1e-9)))
    print(f"{'band':10} {'N':>4} | {'den cos-eps':>11} {'den cosGT':>9} | {'MIP cosGT':>9} | {'MSE cosGT':>9}")
    for lo,hi in [(0,1.5),(1.5,2.5),(2.5,6)]:
        m=(DZN>=lo)&(DZN<hi)
        if m.sum()<10: print(f"[{lo},{hi})   {m.sum():>4}  -- few"); continue
        print(f"[{lo},{hi})   {int(m.sum()):>4} | {cosr(CS,-E6,m):>11.2f} {cosr(CS,GTP,m):>9.2f} | {cosr(CSp,GTP,m):>9.2f} | {cosr(CSm,GTP,m):>9.2f}")
    np.savez("analysis/recovery/perturb_equiv.npz", CA=CA, CS=CS, CSm=CSm, CSp=CSp, EPS=EPS, DPOSE=DPOSE, GTP=GTP, DZN=DZN)
    print("saved analysis/recovery/perturb_equiv.npz")


if __name__ == "__main__":
    main()
