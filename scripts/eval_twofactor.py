"""EXP-2: ROLLOUT_PRECISION_TWO_FACTOR_DIRECT_TEST.
Closed-loop rollout with per-step logging:
  d_t (support distance), executed action a_hat (6d), GT proxy
  a_GT(s) = clip(a_anchor(NN) + G_GT (z - z_anchor), -1, 1)   (first-order servo proxy)
Reports per model:
  - SR, cross2/cross4, SR|cross4, SR|stay, maxd p50/90/95, firstcross4 p50/90
  - d<2 band: |a_hat-a_GT| mean/p50/p90/p95 (total/pos/rot), dd stats, frac(dd>0)
  - [2,4) band: |a_hat-a_GT| mean/p50/p90, dd stats,
    excursion outcomes (entries into [2,4) from below): frac return to d<2 within
    5/10/20 steps, frac reach d>=4 within 5/10/20 steps
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
H = 16


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", required=True); ap.add_argument("--loss", required=True)
    ap.add_argument("--tag", default="")
    ap.add_argument("--dataset", default="data/tool_hang_full2ins_2000.hdf5")
    ap.add_argument("--ref", default="analysis/recovery/badmode_ref.npz")
    ap.add_argument("--seed_lo", type=int, default=21000); ap.add_argument("--seed_hi", type=int, default=21100)
    ap.add_argument("--settle", type=int, default=10); ap.add_argument("--max_steps", type=int, default=700)
    ap.add_argument("--H", type=int, default=16)
    ap.add_argument("--act_dim", type=int, default=10)
    ap.add_argument("--aux_flag", default="")
    ap.add_argument("--randaug", action="store_true")
    ap.add_argument("--AS", type=int, default=0)
    ap.add_argument("--net", default="chiunet")
    ap.add_argument("--ens", default="")
    args = ap.parse_args()
    cfgdir = os.path.abspath("examples/configs")
    with initialize_config_dir(version_base=None, config_dir=cfgdir):
        cfg = compose(config_name="main", overrides=["task=tool_hang_ph_state_delta_legacy",
            f"+task.dataset_path={os.path.abspath(args.dataset)}", f"network={args.net}",
            f"optimization.loss_type={args.loss}", "optimization.auto_resume=false",
            "log.wandb_mode=disabled"]
            + [o for o in os.environ.get("EV_OVERRIDES", "").split() if o])
    OmegaConf.set_struct(cfg, False); cfg.task.obs_dim = 53; cfg.task.horizon = args.H
    if args.act_dim != 10:
        cfg.task.act_dim = args.act_dim
        if args.aux_flag:
            for f in args.aux_flag.split(","):
                setattr(cfg.task, f, True)
        elif args.act_dim == 13:
            cfg.task.phase_indicator = True
        elif args.act_dim == 11:
            cfg.task.progress_indicator = True
    ds = make_dataset(cfg.task)
    agent = TrainingAgent(cfg); agent.load(args.ckpt, load_optimizer=False); agent.eval()
    ens_agents = [agent]
    for p in [q for q in args.ens.split(",") if q]:
        a2 = TrainingAgent(cfg); a2.load(p, load_optimizer=False); a2.eval()
        ens_agents.append(a2)
    dev = cfg.optimization.device; AS = args.AS if args.AS > 0 else cfg.task.act_steps
    start = cfg.task.obs_steps - 1
    no = ds.normalizer["obs"]["state"]; na = ds.normalizer["action"]

    # anchors: 40 clean demos (obs + actions), numeric order
    h = h5py.File(args.dataset, "r"); g = "data" if "data" in h else "demos"
    anc_o, anc_a = [], []
    for i in range(40):
        o = h[f"{g}/demo_{i}/obs"]
        anc_o.append(np.concatenate([np.asarray(o[k]) for k in OK], axis=1).astype(np.float32))
        anc_a.append(np.clip(np.asarray(h[f"{g}/demo_{i}/actions"]), -1, 1).astype(np.float32))
    h.close()
    cl = np.concatenate(anc_o, 0)
    cla = np.concatenate([a[:len(o)] if len(a) >= len(o) else np.pad(a, ((0, len(o) - len(a)), (0, 0)), "edge")
                          for o, a in zip(anc_o, anc_a)], 0)[:, :6]
    mu, sig = cl.mean(0), cl.std(0) + 1e-6
    tree = cKDTree((cl - mu) / sig)
    ref = np.load(args.ref); G_gt = ref["G_gt"]

    def ov(o):
        return np.concatenate([o[KM.get(k, k)] for k in OK]).astype(np.float32)

    def chunk(hist):
        w = np.stack(hist[-int(cfg.task.obs_steps):])[None]
        ot = {"state": torch.tensor(no.normalize(w), device=dev, dtype=torch.float32)}
        with torch.no_grad():
            a0 = torch.randn((1, args.H, args.act_dim), device=dev)
            an = torch.stack([ag_.sample(act_0=a0, obs=ot, use_ema=True) for ag_ in ens_agents]).mean(0)
            if args.randaug:
                from mip.losses import randaug_offset
                an = an - randaug_offset(ot, an)
        return ds.undo_transform_action(na.unnormalize(an.detach().cpu().numpy())[:, start:start + AS])[0]

    _bl = os.environ.get("BLEND2", "")
    _bl_agent = None
    if _bl:
        # dual-policy blend: execute (1-w)*main + w*second in RAW action
        # space. BLEND2="ckpt:loss:w". Second agent gets its OWN sampler.
        _bck, _bloss, _bw = _bl.split(":")
        _bw = float(_bw)
        with initialize_config_dir(version_base=None, config_dir=cfgdir):
            _bcfg = compose(config_name="main", overrides=[
                "task=tool_hang_ph_state_delta_legacy",
                f"+task.dataset_path={os.path.abspath(args.dataset)}",
                f"network={args.net}", f"optimization.loss_type={_bloss}",
                "optimization.auto_resume=false", "log.wandb_mode=disabled"])
        OmegaConf.set_struct(_bcfg, False)
        _bcfg.task.obs_dim = 53; _bcfg.task.horizon = args.H
        _bl_agent = TrainingAgent(_bcfg)
        _bl_agent.load(_bck, load_optimizer=False); _bl_agent.eval()
        print(f"BLEND2 {_bloss} w={_bw}", flush=True)

    _kx = float(os.environ.get("KNN_EXTRAP", "0"))
    if _kx != 0.0:
        # causal test of the NEIGHBOR-AVERAGING mediator: at execution,
        # extrapolate the policy's pose chunk AWAY from (eta>0) or TOWARD
        # (eta<0) the kNN-average chunk of the training set. a_corr =
        # a + eta*(a - a_knn); retention 0.22 predicts eta* ~ 0.28 for L2.
        from scipy.spatial import cKDTree as _KD
        _hh = h5py.File(args.dataset, "r")
        _nm = sorted(_hh["data"].keys(), key=lambda d: int(d.split("_")[1]))
        _KW, _KA = [], []
        for _dn in _nm:
            _o = _hh[f"data/{_dn}/obs"]
            _S = np.concatenate([np.asarray(_o[k]) for k in [
                "object", "robot0_eef_pos", "robot0_eef_quat",
                "robot0_gripper_qpos"]], 1).astype(np.float32)
            _A = np.asarray(_hh[f"data/{_dn}/actions"], dtype=np.float32)
            for _i in range(1, len(_S) - 9):
                _KW.append(np.stack([_S[_i - 1], _S[_i]]).reshape(-1))
                _KA.append(_A[_i:_i + 8])
        _hh.close()
        _KW, _KA = np.stack(_KW), np.stack(_KA)
        _kmu, _ksd = _KW.mean(0), _KW.std(0) + 1e-6
        _ktree = _KD((_KW - _kmu) / _ksd)
        print(f"KNN_EXTRAP eta={_kx} index {len(_KW)} windows", flush=True)

    env = robosuite.make("ToolHang", horizon=4000, **ENV_KWARGS)
    succ = 0; N = 0; ep = []
    E_lo, E_hi = [], []       # (etot, epos, erot) per step per band
    DD_lo, DD_hi = [], []
    exc = []                  # excursion outcomes: (returned5,10,20, crossed5,10,20)
    for sd in range(args.seed_lo, args.seed_hi):
        np.random.seed(sd); env.reset()
        arm = env.robots[0].composite_controller.part_controllers["right"]; arm.update(); arm.reset_goal()
        for _ in range(args.settle):
            env.step(np.zeros(7))
        o = env._get_observations(force_update=True); hist = [ov(o)] * max(2, int(cfg.task.obs_steps))
        steps = 0; asm = False
        dser, eser = [], []
        pser = [np.asarray(env._get_observations()["robot0_eef_pos"]).copy()]
        oser = []
        while steps < args.max_steps and not asm:
            _ch = chunk(hist)
            if _bl_agent is not None:
                _w2 = np.stack(hist[-2:])[None]
                _ot2 = {"state": torch.tensor(no.normalize(_w2), device=dev,
                                              dtype=torch.float32)}
                with torch.no_grad():
                    _a02 = torch.randn((1, args.H, args.act_dim), device=dev)
                    _an2 = _bl_agent.sample(act_0=_a02, obs=_ot2, use_ema=True)
                _ch2 = ds.undo_transform_action(na.unnormalize(
                    _an2.detach().cpu().numpy())[:, start:start + AS])[0]
                _ch = (1.0 - _bw) * np.asarray(_ch) + _bw * np.asarray(_ch2)
            if _kx != 0.0:
                _q = (np.stack(hist[-2:]).reshape(-1) - _kmu) / _ksd
                _dk, _nk = _ktree.query(_q, k=8)
                _wk = 1.0 / (_dk + 1e-6); _wk = _wk / _wk.sum()
                _aknn = (_KA[_nk] * _wk[:, None, None]).sum(0)[:len(_ch)]
                _c = np.array(_ch, dtype=np.float64).copy()
                _c[:, :6] = _c[:, :6] + _kx * (_c[:, :6] - _aknn[:, :6])
                _ch = _c
            _sm = float(os.environ.get("ACT_SMOOTH", "0"))
            if _sm > 0 and len(_ch) > 1:
                # causal test of the incoherent-error account: low-pass the
                # EXECUTED chunk (pose dims), which removes within-chunk
                # jitter while preserving the coherent component
                _c = np.array(_ch, dtype=np.float64).copy()
                _f = _c[:, :6].copy()
                for _k in range(1, len(_f)):
                    _f[_k] = _sm * _f[_k - 1] + (1 - _sm) * _f[_k]
                _c[:, :6] = _f
                _ch = _c
            _sb = float(os.environ.get("ACT_SBIAS", "0"))
            if _sb > 0:
                # causal test of the SYSTEMATIC (per-chunk bias) error
                # component: a CONSTANT offset over the whole chunk, a
                # deterministic function of the state (as a prediction bias
                # is). Accumulates over the 8 open-loop steps, unlike SJIT.
                _c = np.array(_ch, dtype=np.float64).copy()
                _hh = int(abs(hash(np.round(hist[-1], 3).tobytes())) % (2 ** 31))
                _rs = np.random.RandomState(_hh)
                _b = _rs.randn(6)
                _b = _b / (np.linalg.norm(_b) + 1e-12) * _sb * np.sqrt(6)
                _c[:, :6] = np.clip(_c[:, :6] + _b, -1, 1)
                _ch = _c
            _sj = float(os.environ.get("ACT_SJIT", "0"))
            if _sj > 0:
                # FAITHFUL version of the incoherent-error manipulation: the
                # perturbation is a deterministic FUNCTION OF THE STATE (as a
                # policy's prediction error is), and zero-mean over the chunk
                # so it is purely the incoherent component.
                _c = np.array(_ch, dtype=np.float64).copy()
                _hh = int(abs(hash(np.round(hist[-1], 3).tobytes())) % (2 ** 31))
                _rs = np.random.RandomState(_hh)
                _j = _rs.randn(len(_c), 6) * _sj
                _j = _j - _j.mean(0)
                _c[:, :6] = np.clip(_c[:, :6] + _j, -1, 1)
                _ch = _c
            for a in _ch:
                _an = float(os.environ.get("ACT_NOISE", "0"))
                if _an > 0:
                    # exec-noise probe: iid Gaussian on the 6 pose dims of the
                    # EXECUTED action (gripper untouched), same process at any AS
                    a = a.copy()
                    a[:6] = np.clip(a[:6] + np.random.randn(6) * _an, -1, 1)
                s_cur = hist[-1]
                z = (s_cur - mu) / sig
                d, idx = tree.query(z)
                _tk = float(os.environ.get("ACT_TUBEK", "0"))
                if _tk != 0.0:
                    # eval-time servo graft: subtract k * (eef displacement
                    # from nearest training state), in action units
                    # (ACT_TUBEC m per unit) — tests sufficiency of the
                    # restoring annulus field on a FROZEN policy.
                    a = a.copy()
                    _dsp = (s_cur[44:47] - cl[int(idx)][44:47]) / float(
                        os.environ.get("ACT_TUBEC", "0.05"))
                    a[:3] = np.clip(a[:3] - _tk * _dsp, -1, 1)
                agt = np.clip(cla[int(idx)] + G_gt @ (z - (cl[int(idx)] - mu) / sig), -1, 1)
                e = a[:6] - agt
                dser.append(float(d))
                eser.append((float(np.linalg.norm(e)), float(np.linalg.norm(e[:3])), float(np.linalg.norm(e[3:6]))))
                o, _, _, _ = env.step(a); steps += 1; hist.append(ov(o))
                pser.append(np.asarray(o["robot0_eef_pos"]).copy())
                oser.append(np.stack([hist[-2], hist[-1]]))
                if env._check_frame_assembled():
                    asm = True; break
                if steps >= args.max_steps:
                    break
        succ += int(asm); N += 1
        dser = np.array(dser); eser = np.array(eser)
        maxd = dser.max() if len(dser) else 0.0
        fc4 = int(np.argmax(dser >= 4.0)) if (dser >= 4.0).any() else -1
        dd = np.diff(dser)
        blo = dser[:-1] < 2.0; bhi = (dser[:-1] >= 2.0) & (dser[:-1] < 4.0)
        if blo.any():
            E_lo.append(eser[:-1][blo]); DD_lo.append(dd[blo])
        if bhi.any():
            E_hi.append(eser[:-1][bhi]); DD_hi.append(dd[bhi])
        # excursions: first step entering [2,4) from below
        t = 1
        while t < len(dser):
            if dser[t] >= 2.0 and dser[t - 1] < 2.0:
                r5 = int((dser[t:t + 5] < 2.0).any()); r10 = int((dser[t:t + 10] < 2.0).any()); r20 = int((dser[t:t + 20] < 2.0).any())
                c5 = int((dser[t:t + 5] >= 4.0).any()); c10 = int((dser[t:t + 10] >= 4.0).any()); c20 = int((dser[t:t + 20] >= 4.0).any())
                exc.append((r5, r10, r20, c5, c10, c20))
                # skip to after this excursion resolves (next t with d<2)
                nxt = t + 1
                while nxt < len(dser) and dser[nxt] >= 2.0:
                    nxt += 1
                t = nxt
            t += 1
        ep.append((int(asm), maxd, int(maxd >= 2.0), int(maxd >= 4.0), fc4))
        import os as _os
        if _os.environ.get("DUMPTRAJ"):
            _dt = _os.environ["DUMPTRAJ"]
            _os.makedirs(_dt, exist_ok=True)
            np.savez(f"{_dt}/ep_{sd}.npz", eef=np.array(pser), asm=int(asm), seed=sd,
                     obsw=np.array(oser, dtype=np.float32), dser=dser)
    ep = np.array(ep); c2 = ep[:, 2].astype(bool); c4 = ep[:, 3].astype(bool)
    sr_c4 = 100 * ep[c4, 0].mean() if c4.any() else float("nan")
    sr_n4 = 100 * ep[~c4, 0].mean() if (~c4).any() else float("nan")
    fct = ep[ep[:, 4] >= 0, 4]
    print(f"TWOFACTOR {args.tag} SR={succ}/{N} cross2={c2.mean():.2f} cross4={c4.mean():.2f} "
          f"SR|cross4={sr_c4:.0f} SR|stay={sr_n4:.0f} "
          f"maxd_p50/90/95={np.percentile(ep[:,1],50):.2f}/{np.percentile(ep[:,1],90):.2f}/{np.percentile(ep[:,1],95):.2f} "
          f"fc4_p50/90={np.median(fct) if len(fct) else float('nan'):.0f}/{np.percentile(fct,90) if len(fct) else float('nan'):.0f}", flush=True)
    for name, E, DD in [("d<2", E_lo, DD_lo), ("[2,4)", E_hi, DD_hi)]:
        if not E:
            print(f"TWOFACTOR {args.tag} band={name} EMPTY"); continue
        E = np.concatenate(E, 0); DD = np.concatenate(DD, 0)
        print(f"TWOFACTOR {args.tag} band={name} N={len(E)} "
              f"errTot_mean/p50/p90/p95={E[:,0].mean():.4f}/{np.percentile(E[:,0],50):.4f}/{np.percentile(E[:,0],90):.4f}/{np.percentile(E[:,0],95):.4f} "
              f"errPos_p50={np.percentile(E[:,1],50):.4f} errRot_p50={np.percentile(E[:,2],50):.4f} "
              f"dd_mean/p50/p90/p95={DD.mean():+.4f}/{np.percentile(DD,50):+.4f}/{np.percentile(DD,90):+.4f}/{np.percentile(DD,95):+.4f} "
              f"frac_dd>0={(DD>0).mean():.2f}", flush=True)
    if exc:
        X = np.array(exc)
        print(f"TWOFACTOR {args.tag} excursions N={len(X)} "
              f"return<2_within5/10/20={X[:,0].mean():.2f}/{X[:,1].mean():.2f}/{X[:,2].mean():.2f} "
              f"cross4_within5/10/20={X[:,3].mean():.2f}/{X[:,4].mean():.2f}/{X[:,5].mean():.2f}")


if __name__ == "__main__":
    main()
