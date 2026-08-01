"""Roll out DART-6k on the 40 seeds tracking distance to BOTH support clouds:
  d_clean = 1-NN to clean manifold cloud   (the ruler used everywhere = clean-trained PNR)
  d_dart  = 1-NN to DART support cloud      (DART's OWN ruler)
both standardized by the SAME clean mu/sig, matched point counts. Question: when DART's
d_clean exceeds the clean PNR (4), what is its d_dart? If d_dart stays small, then
'DART crosses PNR' is purely a clean-ruler artifact -- in its own units it never left."""
import os
os.environ["MUJOCO_GL"] = "egl"
import argparse
import numpy as np
import torch
import h5py
import sys
sys.path.insert(0, "scripts"); sys.path.insert(0, ".")
from collect_tool_hang_demos import ENV_KWARGS
from hydra import initialize_config_dir, compose
from omegaconf import OmegaConf
from scipy.spatial import cKDTree
import robosuite
from mip.datasets.robomimic_dataset import make_dataset
from mip.agent import TrainingAgent

OK = ["object", "robot0_eef_pos", "robot0_eef_quat", "robot0_gripper_qpos"]
KM = {"object": "object-state"}
DS_DECODE = "data/tool_hang_full2ins_2000.hdf5"


def build_cfg(loss):
    cfgdir = os.path.abspath("examples/configs")
    with initialize_config_dir(version_base=None, config_dir=cfgdir):
        cfg = compose(config_name="main", overrides=[
            "task=tool_hang_ph_state_delta_legacy", f"+task.dataset_path={os.path.abspath(DS_DECODE)}",
            "network=chiunet", f"optimization.loss_type={loss}",
            "optimization.auto_resume=false", "log.wandb_mode=disabled"])
    OmegaConf.set_struct(cfg, False); cfg.task.obs_dim = 53
    return cfg


def stack(path, ndemo=None, stride=1):
    h = h5py.File(path, "r"); g = "data" if "data" in h else "demos"; ks = list(h[g])
    if ndemo: ks = ks[:ndemo]
    out = []
    for k in ks:
        o = h[f"{g}/{k}/obs"]
        v = np.concatenate([np.asarray(o[key]) for key in OK], axis=1).astype(np.float32)
        out.append(v[::stride])
    h.close(); return np.concatenate(out, 0)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--demos", default="data/warmstart_demos.hdf5")
    ap.add_argument("--n", type=int, default=40); ap.add_argument("--manifold_n", type=int, default=30)
    ap.add_argument("--max_steps", type=int, default=500); ap.add_argument("--H", type=int, default=16)
    args = ap.parse_args()

    base_cfg = build_cfg("regression")
    ds = make_dataset(base_cfg.task)
    dev = base_cfg.optimization.device; AS = base_cfg.task.act_steps; start = base_cfg.task.obs_steps - 1
    nD = torch.load("analysis/recovery/norm_puredart6k.pt", weights_only=False)
    ag = TrainingAgent(build_cfg("regression")); ag.load("logs/puredart_mse/models/model_latest.pt", load_optimizer=False); ag.eval()
    no = nD["obs"]["state"]; na = nD["action"]

    df = h5py.File(args.demos, "r"); allk = list(df["demos"].keys())
    eval_keys = allk[:args.n]; man_keys = allk[args.n:args.n + args.manifold_n]
    env = robosuite.make("ToolHang", horizon=4000, **ENV_KWARGS)

    def ov(o): return np.concatenate([o[KM.get(k, k)] for k in OK]).astype(np.float32)
    def predict(hist):
        w = np.stack(hist[-2:])[None]
        ot = {"state": torch.tensor(no.normalize(w), device=dev, dtype=torch.float32)}
        with torch.no_grad():
            an = ag.sample(act_0=torch.randn((1, args.H, 10), device=dev), obs=ot, use_ema=True)
        return ds.undo_transform_action(na.unnormalize(an.detach().cpu().numpy())[:, start:start + AS])[0]

    # clean manifold cloud via env replay (same as escape_stats) -> defines mu/sig (the ruler)
    man = []
    for k in man_keys:
        d = df["demos/" + k]; sd = int(d.attrs["seed"]); acts = np.clip(d["actions"][:], -1, 1); s0 = d["state0"][:]
        np.random.seed(sd); env.reset(); env.sim.set_state_from_flattened(s0); env.sim.forward()
        arm = env.robots[0].composite_controller.part_controllers["right"]; arm.update(); arm.reset_goal()
        man.append(ov(env._get_observations(force_update=True)))
        for t in range(len(acts)):
            o, _, _, _ = env.step(acts[t]); man.append(ov(o))
    man = np.array(man); mu, sig = man.mean(0), man.std(0) + 1e-6
    clean_cloud = (man - mu) / sig
    # DART support cloud from stored obs, SAME mu/sig, matched point count
    dart_raw = stack("data/tool_hang_dart_full2ins_2000.hdf5", ndemo=60)
    Ncap = len(clean_cloud)
    rng = np.random.RandomState(0)
    di = rng.choice(len(dart_raw), min(Ncap, len(dart_raw)), replace=False)
    dart_cloud = (dart_raw[di] - mu) / sig
    print(f"clean cloud={len(clean_cloud)} pts, dart cloud={len(dart_cloud)} pts (same mu/sig)")
    tc = cKDTree(clean_cloud); td = cKDTree(dart_cloud)
    def dc(v): return float(tc.query((v - mu) / sig)[0])
    def dd(v): return float(td.query((v - mu) / sig)[0])

    PNR = 4.0
    rows = []
    cross_pairs = []  # (d_clean, d_dart) at steps where d_clean>PNR
    for k in eval_keys:
        d = df["demos/" + k]; sd = int(d.attrs["seed"])
        np.random.seed(sd); env.reset()
        arm = env.robots[0].composite_controller.part_controllers["right"]; arm.update(); arm.reset_goal()
        for _ in range(10):
            env.step(np.zeros(7))
        o = env._get_observations(force_update=True); hist = [ov(o), ov(o)]; steps = 0; asm = False
        dcs = []; dds = []
        while steps < args.max_steps and not asm:
            for a in predict(hist):
                o, _, _, _ = env.step(a); steps += 1; v = ov(o); hist.append(v)
                cc = dc(v); ee = dd(v); dcs.append(cc); dds.append(ee)
                if cc > PNR:
                    cross_pairs.append((cc, ee))
                if env._check_frame_assembled(): asm = True; break
                if steps >= args.max_steps: break
        dcs = np.array(dcs); dds = np.array(dds)
        rows.append((sd, asm, dcs.max(), dds.max()))
    df.close()

    sr = 100 * np.mean([a for _, a, _, _ in rows])
    reach_clean_pnr = 100 * np.mean([mc > PNR for _, _, mc, _ in rows])
    reach_dart_pnr = 100 * np.mean([md > PNR for _, _, _, md in rows])
    print(f"\nDART-6k  SR={sr:.0f}%")
    print(f"  reach d_CLEAN>4 (clean PNR, the number reported before): {reach_clean_pnr:.0f}% of seeds")
    print(f"  reach d_DART >4 (same threshold, DART's own ruler):      {reach_dart_pnr:.0f}% of seeds")
    print(f"  max d_dart over ALL seeds = {max(md for *_, md in rows):.2f}   (clean self p99=1.93, dart self p99~2.45)")
    if cross_pairs:
        cp = np.array(cross_pairs)
        print(f"\n  at the {len(cp)} steps where d_clean>4 (clean-PNR crossings):")
        print(f"    d_clean: median {np.median(cp[:,0]):.1f}  max {cp[:,0].max():.1f}")
        print(f"    d_dart : median {np.median(cp[:,1]):.2f}  max {cp[:,1].max():.2f}   <-- in DART's own units")
    print("\nper-seed (seed, asm, max d_clean, max d_dart):")
    for sd, asm, mc, md in rows:
        flag = "  <-- clean-PNR crossed" if mc > PNR else ""
        if mc > PNR or not asm:
            print(f"  {sd}  asm={asm}  maxd_clean={mc:6.2f}  maxd_dart={md:5.2f}{flag}")


if __name__ == "__main__":
    main()
