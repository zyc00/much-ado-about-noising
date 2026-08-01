"""Switch by the DISTANCE CURVE: roll out MSE saving per-step sim-state + OOD.
For each FAILED episode, locate (retrospectively) the onset of the FINAL sustained
rise = the last step where OOD < LO before it climbs to the point-of-no-return
(PNR). Restore the sim state THERE and hand to MIP. Does MIP rescue from the exact
pre-commit moment? (Diagnostic: uses future info to pick the switch point.)

  MUJOCO_GL=egl python scripts/eval_switch_curve.py --n 40 --lo 1.5 --pnr 4
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
DS = "data/tool_hang_full2ins_2000.hdf5"


def load(ckpt, loss):
    cfgdir = os.path.abspath("examples/configs")
    with initialize_config_dir(version_base=None, config_dir=cfgdir):
        cfg = compose(config_name="main", overrides=[
            "task=tool_hang_ph_state_delta_legacy", f"+task.dataset_path={os.path.abspath(DS)}",
            "network=chiunet", f"optimization.loss_type={loss}",
            "optimization.auto_resume=false", "log.wandb_mode=disabled"])
    OmegaConf.set_struct(cfg, False); cfg.task.obs_dim = 53
    ds = make_dataset(cfg.task)
    ag = TrainingAgent(cfg); ag.load(ckpt, load_optimizer=False); ag.eval()
    return cfg, ds, ag


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--demos", default="data/warmstart_demos.hdf5")
    ap.add_argument("--n", type=int, default=40); ap.add_argument("--manifold_n", type=int, default=30)
    ap.add_argument("--max_steps", type=int, default=500); ap.add_argument("--H", type=int, default=16)
    ap.add_argument("--lo", type=float, default=1.5); ap.add_argument("--pnr", type=float, default=4.0)
    args = ap.parse_args()
    cfg, ds, mse = load("logs/full_regression_2000/models/model_latest.pt", "regression")
    _, _, mip = load("logs/full_mip_2000/models/model_latest.pt", "mip")
    dev = cfg.optimization.device; AS = cfg.task.act_steps; start = cfg.task.obs_steps - 1
    no = ds.normalizer["obs"]["state"]; na = ds.normalizer["action"]
    df = h5py.File(args.demos, "r"); allk = list(df["demos"].keys())
    eval_keys = allk[:args.n]; man_keys = allk[args.n:args.n + args.manifold_n]
    env = robosuite.make("ToolHang", horizon=4000, **ENV_KWARGS)

    def ov(o): return np.concatenate([o[KM.get(k, k)] for k in OK]).astype(np.float32)
    def pred(ag, hist):
        w = np.stack(hist[-2:])[None]
        ot = {"state": torch.tensor(no.normalize(w), device=dev, dtype=torch.float32)}
        with torch.no_grad():
            an = ag.sample(act_0=torch.randn((1, args.H, 10), device=dev), obs=ot, use_ema=True)
        return ds.undo_transform_action(na.unnormalize(an.detach().cpu().numpy())[:, start:start + AS])[0]

    man = []
    for k in man_keys:
        d = df["demos/" + k]; sd = int(d.attrs["seed"]); acts = np.clip(d["actions"][:], -1, 1); s0 = d["state0"][:]
        np.random.seed(sd); env.reset(); env.sim.set_state_from_flattened(s0); env.sim.forward()
        arm = env.robots[0].composite_controller.part_controllers["right"]; arm.update(); arm.reset_goal()
        man.append(ov(env._get_observations(force_update=True)))
        for t in range(len(acts)):
            o, _, _, _ = env.step(acts[t]); man.append(ov(o))
    man = np.array(man); mu, sig = man.mean(0), man.std(0) + 1e-6
    tree = cKDTree((man - mu) / sig)
    def ood(v): return float(tree.query((v - mu) / sig)[0])

    # 1. MSE rollout, save per-step (ood, sim_state, obs)
    epi = []
    for k in eval_keys:
        d = df["demos/" + k]; sd = int(d.attrs["seed"])
        np.random.seed(sd); env.reset()
        arm = env.robots[0].composite_controller.part_controllers["right"]; arm.update(); arm.reset_goal()
        for _ in range(10):
            env.step(np.zeros(7))
        o = env._get_observations(force_update=True); hist = [ov(o), ov(o)]
        steps = 0; asm = False; oods = []; states = []; obss = [ov(o)]
        while steps < args.max_steps and not asm:
            for a in pred(mse, hist):
                states.append(env.sim.get_state().flatten().copy())
                o, _, _, _ = env.step(a); steps += 1; v = ov(o); hist.append(v)
                oods.append(ood(v)); obss.append(v)
                if env._check_frame_assembled(): asm = True; break
                if steps >= args.max_steps: break
        epi.append(dict(ood=np.array(oods), states=states, obss=obss, asm=asm))

    nfail = sum(not e["asm"] for e in epi)
    print(f"MSE: {sum(e['asm'] for e in epi)}/{len(epi)}  fail={nfail}")

    # 2. for each failure, find final-rise onset and restore+MIP
    rescued = 0; tried = 0; onset_steps = []
    for e in epi:
        if e["asm"]:
            continue
        d = e["ood"]
        cross = np.argmax(d >= args.pnr) if (d >= args.pnr).any() else (len(d) - 1)  # reach PNR (or end if stall)
        below = np.where(d[:cross + 1] < args.lo)[0]
        t0 = int(below[-1]) if len(below) else 0   # last safe step before the climb
        onset_steps.append(t0)
        tried += 1
        # restore the LOW-OOD state at the foot of the final rise (ood[t0] is post-step-t0),
        # i.e. sim-state index t0+1; hand to MIP from there.
        idx = min(t0 + 1, len(e["states"]) - 1)
        env.reset(); env.sim.set_state_from_flattened(e["states"][idx]); env.sim.forward()
        arm = env.robots[0].composite_controller.part_controllers["right"]; arm.update(); arm.reset_goal()
        w0 = e["obss"][t0]; w1 = e["obss"][min(t0 + 1, len(e["obss"]) - 1)]
        hist = [w0.copy(), w1.copy()]; steps = 0; asm = False
        while steps < args.max_steps and not asm:
            for a in pred(mip, hist):
                o, _, _, _ = env.step(a); steps += 1; hist.append(ov(o))
                if env._check_frame_assembled(): asm = True; break
                if steps >= args.max_steps: break
        rescued += int(asm)
    df.close()
    print(f"\nFinal-rise-onset switch to MIP (LO={args.lo}, PNR={args.pnr}):")
    print(f"  rescued {rescued}/{tried} of MSE failures   (mean onset step {np.mean(onset_steps):.0f})")
    print(f"  => MSE {sum(e['asm'] for e in epi)}/{len(epi)} -> with curve-switch {sum(e['asm'] for e in epi)+rescued}/{len(epi)}")


if __name__ == "__main__":
    main()
