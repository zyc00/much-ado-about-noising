"""Collect INSERTION data whose START distribution = the MSE grasp specialist's
grasp output (DAgger-style distribution matching for the handoff).

Per training seed: settled init -> MSE grasp specialist drives to grasp ->
scripted insertion (cfq align + 60deg tilt + release) takes over AND IS RECORDED.
The recorded [handoff -> insertion] segment is saved. Training an insertion
specialist on this makes its training-input distribution match what it sees when
chained after the grasp specialist (fixing the OOD handoff that capped two-stage
MSE at 49%).

Usage:
  MUJOCO_GL=egl python scripts/collect_handoff_demos.py \
     --grasp_ckpt logs/grasp_regression_2000/models/model_latest.pt \
     --grasp_ds data/tool_hang_init2grasp_2000.hdf5 \
     --src data/tool_hang_clean_20000.hdf5 --n_demos 2000 \
     --output data/tool_hang_handoffins_2000.hdf5
"""
import os
os.environ["MUJOCO_GL"] = "egl"
import argparse, json
import numpy as np
import torch
import h5py
import robosuite
import robosuite.utils.transform_utils as T
from scipy.spatial.transform import Rotation as R
from tqdm import tqdm
import sys
sys.path.insert(0, "scripts"); sys.path.insert(0, ".")
from collect_tool_hang_demos import ENV_KWARGS, ENV_ARGS, OBS_KEYS_TO_RECORD, extract_obs
from hydra import initialize_config_dir, compose
from omegaconf import OmegaConf
from mip.datasets.robomimic_dataset import make_dataset
from mip.agent import TrainingAgent

OK = ["object", "robot0_eef_pos", "robot0_eef_quat", "robot0_gripper_qpos"]
KM = {"object": "object-state"}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--grasp_ckpt", required=True)
    ap.add_argument("--grasp_ds", required=True)
    ap.add_argument("--src", default="data/tool_hang_clean_20000.hdf5")
    ap.add_argument("--n_demos", type=int, default=2000)
    ap.add_argument("--grasp_budget", type=int, default=160)
    ap.add_argument("--output", default="data/tool_hang_handoffins_2000.hdf5")
    ap.add_argument("--grasp_loss", default="regression")  # loss of the grasp specialist (regression|mip)
    args = ap.parse_args()

    cfgdir = os.path.abspath("examples/configs")
    with initialize_config_dir(version_base=None, config_dir=cfgdir):
        cfg = compose(config_name="main", overrides=[
            "task=tool_hang_ph_state_delta_legacy",
            f"+task.dataset_path={os.path.abspath(args.grasp_ds)}",
            "network=chiunet", f"optimization.loss_type={args.grasp_loss}",
            "optimization.auto_resume=false", "log.wandb_mode=disabled"])
    OmegaConf.set_struct(cfg, False); cfg.task.obs_dim = 53
    dev = cfg.optimization.device
    H = 16; AS = cfg.task.act_steps; start = cfg.task.obs_steps - 1
    dsg = make_dataset(cfg.task)
    ag = TrainingAgent(cfg); ag.load(args.grasp_ckpt, load_optimizer=False); ag.eval()
    nog = dsg.normalizer["obs"]["state"]; nag = dsg.normalizer["action"]

    sf = h5py.File(args.src, "r")
    src_keys = list(sf["data"].keys())
    env = robosuite.make("ToolHang", horizon=4000, **ENV_KWARGS)

    def ov(o):
        return np.concatenate([o[KM.get(k, k)] for k in OK]).astype(np.float32)

    def grasped(sim):
        g = env._check_grasp(gripper=env.robots[0].gripper["right"], object_geoms=env.frame.contact_geoms)
        fz = sim.data.site_xpos[sim.model.site_name2id("frame_mount_site")][2]
        return bool(g and fz > 0.86)

    demos = []; attempts = 0; ki = 0
    pbar = tqdm(total=args.n_demos, desc="handoff-ins")
    while len(demos) < args.n_demos and ki < len(src_keys):
        d = sf["data/" + src_keys[ki]]; ki += 1
        state0 = d["states"][0]; sd = int(d.attrs.get("seed", 0)); model_xml = d.attrs["model_file"]
        attempts += 1
        np.random.seed(sd); env.reset()
        sim = env.sim; sim.set_state_from_flattened(state0); sim.forward()
        arm = env.robots[0].composite_controller.part_controllers["right"]
        arm.update(); arm.reset_goal()
        o = [env._get_observations(force_update=True)]
        # stage 1: MSE grasp specialist drives to grasp (NOT recorded)
        hist = [ov(o[0]), ov(o[0])]; steps = 0; gdone = False
        while steps < args.grasp_budget and not gdone:
            w = np.stack(hist[-2:])[None]
            ot = {"state": torch.tensor(nog.normalize(w), device=dev, dtype=torch.float32)}
            with torch.no_grad():
                an = ag.sample(act_0=torch.randn((1, H, 10), device=dev), obs=ot, use_ema=True)
            a7 = dsg.undo_transform_action(nag.unnormalize(an.detach().cpu().numpy())[:, start:start+AS])[0]
            for a in a7:
                o[0], _, _, _ = env.step(a); steps += 1; hist.append(ov(o[0]))
                if grasped(sim):
                    gdone = True; break
                if steps >= args.grasp_budget:
                    break
        if not gdone:
            continue
        # stage 2: scripted insertion FROM the grasp-specialist's grasp, RECORDED
        traj = {"actions": [], "states": [], "rewards": [], "dones": [],
                "obs": {k: [] for k in OBS_KEYS_TO_RECORD}}

        def obs():
            return o[0]

        def gs(n):
            return sim.data.site_xpos[sim.model.site_name2id(n)].copy()

        def step(a):
            a = np.clip(a, -1, 1).astype(np.float64)
            traj["actions"].append(a.copy())
            traj["states"].append(sim.get_state().flatten().copy())
            for k, v in extract_obs(o[0]).items():
                traj["obs"][k].append(v)
            o[0], r, dn, _ = env.step(a)
            traj["rewards"].append(float(r)); traj["dones"].append(int(dn))

        def cfq():
            em = T.quat2mat(obs()["robot0_eef_quat"])
            fm = gs("frame_mount_site"); fi = gs("frame_intersection_site"); fh = gs("frame_hang_site")
            nw = (fm - fi) / np.linalg.norm(fm - fi); hw = (fh - fi) / np.linalg.norm(fh - fi)
            ne = em.T @ nw; he = em.T @ hw; neu = ne / np.linalg.norm(ne)
            hep = he - np.dot(he, neu) * neu; heu = hep / np.linalg.norm(hep)
            ed = np.column_stack([neu, heu, np.cross(neu, heu)])
            nt = np.array([0, 0, -1.]); ht = np.array([0, -1, 0.])
            return T.mat2quat(np.column_stack([nt, ht, np.cross(nt, ht)]) @ np.linalg.inv(ed))

        hc = np.mean([sim.data.geom_xpos[sim.model.geom_name2id(f"stand_wall{i}")] for i in range(4)], axis=0)
        sm = gs("stand_mount_site"); hc[2] = sm[2]
        tq2 = cfq()
        for _ in range(80):
            eef = obs()["robot0_eef_pos"]; etn = gs("frame_mount_site") - eef
            tf = hc - etn; tf[2] = sm[2] + 0.05 - etn[2]; dp = np.clip((tf - eef) * 15, -1, 1)
            ea = T.quat2axisangle(T.mat2quat(T.quat2mat(tq2) @ T.quat2mat(obs()["robot0_eef_quat"]).T))
            step(np.concatenate([dp, np.clip(ea * 8, -1, 1), [1]]))
            if np.linalg.norm((gs("frame_mount_site") - hc)[:2]) < 0.002:
                break
        m = T.quat2mat(tq2); ax = np.cross(m[:, 2], [0, 0, -1.]); ax /= np.linalg.norm(ax)
        tqt = T.mat2quat(R.from_rotvec(ax * np.radians(60)).as_matrix() @ m)
        for i in range(120):
            fr = (i + 1) / 120; dot = np.clip(abs(np.dot(tq2, tqt)), -1, 1); th = np.arccos(dot)
            tu = -tqt if np.dot(tq2, tqt) < 0 else tqt
            wp = tu if th < 1e-6 else (np.sin((1-fr)*th)/np.sin(th))*tq2 + (np.sin(fr*th)/np.sin(th))*tu
            wp /= np.linalg.norm(wp)
            ea = T.quat2axisangle(T.mat2quat(T.quat2mat(wp) @ T.quat2mat(obs()["robot0_eef_quat"]).T))
            step(np.concatenate([[0, 0, -1], np.clip(ea * 8, -1, 1), [1]]))
            if obs()["robot0_eef_pos"][2] <= 1.0:
                break
        for _ in range(20):
            step(np.array([0, 0, 0, 0, 0, 0, -1]))
        if env._check_frame_assembled() and len(traj["actions"]) > 0:
            traj["model_file"] = model_xml; traj["seed"] = sd
            demos.append(traj); pbar.update(1)
    pbar.close(); sf.close()
    print(f"Collected {len(demos)} handoff-insertion demos from {attempts} attempts "
          f"({100*len(demos)/max(attempts,1):.0f}%)")

    os.makedirs(os.path.dirname(args.output) or ".", exist_ok=True)
    with h5py.File(args.output, "w") as f:
        data = f.create_group("data")
        data.attrs["env_args"] = json.dumps(ENV_ARGS)
        data.attrs["total"] = sum(len(x["actions"]) for x in demos)
        for i, demo in enumerate(demos):
            grp = data.create_group(f"demo_{i}")
            grp.attrs["num_samples"] = len(demo["actions"])
            grp.attrs["model_file"] = demo["model_file"]; grp.attrs["seed"] = demo["seed"]
            grp.create_dataset("actions", data=np.array(demo["actions"], dtype=np.float64))
            grp.create_dataset("states", data=np.array(demo["states"], dtype=np.float64))
            grp.create_dataset("rewards", data=np.array(demo["rewards"], dtype=np.float64))
            grp.create_dataset("dones", data=np.array(demo["dones"], dtype=np.int64))
            og = grp.create_group("obs")
            for k in OBS_KEYS_TO_RECORD:
                og.create_dataset(k, data=np.array(demo["obs"][k]))
    n = sum(len(x["actions"]) for x in demos)
    print(f"Saved {args.output}: {n} samples, avg {n/max(len(demos),1):.0f}/demo")


if __name__ == "__main__":
    main()
