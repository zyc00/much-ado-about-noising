"""Splice deployment: policy A during PICK segment (knot < SWITCH of 240-knot tube),
policy B afterwards. Reports SR. Args via env: A_CKPT/A_LOSS/B_CKPT/B_LOSS/SWITCH/TAG."""
import os, sys
os.environ["MUJOCO_GL"] = "egl"
sys.path.insert(0, "scripts"); sys.path.insert(0, ".")
import numpy as np, torch, h5py
import robosuite
from hydra import initialize_config_dir, compose
from omegaconf import OmegaConf
from mip.datasets.robomimic_dataset import make_dataset
from mip.agent import TrainingAgent
from scripted_tool_hang_v2 import ENV_KWARGS
OK = ["object", "robot0_eef_pos", "robot0_eef_quat", "robot0_gripper_qpos"]
KM = {"object": "object-state"}
DS = "data/tool_hang_full2ins_2000.hdf5"
SWITCH = int(os.environ.get("SWITCH", "70"))

def load_agent(loss, ck, ds_path=None):
    with initialize_config_dir(version_base=None, config_dir=os.path.abspath("examples/configs")):
        cfg = compose(config_name="main", overrides=["task=tool_hang_ph_state_delta_legacy",
            f"+task.dataset_path={os.path.abspath(ds_path or DS)}", "network=chiunet",
            f"optimization.loss_type={loss}", "optimization.auto_resume=false", "log.wandb_mode=disabled"])
    OmegaConf.set_struct(cfg, False); cfg.task.obs_dim = 53
    ds = make_dataset(cfg.task)
    ag = TrainingAgent(cfg); ag.load(ck, load_optimizer=False); ag.eval()
    return cfg, ds, ag
cfgA, dsA, agA = load_agent(os.environ["A_LOSS"], os.environ["A_CKPT"], os.environ.get("A_DS"))
cfgB, dsB, agB = load_agent(os.environ["B_LOSS"], os.environ["B_CKPT"])
dev = cfgA.optimization.device; AS = cfgA.task.act_steps; start = cfgA.task.obs_steps - 1

h = h5py.File(DS, "r")
keys = sorted(h["data"].keys(), key=lambda k: int(k.split("_")[-1]))
grid = np.linspace(0, 1, 240); tube = []
for k in keys[:300]:
    p = np.asarray(h[f"data/{k}/obs/robot0_eef_pos"]); t = np.linspace(0, 1, len(p))
    tube.append(np.stack([np.interp(grid, t, p[:, i]) for i in range(3)], 1))
C = np.stack(tube).mean(0)
h.close()

def ov(o): return np.concatenate([o[KM.get(k, k)] for k in OK]).astype(np.float32)
def chunk_of(ag, ds, cfg, hist):
    no = ds.normalizer["obs"]["state"]; na = ds.normalizer["action"]
    w = np.stack(hist[-2:])[None]
    ot = {"state": torch.tensor(no.normalize(w), device=dev, dtype=torch.float32)}
    with torch.no_grad():
        an = ag.sample(act_0=torch.randn((1, 16, 10), device=dev), obs=ot, use_ema=True)
    return ds.undo_transform_action(na.unnormalize(an.detach().cpu().numpy())[:, start:start + AS])[0]

env = robosuite.make("ToolHang", horizon=4000, **ENV_KWARGS)
succ = 0; N = 0; sw_share = []
for sd in range(21000, 21048):
    np.random.seed(sd); env.reset()
    arm = env.robots[0].composite_controller.part_controllers["right"]; arm.update(); arm.reset_goal()
    for _ in range(10): env.step(np.zeros(7))
    o = env._get_observations(force_update=True); hist = [ov(o), ov(o)]
    steps = 0; asm = False; ki = 0; a_steps = 0; locked = False; closed_at = None; sw_step = None
    while steps < 700 and not asm:
        eef = hist[-1][44:47]
        lo2, hi2 = max(0, ki - 10), min(240, ki + 26)
        j = int(np.argmin(np.linalg.norm(eef[None] - C[lo2:hi2], axis=1))) + lo2
        ki = max(ki, j)
        MODE = os.environ.get("SW_MODE", "off"); OFF = int(os.environ.get("SW_OFF", "16"))
        if not locked:
            if MODE == "z" and ki >= 60 and hist[-1][46] < 0.82: locked = True
            elif MODE == "off" and closed_at is not None and steps - closed_at >= OFF: locked = True
            if locked: sw_step = steps
        useA = not locked
        ch = chunk_of(agA, dsA, cfgA, hist) if useA else chunk_of(agB, dsB, cfgB, hist)
        if useA and closed_at is None and ch[:, 6].max() >= 0:
            closed_at = steps
            if MODE == "off" and OFF == 0:
                locked = True; ch = chunk_of(agB, dsB, cfgB, hist); useA = False
        if useA: a_steps += AS
        for a in ch:
            o, _, _, _ = env.step(a); steps += 1; hist.append(ov(o))
            if env._check_frame_assembled(): asm = True; break
            if steps >= 700: break
    succ += int(asm); N += 1; sw_share.append(a_steps / max(steps, 1))
    import os as _os
    if _os.environ.get("DUMPSW"):
        _dd = _os.environ["DUMPSW"]; _os.makedirs(_dd, exist_ok=True)
        np.savez(f"{_dd}/sw_{sd}.npz", obs_traj=np.array(hist, dtype=np.float32),
                 sw_step=(-1 if sw_step is None else sw_step),
                 closed_at=(-1 if closed_at is None else closed_at), asm=int(asm), seed=sd, steps=steps)
    if _os.environ.get("DUMPTRAJ"):
        _dt = _os.environ["DUMPTRAJ"]; _os.makedirs(_dt, exist_ok=True)
        np.savez(f"{_dt}/ep_{sd}.npz", eef=np.array([hh[44:47] for hh in hist]), asm=int(asm), seed=sd)
print(f"SPLICE-G {os.environ.get('TAG','')}: SR={succ}/{N} A-share_p50={np.median(sw_share):.2f}")
print("SPLICE-DONE")
