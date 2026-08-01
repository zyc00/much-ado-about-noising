"""Annotated gate videos: live in-hand orientation error + lateral distance burned into
each frame. (a) demo replay, (b) hMSE seed31003 fail, (c) hMIP seed31003 success."""
import os
os.environ["MUJOCO_GL"] = "egl"
import numpy as np, torch, h5py, sys
sys.path.insert(0, "scripts"); sys.path.insert(0, ".")
import robosuite, imageio
from PIL import Image, ImageDraw
from collect_tool_hang_demos import ENV_KWARGS
from hydra import initialize_config_dir, compose
from omegaconf import OmegaConf
from mip.datasets.robomimic_dataset import make_dataset
from mip.agent import TrainingAgent

DSP = "/home/jigu/.cache/huggingface/hub/datasets--ChaoyiPan--mip-dataset/blobs/c5cb01b119a109a3d4842ca515906e86f1f35a8ee1a333d22b3503c1a513a8f4"
OK = ["object", "robot0_eef_pos", "robot0_eef_quat", "robot0_gripper_qpos"]
KM = {"object": "object-state"}
OLD = "/home/jigu/projects/much-ado-about-noising-old/checkpoints"
FQ = slice(17, 21); BP = slice(7, 10); FP = slice(21, 24)
OUT = "analysis/traj_vis/videos"
def qn(q): return q / (np.linalg.norm(q) + 1e-9)
def qangle(q1, q2): return np.degrees(2*np.arccos(np.clip(abs(float(np.dot(q1, q2))), -1, 1)))

h = h5py.File(DSP, "r")
keys = sorted(h["data"].keys(), key=lambda k: int(k.split("_")[-1]))
dq, offs, metas = [], [], []
for k in keys[:40]:
    d = h[f"data/{k}"]
    o = d["obs"]
    ov = np.concatenate([np.asarray(o[q]) for q in OK], axis=1).astype(np.float32)
    a = np.clip(np.asarray(d["actions"]), -1, 1); g = a[:, 6]; T = len(a)
    cl = [t for t in range(1, T) if g[t-1] < 0 and g[t] >= 0]
    op = [t for t in range(1, T) if g[t-1] >= 0 and g[t] < 0]
    if not cl: continue
    c1 = cl[0]
    r1 = next((t for t in op if t > c1), None)
    if not r1: continue
    offs.append(ov[r1-1, FP] - ov[r1-1, BP])
    dq.append(qn(ov[r1-20, FQ]))
    metas.append((k, np.asarray(d["states"]), ov, c1, r1))
h.close()
off = np.median(np.stack(offs), 0)
dq = np.stack(dq); qref = dq[0]
for i in range(1, len(dq)):
    if np.dot(dq[i], qref) < 0: dq[i] = -dq[i]
qmu = qn(dq.mean(0))

def annotate(frame, oerr, lat, step, tag):
    img = Image.fromarray(frame)
    dr = ImageDraw.Draw(img)
    col = (40, 170, 60) if oerr < 12 else ((235, 160, 30) if oerr < 18 else (210, 45, 45))
    dr.rectangle([8, 8, 335, 92], fill=(255, 255, 255))
    dr.text((16, 12), f"{tag}   step {step}", fill=(20, 20, 20))
    dr.text((16, 32), f"in-hand orientation error: {oerr:5.1f} deg  (insertable < ~10)", fill=col)
    dr.text((16, 52), f"lateral distance to gate:  {lat*1000:5.0f} mm", fill=(20, 20, 20))
    w = int(min(oerr, 32) / 32 * 300)
    dr.rectangle([16, 72, 16 + w, 84], fill=col)
    dr.rectangle([16 + int(10/32*300) - 1, 70, 16 + int(10/32*300) + 1, 86], fill=(20, 20, 20))
    return np.asarray(img)

def readouts(s53):
    q = qn(s53[FQ])
    if np.dot(q, qmu) < 0: q = -q
    oerr = qangle(q, qmu)
    lat = np.linalg.norm((s53[FP] - (s53[BP] + off))[:2])
    return oerr, lat

EK = dict(ENV_KWARGS); EK["has_offscreen_renderer"] = True
env = robosuite.make("ToolHang", horizon=4000, **EK)
CAM = os.environ.get("CAM", "frontview")

# (a) demo replay (demo 0), gate segment
k, states, ov_d, c1, r1 = metas[0]
env.reset()
frames = []
for t in range(max(0, c1 - 10), min(r1 + 20, len(states))):
    env.sim.set_state_from_flattened(states[t]); env.sim.forward()
    fr = env.sim.render(height=448, width=512, camera_name=CAM)[::-1].copy()
    oerr, lat = readouts(ov_d[t])
    frames.append(annotate(fr, oerr, lat, t - c1, "DEMO (human)"))
imageio.mimsave(f"{OUT}/annotated_demo_gate_front.mp4", frames, fps=25, macro_block_size=1)
print("VID annotated demo saved", flush=True)

# (b)(c) policy rollouts seed 31003
def load(loss):
    with initialize_config_dir(version_base=None, config_dir=os.path.abspath("examples/configs")):
        cfg = compose(config_name="main", overrides=["task=tool_hang_ph_state_delta_legacy",
            "+task.dataset_path=" + DSP, "network=chiunet",
            f"optimization.loss_type={loss}", "optimization.auto_resume=false", "log.wandb_mode=disabled"])
    OmegaConf.set_struct(cfg, False); cfg.task.obs_dim = 53
    ds = make_dataset(cfg.task)
    return cfg, ds, TrainingAgent(cfg)
for name, loss, ck, tag in [("hMSE_s5", "regression", f"{OLD}/tool_hang_ph_state_regression_chiunet_256_seed5_success52.pt", "MSE policy (FAILS)"),
                            ("hMIP_s5", "mip", f"{OLD}/tool_hang_ph_state_mip_chiunet_256_seed5_success82.pt", "MIP policy (SUCCEEDS)")]:
    cfg, ds, ag = load(loss)
    ag.load(ck, load_optimizer=False); ag.eval()
    dev = cfg.optimization.device; AS = cfg.task.act_steps; start = cfg.task.obs_steps - 1
    no = ds.normalizer["obs"]["state"]; na = ds.normalizer["action"]
    def ovf(o): return np.concatenate([np.asarray(o[KM.get(kk, kk)]) for kk in OK]).astype(np.float32)
    def chunk(hist):
        w = np.stack(hist[-2:])[None]
        ot = {"state": torch.tensor(no.normalize(w), device=dev, dtype=torch.float32)}
        with torch.no_grad():
            an = ag.sample(act_0=torch.randn((1, 16, 10), device=dev), obs=ot, use_ema=True)
        return ds.undo_transform_action(na.unnormalize(an.detach().cpu().numpy())[:, start:start + AS])[0]
    sd = 31003
    np.random.seed(sd); torch.manual_seed(sd); env.reset()
    arm = env.robots[0].composite_controller.part_controllers["right"]; arm.update(); arm.reset_goal()
    for _ in range(10): env.step(np.zeros(7))
    o = env._get_observations(force_update=True); hist = [ovf(o), ovf(o)]
    frames, steps, succ = [], 0, False
    while steps < 800 and not succ:
        for a in chunk(hist):
            o, _, _, _ = env.step(a); steps += 1; hist.append(ovf(o))
            if steps % 2 == 0:
                fr = env.sim.render(height=448, width=512, camera_name=CAM)[::-1].copy()
                oerr, lat = readouts(hist[-1])
                frames.append(annotate(fr, oerr, lat, steps, tag))
            if env._check_success(): succ = True; break
            if steps >= 800: break
    fp = f"{OUT}/annotated_{name}_seed{sd}_front.mp4"
    imageio.mimsave(fp, frames, fps=30, macro_block_size=1)
    print(f"VID annotated {name} saved ({'success' if succ else 'timeout'})", flush=True)
print("ANNOT-DONE")
