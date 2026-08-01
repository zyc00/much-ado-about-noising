"""Retry-mechanism annotated videos (frontview):
  1) DEMO 9 replay      : 4 pocket entries 21->22->19->8deg, active rotation between
  2) hMSE_s5  seed31000 : FAIL — retries never rotate (18->19->22deg, rotcmd ~0.015)
  3) hMIP_s5  seed31002 : PASS — one retreat rotates 19->4deg (rotcmd 0.067)
  4) hrotaux_s5 seed31001: PASS — repaired MSE rotates on retry (22->16deg, rotcmd 0.043)
Overlay: orientation gauge (commit threshold 15deg), |rot cmd| gauge (demo retreat median
0.025), lat/alt, pocket-entry history, phase label."""
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
    ov = np.concatenate([np.asarray(d["obs"][q]) for q in OK], axis=1).astype(np.float32)
    a = np.clip(np.asarray(d["actions"]), -1, 1); g = a[:, 6]; T = len(a)
    cl = [t for t in range(1, T) if g[t-1] < 0 and g[t] >= 0]
    op = [t for t in range(1, T) if g[t-1] >= 0 and g[t] < 0]
    if not cl: continue
    c1 = cl[0]; r1 = next((t for t in op if t > c1), None)
    if not r1: continue
    offs.append(ov[r1-1, FP] - ov[r1-1, BP]); dq.append(qn(ov[r1-20, FQ]))
    metas.append((k, np.asarray(d["states"]), ov, a, c1, r1))
h.close()
off = np.median(np.stack(offs), 0)
dq = np.stack(dq); qref = dq[0]
for i in range(1, len(dq)):
    if np.dot(dq[i], qref) < 0: dq[i] = -dq[i]
qmu = qn(dq.mean(0))

def readouts(s53):
    q = qn(s53[FQ])
    if np.dot(q, qmu) < 0: q = -q
    v = s53[FP] - (s53[BP] + off)
    return qangle(q, qmu), float(np.linalg.norm(v[:2])), float(v[2])

class Tracker:
    def __init__(self):
        self.entries = []; self.last_in = -99; self.entry_alt = None
        self.phase = "approach"; self.rot = []
    def update(self, t, oerr, lat, alt, rotmag):
        self.rot.append(rotmag)
        inp = 0.025 <= alt < 0.060 and 0.010 <= lat < 0.045
        if inp:
            if t - self.last_in > 5:
                self.entries.append(oerr); self.entry_alt = alt
            self.last_in = t; self.phase = f"IN POCKET (entry #{len(self.entries)})"
        elif alt < 0.022 and lat < 0.015:
            self.phase = "INSERTING"
        elif self.entries and self.entry_alt is not None:
            if alt >= self.entry_alt + 0.010: self.phase = "RETREAT (retry)"
            elif alt <= self.entry_alt - 0.010: self.phase = "COMMIT (descend)"
    def rotsm(self):
        return float(np.mean(self.rot[-10:])) if self.rot else 0.0

def annotate(frame, tag, step, oerr, lat, alt, trk):
    img = Image.fromarray(frame); dr = ImageDraw.Draw(img)
    dr.rectangle([8, 8, 372, 152], fill=(255, 255, 255))
    dr.text((16, 12), f"{tag}   step {step}", fill=(20, 20, 20))
    oc = (40, 150, 60) if oerr < 15 else ((235, 160, 30) if oerr < 21 else (210, 45, 45))
    dr.text((16, 30), f"in-hand orientation err: {oerr:5.1f} deg  (commit < ~15)", fill=oc)
    w = int(min(oerr, 32) / 32 * 300)
    dr.rectangle([16, 46, 16 + w, 56], fill=oc)
    m = 16 + int(15 / 32 * 300); dr.rectangle([m - 1, 44, m + 1, 58], fill=(20, 20, 20))
    r = trk.rotsm()
    rc = (40, 150, 60) if r >= 0.020 else ((235, 160, 30) if r >= 0.015 else (210, 45, 45))
    dr.text((16, 62), f"rotation command |w|: {r:5.3f}  (demo retry level 0.025)", fill=rc)
    w2 = int(min(r, 0.08) / 0.08 * 300)
    dr.rectangle([16, 78, 16 + w2, 88], fill=rc)
    m2 = 16 + int(0.025 / 0.08 * 300); dr.rectangle([m2 - 1, 76, m2 + 1, 90], fill=(20, 20, 20))
    dr.text((16, 94), f"lateral {lat*1000:4.0f} mm   height above gate {alt*1000:4.0f} mm", fill=(20, 20, 20))
    ent = "  ->  ".join(f"{e:.0f}deg" for e in trk.entries[-5:]) if trk.entries else "(none yet)"
    dr.text((16, 112), f"pocket entries: {ent}", fill=(60, 60, 60))
    pc = (210, 45, 45) if "RETREAT" in trk.phase else ((40, 150, 60) if ("COMMIT" in trk.phase or "INSERT" in trk.phase) else (60, 60, 60))
    dr.text((16, 130), f"phase: {trk.phase}", fill=pc)
    return np.asarray(img)

EK = dict(ENV_KWARGS); EK["has_offscreen_renderer"] = True
env = robosuite.make("ToolHang", horizon=4000, **EK)
CAM = os.environ.get("CAM", "frontview")

# 1) demo 9 replay
k, states, ov_d, act_d, c1, r1 = metas[9]
env.reset(); trk = Tracker(); frames = []
for t in range(c1 + 5, min(r1 + 15, len(states))):
    env.sim.set_state_from_flattened(states[t]); env.sim.forward()
    oerr, lat, alt = readouts(ov_d[t])
    trk.update(t, oerr, lat, alt, float(np.linalg.norm(act_d[t, 3:6])))
    if t % 2 == 0:
        fr = env.sim.render(height=448, width=512, camera_name=CAM)[::-1].copy()
        frames.append(annotate(fr, "DEMO (human): retries rotate the tool", t - c1, oerr, lat, alt, trk))
imageio.mimsave(f"{OUT}/retry_demo9.mp4", frames, fps=25, macro_block_size=1)
print("VID retry_demo9 saved", flush=True)

# 2-4) policy rollouts
def load(loss, extra=()):
    with initialize_config_dir(version_base=None, config_dir=os.path.abspath("examples/configs")):
        cfg = compose(config_name="main", overrides=["task=tool_hang_ph_state_delta_legacy",
            "+task.dataset_path=" + DSP, "network=chiunet",
            f"optimization.loss_type={loss}", "optimization.auto_resume=false", "log.wandb_mode=disabled"] + list(extra))
    OmegaConf.set_struct(cfg, False); cfg.task.obs_dim = 53
    ds = make_dataset(cfg.task)
    return cfg, ds, TrainingAgent(cfg)

RUNS = [("hMSE_s5", "regression", f"{OLD}/tool_hang_ph_state_regression_chiunet_256_seed5_success52.pt", (),
         31000, "MSE (FAIL): retries never rotate"),
        ("hMIP_s5", "mip", f"{OLD}/tool_hang_ph_state_mip_chiunet_256_seed5_success82.pt", (),
         31002, "MIP (SUCCESS): retry rotates 19->4 deg"),
        ("hrotaux_s5", "regression", "logs/hrotaux_s5/models/model_latest.pt",
         ("+task.rot_indicator=true", "task.act_dim=13"),
         31001, "MSE+rot-aux (SUCCESS): repaired retry")]
for name, loss, ck, extra, sd, tag in RUNS:
    cfg, ds, ag = load(loss, extra)
    AD = int(cfg.task.act_dim)
    ag.load(ck, load_optimizer=False); ag.eval()
    dev = cfg.optimization.device; AS = cfg.task.act_steps; start = cfg.task.obs_steps - 1
    no = ds.normalizer["obs"]["state"]; na = ds.normalizer["action"]
    def ovf(o): return np.concatenate([np.asarray(o[KM.get(kk, kk)]) for kk in OK]).astype(np.float32)
    def chunk(hist):
        w = np.stack(hist[-2:])[None]
        ot = {"state": torch.tensor(no.normalize(w), device=dev, dtype=torch.float32)}
        with torch.no_grad():
            an = ag.sample(act_0=torch.randn((1, 16, AD), device=dev), obs=ot, use_ema=True)
        return ds.undo_transform_action(na.unnormalize(an.detach().cpu().numpy())[:, start:start + AS])[0]
    np.random.seed(sd); torch.manual_seed(sd); env.reset()
    arm = env.robots[0].composite_controller.part_controllers["right"]; arm.update(); arm.reset_goal()
    for _ in range(10): env.step(np.zeros(7))
    o = env._get_observations(force_update=True); hist = [ovf(o), ovf(o)]
    trk = Tracker(); frames, steps, succ = [], 0, False
    while steps < 800 and not succ:
        for a in chunk(hist):
            o, _, _, _ = env.step(a); steps += 1; hist.append(ovf(o))
            oerr, lat, alt = readouts(hist[-1])
            trk.update(steps, oerr, lat, alt, float(np.linalg.norm(a[3:6])))
            if steps % 2 == 0:
                fr = env.sim.render(height=448, width=512, camera_name=CAM)[::-1].copy()
                frames.append(annotate(fr, tag, steps, oerr, lat, alt, trk))
            if env._check_success(): succ = True; break
            if steps >= 800: break
    fp = f"{OUT}/retry_{name}_seed{sd}.mp4"
    imageio.mimsave(fp, frames, fps=30, macro_block_size=1)
    print(f"VID retry_{name} saved ({'success' if succ else 'timeout'}) entries={['%.0f' % e for e in trk.entries]}", flush=True)
print("RETRYVID-DONE")
