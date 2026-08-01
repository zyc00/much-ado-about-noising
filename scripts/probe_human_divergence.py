"""Closed-loop divergence anatomy: where MSE and MIP behavior actually separates (human).

Uses the matched-seed dumps (both models, seeds 31000-31019, same initial states).

A. DIVERGENCE ONSET: per seed, first step where the two eef trajectories separate by >10mm;
   report the task stage (via nearest GT window) and tube distance at onset.
B. DRIFT FIELD: E[delta d | d] per tube-distance bin per model — the closed-loop
   contraction/expansion curve (the scripted discriminator, now on human rollouts).
C. MODE TAXONOMY: per 20-step window classify progress / stall (net eef move < 5mm, d < 3)
   / offtrack (d >= 3); occupancy, transition rates, P(exit stall within 40 steps).
D. COUNTERFACTUAL REPLAY: evaluate BOTH policies along MSE's stall sequences and MIP's
   align sequences: command magnitude, direction agreement cos(a_MSE, a_MIP), per-policy
   sequence coherence (TV along the visited states), and alignment with a kNN-GT servo
   proxy (nearest demo state's action direction).
"""
import os, sys
os.environ["MUJOCO_GL"] = "egl"
sys.path.insert(0, "scripts"); sys.path.insert(0, ".")
import numpy as np, h5py, torch
from scipy.spatial import cKDTree
from hydra import initialize_config_dir, compose
from omegaconf import OmegaConf
from mip.datasets.robomimic_dataset import make_dataset
from mip.agent import TrainingAgent

DSP = "/home/jigu/.cache/huggingface/hub/datasets--ChaoyiPan--mip-dataset/blobs/c5cb01b119a109a3d4842ca515906e86f1f35a8ee1a333d22b3503c1a513a8f4"
OK = ["object", "robot0_eef_pos", "robot0_eef_quat", "robot0_gripper_qpos"]
OLD = "/home/jigu/projects/much-ado-about-noising-old/checkpoints"

def load(loss):
    with initialize_config_dir(version_base=None, config_dir=os.path.abspath("examples/configs")):
        cfg = compose(config_name="main", overrides=["task=tool_hang_ph_state_delta_legacy",
            "+task.dataset_path=" + DSP, "network=chiunet",
            f"optimization.loss_type={loss}", "optimization.auto_resume=false", "log.wandb_mode=disabled"])
    OmegaConf.set_struct(cfg, False); cfg.task.obs_dim = 53
    ds = make_dataset(cfg.task)
    return cfg, ds, TrainingAgent(cfg)

cfg, ds, _ = load("regression")
no = ds.normalizer["obs"]["state"]; na = ds.normalizer["action"]; dev = cfg.optimization.device
START = cfg.task.obs_steps - 1

# GT anchor tree + per-state action + window labels
h = h5py.File(DSP, "r")
keys = sorted(h["data"].keys(), key=lambda k: int(k.split("_")[-1]))
CL, CA, CW = [], [], []
for di, k in enumerate(keys[:60]):
    o = h[f"data/{k}/obs"]
    ov = np.concatenate([np.asarray(o[q]) for q in OK], axis=1).astype(np.float32)
    a = np.clip(np.asarray(h[f"data/{k}/actions"]), -1, 1); g = a[:, 6]; T = len(a)
    cl = [t for t in range(1, T) if g[t-1] < 0 and g[t] >= 0]
    op = [t for t in range(1, T) if g[t-1] >= 0 and g[t] < 0]
    if not cl: continue
    c1 = cl[0]
    r1 = next((t for t in op if t > c1), None)
    c2 = next((t for t in cl if r1 and t > r1), None)
    for t in range(T):
        w = 8
        if t < c1 - 12: w = 0
        elif t < c1 + 5: w = 1
        elif r1 and t < r1 - 30: w = 2
        elif r1 and t < r1: w = 3
        elif r1 and c2 and t < c2 - 12: w = 4
        elif c2 and t < c2 + 5: w = 5
        elif c2 and t < T - 40: w = 6
        elif c2: w = 7
        CL.append(ov[t]); CA.append(a[t]); CW.append(w)
h.close()
CL = np.stack(CL); CA = np.stack(CA); CW = np.array(CW)
mu, sig = CL.mean(0), CL.std(0) + 1e-6
tree = cKDTree((CL - mu) / sig)
WNAMES = ["approach1", "settle1", "carry1", "align1", "between", "settle2", "carry2", "align2", "other"]

Z = {m: np.load(f"analysis/traj_vis/human_{m}.npz") for m in ["hMSE_s5", "hMIP_s5"]}
def eps_of(z):
    out = []
    i = 0
    while f"ep{i}_obs" in z.files:
        out.append((z[f"ep{i}_obs"], z[f"ep{i}_act"], z[f"ep{i}_pos"], z[f"ep{i}_meta"]))
        i += 1
    return out
EPS = {m: eps_of(Z[m]) for m in Z}

# ---- A. divergence onset (matched seeds)
print("=== A. DIVERGENCE ONSET (matched seeds)")
for j in range(20):
    o1, a1, p1, m1 = EPS["hMSE_s5"][j]; o2, a2, p2, m2 = EPS["hMIP_s5"][j]
    L = min(len(p1), len(p2))
    dd = np.linalg.norm(p1[:L] - p2[:L], axis=1)
    t = int(np.argmax(dd > 0.010)) if (dd > 0.010).any() else -1
    if t <= 0: t = L - 1
    dq, iq = tree.query((o1[min(t, len(o1)-1)] - mu) / sig)
    print(f"DIV seed{31000+j}: t*={t} window={WNAMES[CW[iq]]} d={dq:.1f} | MSE(succ={int(m1[0])},asm={int(m1[1])}) MIP(succ={int(m2[0])},asm={int(m2[1])})", flush=True)

# ---- B. drift field + C. mode taxonomy
print("=== B/C. DRIFT FIELD + MODES")
BINS = [(0, 1), (1, 2), (2, 3), (3, 5), (5, 1e9)]
for mname in ["hMSE_s5", "hMIP_s5"]:
    dd_by, prog_by = {b: [] for b in range(5)}, {b: [] for b in range(5)}
    occ = {"progress": 0, "stall": 0, "offtrack": 0}
    stall_exits, stall_tot = 0, 0
    for o, a, p, m in EPS[mname]:
        d, _ = tree.query((o - mu) / sig)
        deld = np.diff(d)
        for t in range(len(deld)):
            for b, (lo, hi) in enumerate(BINS):
                if lo <= d[t] < hi: dd_by[b].append(deld[t]); break
        L = min(len(p), len(d))
        modes = []
        for t in range(0, L - 20, 20):
            net = np.linalg.norm(p[t + 20] - p[t])
            md = d[t:t + 20].mean()
            mode = "offtrack" if md >= 3 else ("stall" if net < 0.005 else "progress")
            occ[mode] += 1; modes.append(mode)
        for i_, md_ in enumerate(modes[:-2]):
            if md_ == "stall":
                stall_tot += 1
                if "progress" in modes[i_+1:i_+3]: stall_exits += 1
    tot = sum(occ.values())
    print(f"MODES {mname}: progress={occ['progress']/tot:.0%} stall={occ['stall']/tot:.0%} offtrack={occ['offtrack']/tot:.0%} | P(stall->progress within 40 steps)={stall_exits/max(stall_tot,1):.0%} (n={stall_tot})", flush=True)
    line = f"DRIFT {mname} E[dd|d]:"
    for b, (lo, hi) in enumerate(BINS):
        if dd_by[b]:
            line += f" [{lo},{'inf' if hi>100 else hi})={np.mean(dd_by[b]):+.4f}(n={len(dd_by[b])})"
    print(line, flush=True)

# ---- D. counterfactual replay
print("=== D. COUNTERFACTUAL REPLAY")
POL = {}
for name, loss, ck in [("hMSE_s5", "regression", f"{OLD}/tool_hang_ph_state_regression_chiunet_256_seed5_success52.pt"),
                       ("hMIP_s5", "mip", f"{OLD}/tool_hang_ph_state_mip_chiunet_256_seed5_success82.pt")]:
    c_, d_, ag = load(loss)
    ag.load(ck, load_optimizer=False); ag.eval()
    POL[name] = (ag.encoder_ema, ag.flow_map_ema if hasattr(ag, "flow_map_ema") else ag.flow_map)
def pol_act(name, X):
    enc, fm = POL[name]
    preds = []
    for i in range(0, len(X), 512):
        x = torch.tensor(no.normalize(X[i:i+512]), device=dev, dtype=torch.float32)
        with torch.no_grad():
            e = enc({"state": x}, None)
            y = fm.get_velocity(torch.zeros(len(x), device=dev), torch.zeros(len(x), 16, 10, device=dev), e)
        preds.append(y[:, START].cpu().numpy())
    return np.concatenate(preds)

# MSE stall sequences: windows classified stall in failed MSE episodes
segs = []
for o, a, p, m in EPS["hMSE_s5"]:
    if int(m[0]) == 1: continue
    d, _ = tree.query((o - mu) / sig)
    L = min(len(p), len(o)) - 1
    for t in range(0, L - 20, 20):
        if np.linalg.norm(p[t + 20] - p[t]) < 0.005 and d[t:t + 20].mean() < 3:
            segs.append(np.stack([o[max(t - 1, 0):t + 19], o[t:t + 20]], axis=1))
segs = segs[:60]
print(f"stall segments: {len(segs)}", flush=True)
agree, magr, tv_m, tv_p, gtal_m, gtal_p = [], [], [], [], [], []
for W in segs:
    am = pol_act("hMSE_s5", W); ap = pol_act("hMIP_s5", W)
    am3 = ds.undo_transform_action(na.unnormalize(np.repeat(am[:, None], 16, 1)))[:, 0, :3]
    ap3 = ds.undo_transform_action(na.unnormalize(np.repeat(ap[:, None], 16, 1)))[:, 0, :3]
    n1 = np.linalg.norm(am3, axis=1) + 1e-9; n2 = np.linalg.norm(ap3, axis=1) + 1e-9
    agree.append(np.median((am3 * ap3).sum(1) / (n1 * n2)))
    magr.append(np.median(n2) / np.median(n1))
    tv_m.append(np.linalg.norm(np.diff(am3, axis=0), axis=1).mean() / (np.median(n1) + 1e-9))
    tv_p.append(np.linalg.norm(np.diff(ap3, axis=0), axis=1).mean() / (np.median(n2) + 1e-9))
    # kNN GT servo proxy
    dq, iq = tree.query((W[:, 1] - mu) / sig)
    gt3 = CA[iq][:, :3]; ng = np.linalg.norm(gt3, axis=1) + 1e-9
    gtal_m.append(np.median((am3 * gt3).sum(1) / (n1 * ng)))
    gtal_p.append(np.median((ap3 * gt3).sum(1) / (n2 * ng)))
print(f"REPLAY at MSE-stall states: cos(a_MSE,a_MIP) p50={np.median(agree):+.2f} | |a_MIP|/|a_MSE| p50={np.median(magr):.2f} | relTV MSE={np.median(tv_m):.2f} MIP={np.median(tv_p):.2f} | GT-align MSE={np.median(gtal_m):+.2f} MIP={np.median(gtal_p):+.2f}", flush=True)
print("DIVERGENCE-DONE")
