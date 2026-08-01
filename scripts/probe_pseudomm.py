"""PSEUDO-MULTIMODALITY probe: is the local action distribution at the
learner's neighborhood radius BRANCH-DISCRETE, does the truth lie ON a
branch, and do policies predict IN THE GAP (mode-averaging signature)?

Per held-out query state x (pocket stratum + all):
 A. neighbors within radius eps from the MP-200 pool, grouped by episode:
    separation S(eps) = mean pairwise dist between episode-mean chunks /
    mean within-episode chunk std  (S~1 unimodal, S>>1 discrete branches).
    Same with MP-2k neighbors at ITS operating radius (resolution claim).
 B. queries with >=2 branches within EPS0: project GT executed chunk and
    each arm's prediction onto the axis between the two nearest branch
    means m1,m2: t = <v-m1, m2-m1>/||m2-m1||^2, offaxis. GT should have
    t ~ 0/1 (truth on-branch); arm in-gap frac = P(0.25<t<0.75 &
    offaxis<0.6).
Prints PMM lines. Env: PM_ARMS name:loss:ckpt, PM_N queries.
"""
import os
import sys

os.environ["MUJOCO_GL"] = "egl"
sys.path.insert(0, "scripts")
sys.path.insert(0, ".")
import h5py
import numpy as np
import torch
from hydra import compose, initialize_config_dir
from omegaconf import OmegaConf
from scipy.spatial import cKDTree

from mip.agent import TrainingAgent
from mip.datasets.robomimic_dataset import make_dataset
from mip.samplers import get_sampler

D2 = "data/tool_hang_full2ins_mp_200.hdf5"
DK = "data/tool_hang_full2ins_mp_2000.hdf5"
HELD = "data/tool_hang_full2ins_mp_20k.hdf5"
OK = ["object", "robot0_eef_pos", "robot0_eef_quat", "robot0_gripper_qpos"]
ARMS = [tuple(a.split(":")) for a in os.environ["PM_ARMS"].split(",")]
NQ = int(os.environ.get("PM_N", "1200"))
EPS_LADDER = [0.4, 0.8, 1.2, 1.6, 2.4]
EPS0 = 1.5          # branch-detection radius for part B (learner scale)
AS_LO, AS_HI = 1, 9


def load_pool(path, stop=None):
    h = h5py.File(path, "r")
    names = sorted(h["data"].keys(), key=lambda d: int(d.split("_")[1]))
    if stop:
        names = names[:stop]
    X, Y, TID = [], [], []
    for ti, dn in enumerate(names):
        o = h[f"data/{dn}/obs"]
        S = np.concatenate([np.asarray(o[k]) for k in OK], 1).astype(np.float32)
        A = np.asarray(h[f"data/{dn}/actions"], dtype=np.float32)
        if len(S) < 12:
            continue
        for i in range(1, len(S) - 9):
            X.append(np.stack([S[i - 1], S[i]]).reshape(-1))
            Y.append(A[i - 1 + AS_LO:i - 1 + AS_HI, 0:3].reshape(-1))  # 24d pose
        TID.extend([ti] * (len(S) - 10))
    h.close()
    return np.stack(X), np.stack(Y), np.asarray(TID)


X2, Y2, T2 = load_pool(D2)
Xk, Yk, Tk = load_pool(DK)
mu, sd = X2.mean(0), X2.std(0) + 1e-6
amu = Y2.mean(0)
asd = Y2.std() + 1e-6
tree2 = cKDTree((X2 - mu) / sd)
treek = cKDTree((Xk - mu) / sd)
print(f"PMM pools 200:{len(X2)} 2k:{len(Xk)}", flush=True)

# held-out queries (pocket-weighted: half from progress .3-.5, half anywhere)
h = h5py.File(HELD, "r")
names = sorted(h["data"].keys(), key=lambda d: int(d.split("_")[1]))[2000:2400]
rng = np.random.RandomState(0)
QW, QA, QP = [], [], []
for dn in names:
    o = h[f"data/{dn}/obs"]
    S = np.concatenate([np.asarray(o[k]) for k in OK], 1).astype(np.float32)
    A = np.asarray(h[f"data/{dn}/actions"], dtype=np.float32)
    L = len(S)
    if L < 30:
        continue
    for i in [rng.randint(int(0.3 * L), int(0.5 * L)),
              rng.randint(1, L - 10)]:
        QW.append(np.stack([S[i - 1], S[i]]))
        QA.append(A[i - 1 + AS_LO:i - 1 + AS_HI, 0:3].reshape(-1))
        QP.append(i / L)
h.close()
QW, QA, QP = np.stack(QW)[:NQ], np.stack(QA)[:NQ], np.asarray(QP)[:NQ]
Qn = (QW.reshape(len(QW), -1) - mu) / sd
print(f"PMM queries {len(QW)}", flush=True)


def branch_stats(tree, Y, TID, q, eps):
    idx = tree.query_ball_point(q, eps)
    if len(idx) < 4:
        return None
    eps_ids = TID[idx]
    ys = (Y[idx] - amu) / asd
    groups = {}
    for e, y in zip(eps_ids, ys):
        groups.setdefault(e, []).append(y)
    means, wstds = [], []
    for e, g in groups.items():
        g = np.stack(g)
        means.append(g.mean(0))
        if len(g) > 1:
            wstds.append(np.linalg.norm(g - g.mean(0), axis=1).mean())
    if len(means) < 2:
        return None
    means = np.stack(means)
    pair = [np.linalg.norm(means[i] - means[j])
            for i in range(len(means)) for j in range(i + 1, len(means))]
    within = np.mean(wstds) if wstds else np.nan
    return np.mean(pair), within, len(means)


# ---- A: separation vs radius, pocket stratum -----------------------------
pocket = (QP >= 0.3) & (QP < 0.5)
for label, m in (("pocket", pocket), ("all", np.ones(len(QW), bool))):
    for eps in EPS_LADDER:
        seps, nmodes = [], []
        for q in Qn[m][:400]:
            r = branch_stats(tree2, Y2, T2, q, eps)
            if r and np.isfinite(r[1]) and r[1] > 1e-6:
                seps.append(r[0] / r[1])
                nmodes.append(r[2])
        if seps:
            print(f"PMM A200 {label} eps={eps} S_p50 {np.median(seps):.2f} "
                  f"p90 {np.percentile(seps,90):.2f} modes_p50 "
                  f"{np.median(nmodes):.0f} n={len(seps)}", flush=True)
    # 2k at its operating radius
    for eps in (0.8, 1.2):
        seps = []
        for q in Qn[m][:400]:
            r = branch_stats(treek, Yk, Tk, q, eps)
            if r and np.isfinite(r[1]) and r[1] > 1e-6:
                seps.append(r[0] / r[1])
        if seps:
            print(f"PMM A2k {label} eps={eps} S_p50 {np.median(seps):.2f} "
                  f"n={len(seps)}", flush=True)

# ---- B: two-branch queries; GT and per-arm projection --------------------
tw = []
for qi in range(len(QW)):
    idx = tree2.query_ball_point(Qn[qi], EPS0)
    if len(idx) < 4:
        continue
    ids = T2[idx]
    d = np.linalg.norm((X2[idx] - mu) / sd - Qn[qi], axis=1)
    # two nearest distinct episodes
    order = np.argsort(d)
    e1 = ids[order[0]]
    e2i = next((o for o in order if ids[o] != e1), None)
    if e2i is None:
        continue
    e2 = ids[e2i]
    m1 = (Y2[idx][ids == e1] - amu).mean(0) / asd
    m2 = (Y2[idx][ids == e2] - amu).mean(0) / asd
    sep = np.linalg.norm(m2 - m1)
    if sep < 0.5:
        continue                       # branches must genuinely disagree
    tw.append((qi, m1, m2))
print(f"PMM B two-branch queries {len(tw)}", flush=True)


def proj(v, m1, m2):
    ax = m2 - m1
    t = float(np.dot(v - m1, ax) / (np.dot(ax, ax) + 1e-12))
    off = float(np.linalg.norm(v - m1 - t * ax) / (np.linalg.norm(ax) + 1e-9))
    return t, off


gts = [proj((QA[qi].astype(np.float64) - amu) / asd, m1, m2)
       for qi, m1, m2 in tw]
gt_t = np.array([t for t, o in gts])
gt_ingap = np.mean([(0.25 < t < 0.75) and o < 0.6 for t, o in gts])
print(f"PMM GT t_p50 {np.median(np.abs(gt_t - 0.5)):.2f} (dist from mid) "
      f"ingap {gt_ingap:.2f} onbranch(|t|<.25 or |t-1|<.25 & off<.6) "
      f"{np.mean([((t < 0.25 or t > 0.75)) and o < 0.6 for t, o in gts]):.2f}",
      flush=True)

for name, loss, ck in ARMS:
    with initialize_config_dir(version_base=None,
                               config_dir=os.path.abspath("examples/configs")):
        cfg = compose(config_name="main", overrides=[
            "task=tool_hang_ph_state_delta_legacy",
            "+task.dataset_path=" + os.path.abspath(D2), "network=chiunet",
            f"optimization.loss_type={loss}", "optimization.auto_resume=false",
            "log.wandb_mode=disabled"])
    OmegaConf.set_struct(cfg, False)
    cfg.task.obs_dim = 53
    cfg.task.horizon = int(2 ** np.ceil(np.log2(cfg.task.horizon)))
    H = int(cfg.task.horizon)
    ds = make_dataset(cfg.task)
    ag = TrainingAgent(cfg)
    ag.load(ck, load_optimizer=False)
    ag.eval()
    sampler = get_sampler(loss)
    no, na = ds.normalizer["obs"]["state"], ds.normalizer["action"]
    dev = cfg.optimization.device
    qidx = [qi for qi, _, _ in tw]
    preds = []
    for i in range(0, len(qidx), 256):
        wb = QW[qidx[i:i + 256]]
        xb = torch.tensor(np.stack([no.normalize(w) for w in wb]),
                          device=dev, dtype=torch.float32)
        with torch.no_grad():
            a0 = torch.zeros((len(xb), H, 10), device=dev)
            an = sampler(cfg.optimization, ag.flow_map_ema,
                         ag.encoder_ema, a0, {"state": xb})
        pe = np.asarray(ds.undo_transform_action(
            na.unnormalize(an.cpu().numpy())[:, AS_LO:AS_HI]))
        preds.append(pe[:, :, 0:3].reshape(len(pe), -1))
    preds = np.concatenate(preds)
    ps = [proj((preds[k].astype(np.float64) - amu) / asd, tw[k][1], tw[k][2])
          for k in range(len(tw))]
    ingap = np.mean([(0.25 < t < 0.75) and o < 0.6 for t, o in ps])
    onb = np.mean([((t < 0.25) or (t > 0.75)) and o < 0.6 for t, o in ps])
    print(f"PMM {name} ingap {ingap:.3f} onbranch {onb:.3f} "
          f"tmid_p50 {np.median([abs(t-0.5) for t, o in ps]):.2f} "
          f"off_p50 {np.median([o for t, o in ps]):.2f}", flush=True)
print("PMM done", flush=True)
