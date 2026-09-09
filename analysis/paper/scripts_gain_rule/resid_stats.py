"""Model-free residual statistics of demonstration datasets (tail + heteroscedasticity + scale).

Residual families per action chunk (H steps x d dims, actions robust-normalized to ~[-1,1]):
  knn : chunk minus the mean chunk of the K nearest-state chunks from OTHER episodes of the same task
        (state-conditional spread; the closest model-free analogue of an MSE model's training residual)
  zoh : chunk minus the last executed action held constant (dynamics innovation; uniform, needs no state)
  sg  : chunk minus a Savitzky-Golay smooth of the episode's actions (high-frequency jitter only)
Statistics per family: rms; tail = top-1%/5% chunk share of the total squared residual (with the Gaussian
reference for the same H*d), element excess kurtosis, Student-t df MLE; heteroscedasticity = split-half
reliability of the per-chunk log mean-square across the two TIME halves (Spearman-Brown) and across random
DIM halves, dispersion sd(log m), and their product (the reliable between-chunk scale variation).
Usage: python resid_stats.py <dataset> [--H 8,10] [--out dir]
"""
import argparse, glob, json, os, sys, time
import numpy as np
from scipy.spatial import cKDTree
from scipy.signal import savgol_filter
from scipy.special import gammaln

ROOT = "/mnt/pfs/yuchen"
RNG = np.random.default_rng(0)

# ----------------------------------------------------------------------------- loaders
# each returns (episodes, grip_dims, task_names) with episodes = list of (state[T,s], action[T,d], task_id)
def load_robomimic(task, quality):
    import h5py
    f = h5py.File(f"{ROOT}/data/mip/robomimic/{task}/{quality}/low_dim.hdf5", "r")
    keys = ["robot0_eef_pos", "robot0_eef_quat", "robot0_gripper_qpos", "object"]
    if task == "transport":
        keys += ["robot1_eef_pos", "robot1_eef_quat", "robot1_gripper_qpos"]
    eps = []
    for k in f["data"].keys():
        d = f["data"][k]
        a = d["actions"][:].astype(np.float32)
        s = np.concatenate([d["obs"][o][:] for o in keys], 1).astype(np.float32)
        eps.append((s, a, 0))
    grip = [6, 13] if task == "transport" else [6]
    return eps, grip, {0: f"{task}-{quality}"}

def load_pusht():
    z = np.load(f"{ROOT}/gainrule/pusht.npz")
    a, s, ends = z["action"].astype(np.float32), z["state"].astype(np.float32), z["episode_ends"]
    eps, st = [], 0
    for e in ends:
        eps.append((s[st:e], a[st:e], 0)); st = e
    return eps, [], {0: "push-T"}

def _read_parquets(files, max_eps=None, task_filter=None):
    import pyarrow.parquet as pq
    eps = {}
    for fn in files:
        t = pq.read_table(fn, columns=["observation.state", "action", "episode_index", "task_index"])
        st = np.stack(t.column("observation.state").to_numpy(zero_copy_only=False)).astype(np.float32)
        ac = np.stack(t.column("action").to_numpy(zero_copy_only=False)).astype(np.float32)
        ei = t.column("episode_index").to_numpy(); ti = t.column("task_index").to_numpy()
        for e in np.unique(ei):
            msk = ei == e
            tid = int(ti[msk][0])
            if task_filter is not None and tid not in task_filter: continue
            eps[int(e)] = (st[msk], ac[msk], tid)
        if max_eps and len(eps) >= max_eps: break
    return [eps[k] for k in sorted(eps)]

def load_gr1(per_task=40):
    dirs = sorted(glob.glob(f"{ROOT}/groot/lerobot/LeRobot/gr1_unified.*"))
    eps, names = [], {}
    for i, d in enumerate(dirs):
        files = sorted(glob.glob(f"{d}/data/chunk-*/episode_*.parquet"))
        pick = list(RNG.choice(len(files), size=min(per_task, len(files)), replace=False))
        for (s, a, _) in _read_parquets([files[j] for j in pick]):
            eps.append((s[:, GR1_DIMS], a[:, GR1_DIMS], i))
        names[i] = os.path.basename(d).split(".")[-1]
    return eps, list(range(14, 26)), names
GR1_DIMS = list(range(0, 7)) + list(range(22, 29)) + list(range(7, 13)) + list(range(29, 35)) + list(range(41, 44))  # modality.json: arms, hands, waist

def load_lerobot_sample(pattern, n_files, per_task_min=None):
    files = sorted(glob.glob(pattern))
    pick = RNG.choice(len(files), size=min(n_files, len(files)), replace=False)
    eps = _read_parquets([files[j] for j in sorted(pick)])
    return eps

def load_bridge(n=3000):
    eps = load_lerobot_sample(f"{ROOT}/groot/bridge_orig_lerobot/data/chunk-*/episode_*.parquet", n)
    return eps, [6], {}

def load_fractal(n=3000):
    eps = load_lerobot_sample(f"{ROOT}/groot/fractal_lerobot/data/chunk-*/episode_*.parquet", n)
    return eps, [6], {}

def load_libero10_v3():
    files = sorted(glob.glob(f"{ROOT}/cosmos3/LIBERO_LeRobot_v3/libero_10/data/chunk-*/file-*.parquet"))
    eps = _read_parquets(files)
    return eps, [6], {}

def _task_names(meta_dir):
    import pyarrow.parquet as pq
    t = pq.read_table(f"{meta_dir}/tasks.parquet").to_pandas()
    return {int(r["task_index"]): " ".join(str(i).lower().split()) for i, r in t.iterrows()}

def load_pi05_libero(n_files=140, subset=None):
    names = _task_names(f"{ROOT}/pi05/libero_lerobot/meta")
    filt = None
    if subset == "libero10":
        l10 = set(_task_names(f"{ROOT}/cosmos3/LIBERO_LeRobot_v3/libero_10/meta").values())
        filt = {k for k, v in names.items() if v in l10}
        print("pi05 libero_10 subset: matched", len(filt), "of", len(l10), "task names", flush=True)
    files = sorted(glob.glob(f"{ROOT}/pi05/libero_lerobot/data/chunk-*/file-*.parquet"))
    pick = RNG.choice(len(files), size=min(n_files, len(files)), replace=False)
    eps = _read_parquets([files[j] for j in sorted(pick)], task_filter=filt)
    return eps, [6], names

LOADERS = {
    "lift-ph": lambda: load_robomimic("lift", "ph"), "lift-mh": lambda: load_robomimic("lift", "mh"),
    "can-ph": lambda: load_robomimic("can", "ph"), "can-mh": lambda: load_robomimic("can", "mh"),
    "square-ph": lambda: load_robomimic("square", "ph"), "square-mh": lambda: load_robomimic("square", "mh"),
    "transport-ph": lambda: load_robomimic("transport", "ph"), "transport-mh": lambda: load_robomimic("transport", "mh"),
    "toolhang-ph": lambda: load_robomimic("tool_hang", "ph"), "push-T": load_pusht,
    "gr1": load_gr1, "bridge": load_bridge, "fractal": load_fractal, "libero10-v3": load_libero10_v3,
    "pi05-libero": load_pi05_libero, "pi05-libero10": lambda: load_pi05_libero(n_files=377, subset="libero10"),
}

# ----------------------------------------------------------------------------- residuals
def robust_norm(a):
    lo, hi = np.percentile(a, 1, axis=0), np.percentile(a, 99, axis=0)
    c, s = (hi + lo) / 2, (hi - lo) / 2
    keep = s > 1e-6
    return ((a - c) / np.where(keep, s, 1.0))[:, keep], keep

def make_chunks(eps, H, stride):
    """returns (chunk targets [N,H,d], states at chunk start [N,s], prev action [N,d], episode ids, task ids, sg residual [N,H,d])"""
    X, S, P, E, T, G = [], [], [], [], [], []
    for ei, (s, a, tid) in enumerate(eps):
        Tn = len(a)
        if Tn < H + 2: continue
        w = min(11, Tn if Tn % 2 == 1 else Tn - 1)
        sg = a - savgol_filter(a, w, 3, axis=0) if w >= 5 else np.zeros_like(a)
        for t in range(1, Tn - H + 1, stride):
            X.append(a[t:t + H]); S.append(s[t]); P.append(a[t - 1]); E.append(ei); T.append(tid); G.append(sg[t:t + H])
    return np.array(X), np.array(S), np.array(P), np.array(E), np.array(T), np.array(G)

def knn_residual(X, S, E, T, K=8):
    N, H, d = X.shape
    Sn, _ = robust_norm(S)
    Sn = Sn / (Sn.std(0) + 1e-6)
    R = np.zeros_like(X); ok = np.zeros(N, bool)
    for tid in np.unique(T):
        idx = np.where(T == tid)[0]
        if len(np.unique(E[idx])) < 3: continue
        tree = cKDTree(Sn[idx])
        _, nb = tree.query(Sn[idx], k=min(K * 6 + 1, len(idx)))
        for row, i in enumerate(idx):
            cand = idx[nb[row]]
            cand = cand[E[cand] != E[i]][:K]
            if len(cand) < K // 2: continue
            R[i] = X[i] - X[cand].mean(0); ok[i] = True
    return R, ok

def pearson(x, y):
    x, y = x - x.mean(), y - y.mean()
    return float((x * y).sum() / np.sqrt((x * x).sum() * (y * y).sum() + 1e-30))

def t_df_mle(z, grid=np.logspace(-0.5, 2.5, 37)):
    z = z[np.isfinite(z)]
    if len(z) > 300000: z = RNG.choice(z, 300000, replace=False)
    best = (None, -np.inf)
    for nu in grid:
        # profile the scale on a small grid around the robust scale
        s0 = 1.4826 * np.median(np.abs(z)) + 1e-6
        for s in s0 * np.array([0.5, 0.7, 0.85, 1.0, 1.2, 1.5, 2.0]):
            ll = (gammaln((nu + 1) / 2) - gammaln(nu / 2) - 0.5 * np.log(nu * np.pi) - np.log(s)
                  - (nu + 1) / 2 * np.log1p((z / s) ** 2 / nu)).mean()
            if ll > best[1]: best = (nu, ll)
    return float(best[0])

def chunk_stats(R, grip_cols, label):
    N, H, d = R.shape
    out = {"N": int(N), "H": int(H), "d": int(d)}
    for tag, cols in [("all", np.arange(d)), ("cont", np.array([c for c in range(d) if c not in grip_cols]))]:
        if len(cols) == 0: continue
        r2 = R[:, :, cols] ** 2
        m = r2.mean((1, 2)); tot = m.sum()
        s = np.sort(m)[::-1]
        n1, n5 = max(1, int(0.01 * N)), max(1, int(0.05 * N))
        deff = H * len(cols)
        g = RNG.chisquare(deff, size=200000) / deff; gs = np.sort(g)[::-1]
        z = R[:, :, cols] / np.sqrt(r2.mean((0, 1)) + 1e-12)
        zf = z.reshape(-1)
        zs = zf[np.isfinite(zf)]
        if len(zs) > 2000000: zs = RNG.choice(zs, 2000000, replace=False)
        kurt = float((zs ** 4).mean() / (zs ** 2).mean() ** 2 - 3)
        lm = np.log(m + 1e-4 * np.median(m) + 1e-12)
        mA, mB = r2[:, : H // 2].mean((1, 2)), r2[:, H // 2:].mean((1, 2))
        la, lb = np.log(mA + 1e-4 * np.median(m) + 1e-12), np.log(mB + 1e-4 * np.median(m) + 1e-12)
        rt = pearson(la, lb); rel_t = 2 * rt / (1 + rt)
        perm = RNG.permutation(len(cols)); ca, cb = cols[perm[: len(cols) // 2]], cols[perm[len(cols) // 2:]]
        if len(ca) and len(cb):
            da, db = (R[:, :, ca] ** 2).mean((1, 2)), (R[:, :, cb] ** 2).mean((1, 2))
            rd = pearson(np.log(da + 1e-4 * np.median(m) + 1e-12), np.log(db + 1e-4 * np.median(m) + 1e-12)); rel_d = 2 * rd / (1 + rd)
        else:
            rel_d = float("nan")
        out[tag] = {
            "rms": float(np.sqrt(m.mean())), "rms_median_chunk": float(np.sqrt(np.median(m))),
            "top1_share": float(s[:n1].sum() / tot), "top5_share": float(s[:n5].sum() / tot),
            "top1_share_gauss": float(gs[: max(1, int(0.01 * len(gs)))].sum() / gs.sum()),
            "top5_share_gauss": float(gs[: max(1, int(0.05 * len(gs)))].sum() / gs.sum()),
            "kurt_elem": kurt, "t_df": t_df_mle(zs),
            "rel_time": float(rel_t), "rel_dim": float(rel_d), "sd_logm": float(lm.std()),
            "sd_logm_gauss": float(np.log(g).std()), "hetero_signal": float(max(rel_t, 0) * lm.var()),
            "frac_m_gt4med": float((m > 4 * np.median(m)).mean()), "frac_m_lt_quartermed": float((m < np.median(m) / 4).mean()),
        }
    return out

def main():
    ap = argparse.ArgumentParser(); ap.add_argument("name"); ap.add_argument("--H", default="8"); ap.add_argument("--stride", type=int, default=2)
    ap.add_argument("--out", default=f"{ROOT}/gainrule/out"); ap.add_argument("--K", type=int, default=8); ap.add_argument("--max_chunks", type=int, default=250000); ap.add_argument("--clip", type=float, default=1.0)
    a = ap.parse_args(); os.makedirs(a.out, exist_ok=True)
    t0 = time.time(); eps, grip, names = LOADERS[a.name]()
    # normalize actions once over the dataset
    A = np.concatenate([e[1] for e in eps]); _, keep = robust_norm(A)
    lo, hi = np.percentile(A, 1, axis=0), np.percentile(A, 99, axis=0); c, sc = (hi + lo) / 2, np.where(hi - lo > 1e-6, (hi - lo) / 2, 1.0)
    clipf = (lambda x: np.clip(x, -a.clip, a.clip)) if a.clip > 0 else (lambda x: x)
    eps = [(s, clipf((ac - c) / sc)[:, keep], tid) for (s, ac, tid) in eps]
    kept_idx = np.where(keep)[0].tolist(); grip_cols = [kept_idx.index(g) for g in grip if g in kept_idx]
    res = {"dataset": a.name, "n_episodes": len(eps), "n_steps": int(sum(len(e[1]) for e in eps)), "action_dim": int(keep.sum()), "grip_cols": grip_cols,
           "n_tasks": int(len(set(e[2] for e in eps))), "load_s": time.time() - t0, "clip": a.clip}
    for H in [int(h) for h in a.H.split(",")]:
        X, S, P, E, T, G = make_chunks(eps, H, a.stride)
        if len(X) > a.max_chunks:
            sel = np.sort(RNG.choice(len(X), a.max_chunks, replace=False)); X, S, P, E, T, G = X[sel], S[sel], P[sel], E[sel], T[sel], G[sel]
        fam = {}
        fam["zoh"] = chunk_stats(X - P[:, None, :], grip_cols, "zoh")
        fam["sg"] = chunk_stats(G, grip_cols, "sg")
        if S.shape[1] > 0:
            R, ok = knn_residual(X, S, E, T, a.K)
            if ok.sum() > 100: fam["knn"] = chunk_stats(R[ok], grip_cols, "knn"); fam["knn"]["frac_ok"] = float(ok.mean())
        res[f"H{H}"] = fam
        print(f"[{a.name}] H={H} chunks={len(X)} done {time.time()-t0:.0f}s", flush=True)
    json.dump(res, open(f"{a.out}/{a.name}.json", "w"), indent=1)
    print(json.dumps({k: v for k, v in res.items() if not k.startswith('H')}))

if __name__ == "__main__":
    main()
