"""Support + distance structure of the GRASP and INSERTION specialists.
Each specialist's support = the state coverage of its training data:
  grasp support   = states in tool_hang_init2grasp_20000  (init->grasp phase)
  insert support  = states in tool_hang_insertion_20000    (insertion phase)
Common z-scoring (full2ins stats), matched point counts. For full-task trajectories we
plot distance-to-grasp-support and distance-to-insert-support vs normalized progress:
shows which phase each covers, the crossover, and any middle GAP neither covers (the
handoff region). Pure obs geometry; no models, no env."""
import numpy as np
import h5py
from scipy.spatial import cKDTree
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
np.set_printoptions(precision=3, suppress=True)

OK = ["object", "robot0_eef_pos", "robot0_eef_quat", "robot0_gripper_qpos"]


def stack(path, ndemo=None, stride=1, per_traj=False):
    h = h5py.File(path, "r"); g = "data" if "data" in h else "demos"; ks = list(h[g])
    if ndemo:
        ks = ks[:ndemo]
    out = []
    for k in ks:
        o = h[f"{g}/{k}/obs"]
        v = np.concatenate([np.asarray(o[key]) for key in OK], axis=1).astype(np.float32)
        out.append(v[::stride])
    h.close()
    return out if per_traj else np.concatenate(out, 0)


# common standardization from the FULL task
full_stack = stack("data/tool_hang_full2ins_20000.hdf5", ndemo=200)
mu, sig = full_stack.mean(0), full_stack.std(0) + 1e-6

grasp_raw = stack("data/tool_hang_init2grasp_20000.hdf5", ndemo=200)
insert_raw = stack("data/tool_hang_insertion_20000.hdf5", ndemo=200)
N = min(len(grasp_raw), len(insert_raw))
rng = np.random.RandomState(0)
gcloud = (grasp_raw[rng.choice(len(grasp_raw), N, replace=False)] - mu) / sig
icloud = (insert_raw[rng.choice(len(insert_raw), N, replace=False)] - mu) / sig
print(f"grasp cloud N={N}, insert cloud N={N} (common full-task mu/sig)")
tg = cKDTree(gcloud); ti = cKDTree(icloud)


def loo(tree, cloud):
    d, _ = tree.query(cloud, k=2); return np.percentile(d[:, 1], [50, 95, 99])
print(f"grasp  support self-NN p50/95/99 = {loo(tg, gcloud)}")
print(f"insert support self-NN p50/95/99 = {loo(ti, icloud)}")

# overlap: how far is each cloud from the OTHER's support
d_g_to_i = ti.query(gcloud)[0]   # grasp states -> insert support
d_i_to_g = tg.query(icloud)[0]   # insert states -> grasp support
print(f"\ngrasp states -> insert support: median {np.median(d_g_to_i):.2f}  (large = disjoint)")
print(f"insert states -> grasp support: median {np.median(d_i_to_g):.2f}")

# distance vs progress along full-task trajectories
trajs = stack("data/tool_hang_full2ins_20000.hdf5", ndemo=250, per_traj=True)
NB = 20
acc_g = [[] for _ in range(NB)]; acc_i = [[] for _ in range(NB)]
for tr in trajs:
    z = (tr - mu) / sig
    dg = tg.query(z)[0]; di = ti.query(z)[0]
    T = len(z)
    for t in range(T):
        b = min(NB - 1, int(NB * t / T))
        acc_g[b].append(dg[t]); acc_i[b].append(di[t])
prog = [(b + 0.5) / NB for b in range(NB)]
mg = [np.median(x) for x in acc_g]; mi = [np.median(x) for x in acc_i]
print(f"\n{'progress':>9} {'d->grasp-supp':>13} {'d->insert-supp':>14}")
for b in range(NB):
    print(f"{prog[b]:>9.2f} {mg[b]:>13.2f} {mi[b]:>14.2f}")

fig, ax = plt.subplots(figsize=(9, 5.2))
gg = [np.array(x) for x in acc_g]; ii = [np.array(x) for x in acc_i]
ax.plot(prog, mg, "-o", color="tab:blue", lw=2, label="dist → GRASP support")
ax.fill_between(prog, [np.percentile(x, 25) for x in gg], [np.percentile(x, 75) for x in gg], color="tab:blue", alpha=0.15)
ax.plot(prog, mi, "-o", color="tab:red", lw=2, label="dist → INSERT support")
ax.fill_between(prog, [np.percentile(x, 25) for x in ii], [np.percentile(x, 75) for x in ii], color="tab:red", alpha=0.15)
ax.axhline(np.percentile(loo(tg, gcloud), 100), color="gray", ls=":", lw=0.8)
ax.set_xlabel("normalized task progress (0=init, 1=inserted)")
ax.set_ylabel("1-NN distance to specialist support (z-scored obs)")
ax.set_title("Specialist supports along the full task: grasp covers early, insert covers late\n"
             "(where BOTH curves are high = handoff region neither specialist covers)")
ax.legend(); ax.grid(alpha=0.25)
plt.tight_layout(); plt.savefig("analysis/recovery/specialist_supports.png", dpi=120)
print("\nsaved analysis/recovery/specialist_supports.png")
