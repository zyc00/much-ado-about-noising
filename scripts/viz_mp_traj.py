"""One episode per collector mode (default pointing-servo vs MJ_FF motion-plan
tracking); 3D EEF trajectory with per-step action direction arrows."""
import os
import sys

os.environ["MUJOCO_GL"] = "egl"
sys.path.insert(0, "scripts")
sys.path.insert(0, ".")
import matplotlib
matplotlib.use("Agg")
import importlib
import matplotlib.pyplot as plt
import numpy as np


def run_mode(mode_env, seed=3):
    for k in ("MJTRAJ", "MJ_FF", "MJ_DECOUPLE", "SMOOTH_CAP"):
        os.environ.pop(k, None)
    os.environ.update(mode_env)
    import scripted_tool_hang_v2 as V
    importlib.reload(V)
    ok, traj, _, _ = V.run_episode(seed, record=True)
    P = np.array(traj["obs"]["robot0_eef_pos"])
    A = np.array(traj["actions"])
    print(f"mode {mode_env}: ok={ok} steps={len(P)}", flush=True)
    return ok, P, A


ok1, P1, A1 = run_mode({})                                   # default servo
ok2, P2, A2 = run_mode({"MJTRAJ": "1", "MJ_FF": "1", "MJ_DECOUPLE": "1"})  # MP tracking

fig = plt.figure(figsize=(16, 7.5))
for i, (name, P, A, ok) in enumerate([
        ("default collector: pointing-servo actions", P1, A1, ok1),
        ("MP mode (min-jerk plan + FF/FB tracking)", P2, A2, ok2)]):
    ax = fig.add_subplot(1, 2, i + 1, projection="3d")
    t = np.linspace(0, 1, len(P))
    ax.scatter(P[:, 0], P[:, 1], P[:, 2], c=t, cmap="viridis", s=2)
    ss = slice(0, len(P) - 1, 5)
    q = A[ss, :3]
    qn = q / (np.linalg.norm(q, axis=1, keepdims=True) + 1e-9)
    mag = np.linalg.norm(q, axis=1, keepdims=True)
    L = 0.018 * np.clip(mag, 0.05, 1.0)  # arrow length ~ |action|
    ax.quiver(P[ss, 0], P[ss, 1], P[ss, 2],
              (qn * L)[:, 0], (qn * L)[:, 1], (qn * L)[:, 2],
              color="crimson", lw=0.7, arrow_length_ratio=0.35, alpha=0.8)
    ax.set_title(f"{name}\nsuccess={ok}, {len(P)} steps  "
                 f"(arrow = action dir, length ~ |a|)", fontsize=10)
    ax.set_xlabel("x"); ax.set_ylabel("y"); ax.set_zlabel("z")
    ax.view_init(elev=22, azim=-60)
fig.tight_layout()
fig.savefig("analysis/paper/mp_traj_demo.png", dpi=150, bbox_inches="tight")
print("wrote analysis/paper/mp_traj_demo.png", flush=True)
