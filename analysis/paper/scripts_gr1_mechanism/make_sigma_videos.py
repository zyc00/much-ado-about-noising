"""Render GR1 HT rollout videos with the predicted sigma underneath.

Inputs (per task directory, produced by run_gr1_sigma_videos.sh on the cluster):
  videos/<uuid>_s{0,1}.mp4   one file per episode (20 fps, one frame per env step), in creation order
  traj.npz                   client dump, one row per policy call: done/success [S, E], obs.state.* [S, E, 1, d], act.action.* [S, E, 8, d]
  sigma.jsonl                server dump, one row per policy call: sigma (training definition), per_step [8], per_dim [29], groups

Each policy call executes 8 env steps; the video records one frame per 2 env steps (frames_per_call = 4, derived from the counts).
Usage: python make_sigma_videos.py <root with one dir per task> <out dir> [--fps 20] [--episode first_success]
"""
import argparse, json, os, subprocess, sys, glob
import numpy as np, cv2, matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

SHORT = {"PosttrainPnPNovelFromCuttingboardToPanSplitA": "cutting board -> pan", "PosttrainPnPNovelFromTrayToPotSplitA": "tray -> pot",
         "PosttrainPnPNovelFromPlacematToBasketSplitA": "placemat -> basket", "PnPBottleToCabinetClose": "bottle -> cabinet, close door",
         "PnPCanToDrawerClose": "can -> drawer, close drawer"}


def load_task(d):
    z = np.load(os.path.join(d, "traj_slim.npz" if os.path.exists(os.path.join(d, "traj_slim.npz")) else "traj.npz"), allow_pickle=False)
    rows = [json.loads(l) for l in open(os.path.join(d, "sigma.jsonl"))]
    done, succ = z["done"][:, 0], z["success"][:, 0]
    S = len(done)
    assert len(rows) >= S, f"{d}: {len(rows)} sigma rows < {S} client calls"
    if len(rows) != S:
        print(f"[warn] {d}: {len(rows)} sigma rows vs {S} client calls; using the last {S}", file=sys.stderr); rows = rows[-S:]
    st_hand = np.concatenate([z["obs.state.left_hand"], z["obs.state.right_hand"]], -1)[:, 0, 0]      # [S, 12]
    st_arm = np.concatenate([z["obs.state.left_arm"], z["obs.state.right_arm"]], -1)[:, 0, 0]         # [S, 14]
    eps, start = [], 0
    for s in range(S):
        if done[s] or s == S - 1:
            sl = slice(start, s + 1)
            eps.append(dict(t0=start, t1=s + 1, success=bool(succ[sl].any()), sigma=np.array([r["sigma"] for r in rows[sl]]),
                            arm=np.array([r["groups"]["arm"] for r in rows[sl]]), hand=np.array([r["groups"]["hand"] for r in rows[sl]]),
                            waist=np.array([r["groups"]["waist"] for r in rows[sl]]),
                            per_step=np.array([r["per_step"] for r in rows[sl]]), st_hand=st_hand[sl], st_arm=st_arm[sl]))
            start = s + 1
    vids = sorted(glob.glob(os.path.join(d, "videos", "*.mp4")), key=os.path.getmtime)
    return eps, vids


CROP_ROWS = (40, 212)   # the recorded composite (512 x 278) has black bars above/below the two 256-px camera views and a caption at the bottom


def read_frames(path):
    cap = cv2.VideoCapture(path); frames = []
    while True:
        ok, f = cap.read()
        if not ok: break
        frames.append(cv2.cvtColor(f, cv2.COLOR_BGR2RGB)[CROP_ROWS[0]:CROP_ROWS[1]])
    cap.release(); return frames


def render(task, ep, frames, out_mp4, out_png, fps):
    n_calls = len(ep["sigma"]); steps = np.arange(n_calls) * 8
    def closed_spans(x):
        """steps where a hand is more closed than its mid-range (normalized joint state, higher = more closed)"""
        closed = x > 0.5 * (x.min() + x.max()); out = []; i = 0
        while i < n_calls:
            if closed[i]:
                j = i
                while j + 1 < n_calls and closed[j + 1]: j += 1
                out.append((8 * i, 8 * (j + 1))); i = j + 1
            else: i += 1
        return out
    hand_close = ep["st_hand"][:, 6:].mean(1)                         # right hand (stage-cue strip)
    left_close = ep["st_hand"][:, :6].mean(1)
    rng = max(np.ptp(left_close), np.ptp(hand_close))                    # a hand that never opens/closes (range < half the other's) gets no spans
    spans_l = closed_spans(left_close) if np.ptp(left_close) > 0.5 * rng else []
    spans_r = closed_spans(hand_close) if np.ptp(hand_close) > 0.5 * rng else []
    SPANS = [(spans_l, "#1f77b4", "left hand closed"), (spans_r, "#2ca02c", "right hand closed")]
    arm_speed = np.r_[0, np.linalg.norm(np.diff(ep["st_arm"], axis=0), axis=1)]
    nF = len(frames); fpc = max(1, int(round(nF / n_calls)))          # frames per policy call (4: one frame per 2 env steps)
    steps_per_frame = 8 / fpc; calls_per_frame = np.minimum(np.arange(nF) // fpc, n_calls - 1)
    if abs(nF - fpc * n_calls) > fpc: print(f"[warn] {task}: {nF} frames vs {n_calls} calls x {fpc}", file=sys.stderr)
    tmp = out_mp4 + "_frames"; os.makedirs(tmp, exist_ok=True)
    ymax = max(ep["sigma"].max(), ep["arm"].max(), ep["hand"].max(), ep["waist"].max()) * 1.12; xmax = 8 * n_calls
    for f in range(nF):
        k = calls_per_frame[f]
        fig = plt.figure(figsize=(6.4, 7.2), dpi=100)
        gs = fig.add_gridspec(3, 1, height_ratios=[3.0, 2.2, 1.2], hspace=0.42, left=0.11, right=0.98, top=0.93, bottom=0.08)
        ax = fig.add_subplot(gs[0]); ax.imshow(frames[f]); ax.set_xticks([]); ax.set_yticks([])
        ax.set_title(f"GR1, HT head (c=2): {SHORT.get(task, task)}   env step {int(f*steps_per_frame)}/{int((nF-1)*steps_per_frame)}   success={int(ep['success'])}", fontsize=9, loc="left")
        ax = fig.add_subplot(gs[1])
        for sp_, col, lab in SPANS:
            for a, b in sp_: ax.axvspan(a, b, color=col, alpha=0.08, lw=0)
        ax.plot(steps, ep["sigma"], color="#b30000", lw=1.8, label="predicted $\\sigma$ (chunk mean, training definition)")
        ax.plot(steps, ep["arm"], color="#1f77b4", lw=1.1, label="arm channels")
        ax.plot(steps, ep["hand"], color="#2ca02c", lw=1.1, label="hand channels")
        ax.plot(steps, ep["waist"], color="#9467bd", lw=1.1, label="waist channels")
        ax.axvline(f * steps_per_frame, color="k", lw=0.8); ax.plot([steps[k]], [ep["sigma"][k]], "o", color="#b30000", ms=5)
        ax.set_ylim(0, ymax); ax.set_xlim(0, xmax); ax.set_ylabel("$\\sigma$ (normalized units)", fontsize=8)
        ax.legend(fontsize=6.5, frameon=False, loc="upper right", ncol=2); ax.grid(alpha=0.25); ax.tick_params(labelsize=7)
        ax.text(0.01, 0.9, f"$\\sigma$ = {ep['sigma'][k]:.3f}", transform=ax.transAxes, fontsize=8, color="#b30000")
        ax = fig.add_subplot(gs[2])
        ax.plot(steps, hand_close, color="#2ca02c", lw=1.1, label="right-hand closure (state)")
        ax2 = ax.twinx(); ax2.plot(steps, arm_speed, color="#7f7f7f", lw=0.9, label="arm joint speed"); ax2.set_yticks([])
        ax.axvline(f * steps_per_frame, color="k", lw=0.8); ax.set_xlim(0, xmax); ax.set_xlabel("env step", fontsize=8); ax.tick_params(labelsize=7)
        h1, l1 = ax.get_legend_handles_labels(); h2, l2 = ax2.get_legend_handles_labels(); ax.legend(h1 + h2, l1 + l2, fontsize=6.5, frameon=False, loc="upper right", ncol=2)
        fig.savefig(os.path.join(tmp, f"{f:05d}.png")); plt.close(fig)
    subprocess.run(["ffmpeg", "-y", "-loglevel", "error", "-framerate", str(fps), "-i", os.path.join(tmp, "%05d.png"), "-c:v", "libx264", "-pix_fmt", "yuv420p", "-crf", "23", out_mp4], check=True)
    # static contact sheet: 6 keyframes + the sigma curve with the keyframe positions marked
    ks = np.linspace(0, nF - 1, 6).astype(int)
    fig, axs = plt.subplots(2, 6, figsize=(12, 4.2), gridspec_kw=dict(height_ratios=[0.75, 1.1], hspace=0.3, wspace=0.06))
    for j, f in enumerate(ks):
        axs[0, j].imshow(frames[f]); axs[0, j].set_xticks([]); axs[0, j].set_yticks([]); axs[0, j].set_title(f"env step {int(f*steps_per_frame)}", fontsize=8)
    gs = axs[1, 0].get_gridspec(); [a.remove() for a in axs[1, :]]; ax = fig.add_subplot(gs[1, :])
    for sp_, col, lab in SPANS:
        for n_, (a, b) in enumerate(sp_): ax.axvspan(a, b, color=col, alpha=0.08, lw=0, label=lab if n_ == 0 else None)
    ax.plot(steps, ep["sigma"], color="#b30000", lw=1.8, label="predicted $\\sigma$"); ax.plot(steps, ep["arm"], color="#1f77b4", lw=1.0, label="arm channels"); ax.plot(steps, ep["hand"], color="#2ca02c", lw=1.0, label="hand channels"); ax.plot(steps, ep["waist"], color="#9467bd", lw=1.0, label="waist channels")
    for f in ks: ax.axvline(f * steps_per_frame, color=".7", lw=0.7, ls=":")
    ax.set_xlim(0, xmax); ax.set_xlabel("env step"); ax.set_ylabel("$\\sigma$ (normalized)"); ax.legend(frameon=False, fontsize=7.5, ncol=6); ax.grid(alpha=0.25)
    ax.set_title(f"GR1 HT (c=2): {SHORT.get(task, task)}, success={int(ep['success'])}", loc="left", fontsize=9)
    fig.savefig(out_png, dpi=200, bbox_inches="tight"); plt.close(fig)
    for p in glob.glob(os.path.join(tmp, "*.png")): os.remove(p)
    os.rmdir(tmp)


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("root"); ap.add_argument("out"); ap.add_argument("--fps", type=int, default=20); ap.add_argument("--episode", default="first_success")
    a = ap.parse_args(); os.makedirs(a.out, exist_ok=True); summary = {}
    for d in sorted(glob.glob(os.path.join(a.root, "*/"))):
        task = os.path.basename(os.path.normpath(d))
        if not os.path.exists(os.path.join(d, "sigma.jsonl")) or not (os.path.exists(os.path.join(d, "traj_slim.npz")) or os.path.exists(os.path.join(d, "traj.npz"))): continue
        eps, vids = load_task(d)
        if len(vids) != len(eps): print(f"[warn] {task}: {len(vids)} videos vs {len(eps)} episodes", file=sys.stderr)
        pick = next((i for i, e in enumerate(eps) if e["success"]), None) if a.episode == "first_success" else int(a.episode)
        if pick is None: print(f"[skip] {task}: no successful episode among {len(eps)}", file=sys.stderr); continue
        frames = read_frames(vids[pick]); ep = eps[pick]
        render(task, ep, frames, os.path.join(a.out, f"{task}_ep{pick}.mp4"), os.path.join(a.out, f"{task}_ep{pick}.png"), a.fps)
        summary[task] = dict(episode=pick, calls=len(ep["sigma"]), frames=len(frames), success=ep["success"], sigma_min=float(ep["sigma"].min()), sigma_max=float(ep["sigma"].max()),
                             sigma_first=float(ep["sigma"][0]), sigma_peak_step=int(8 * ep["sigma"].argmax()), waist_mean=float(ep["waist"].mean()), arm_mean=float(ep["arm"].mean()), hand_mean=float(ep["hand"].mean()), n_success=sum(e["success"] for e in eps), n_eps=len(eps))
        print(task, summary[task])
    json.dump(summary, open(os.path.join(a.out, "summary.json"), "w"), indent=1)


if __name__ == "__main__":
    main()
