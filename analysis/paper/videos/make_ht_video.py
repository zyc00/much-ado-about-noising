"""Render the long-tail video for one pi0.5 episode: camera frames on top, GT vs predicted chunk below,
timeline with HT-gated frames (red) and flow-low-gradient frames (blue)."""
import io, sys, os, subprocess, numpy as np, matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
from PIL import Image
f = sys.argv[1]; out = sys.argv[2]; fps = int(sys.argv[3]) if len(sys.argv) > 3 else 10
z = np.load(f, allow_pickle=True); gt, hp, fp = z["gt"], z["ht_pred"], z["flow_pred"]; S, sig, w, fl = z["S"], z["sigma"], z["w"], z["flow_loss"]
nu, d = float(z["nu"]), int(z["d"]); N = len(gt); task = str(z["task"]); e = int(z["episode"])
u = S / sig ** 2; gated = u > nu                                  # HT: past the knee (down-weighted)
lowgrad = fl <= np.quantile(fl, 0.10)                            # flow: smallest per-sample loss (smallest gradient)
names = ["dx", "dy", "dz", "drx", "dry", "drz", "grip"]; T = gt.shape[1]
tmpdir = out + "_frames"; os.makedirs(tmpdir, exist_ok=True)
for t in range(N):
    fig = plt.figure(figsize=(14, 9)); gs = fig.add_gridspec(3, 7, height_ratios=[2.2, 2.0, 0.9], hspace=0.45, wspace=0.35)
    for j, key in enumerate(("img", "img2")):
        ax = fig.add_subplot(gs[0, j*3:(j+1)*3+ (1 if j==1 else 0)]); ax.imshow(Image.open(io.BytesIO(z[key][t]))); ax.set_xticks([]); ax.set_yticks([])
        ax.set_title("third-person view" if j == 0 else "wrist view", fontsize=9)
    ax = fig.add_subplot(gs[0, 6]); ax.axis("off")
    status = ("HT DOWN-WEIGHTED (past knee)" if gated[t] else "HT: normal weight") + "\n" + ("FLOW: small gradient (bottom 10%)" if lowgrad[t] else "flow: normal gradient")
    ax.text(0, 0.95, f"episode {e}  frame {t}/{N-1}\n{task}\n\nS/σ² = {u[t]:.0f}  (ν = {nu:.0f})\nσ = {sig[t]:.3f}   gate w = {w[t]:.2f}\nresidual rms = {np.sqrt(S[t]/d):.3f}\nflow loss = {fl[t]:.4f}\n\n{status}", fontsize=8.5, va="top", family="monospace",
            color=("#B00020" if gated[t] else ("#0055AA" if lowgrad[t] else "black")))
    for k in range(7):
        ax = fig.add_subplot(gs[1, k]); ax.plot(gt[t, :, k], color="black", lw=1.8, label="GT"); ax.plot(hp[t, :, k], color="#D55E00", lw=1.2, label="HT single-pass"); ax.plot(fp[t, :, k], color="#0072B2", lw=1.0, alpha=0.8, label="flow sample")
        ax.set_title(names[k], fontsize=8); ax.tick_params(labelsize=6); ax.set_xlim(0, T - 1); ax.set_ylim(-3.2, 3.2); ax.grid(alpha=0.25)
        if k == 0: ax.set_ylabel("normalized action", fontsize=7); ax.legend(fontsize=6, frameon=False, loc="upper left")
        if gated[t]: 
            for s in ax.spines.values(): s.set_edgecolor("#B00020"); s.set_linewidth(2)
    ax = fig.add_subplot(gs[2, :]); x = np.arange(N)
    ax.fill_between(x, 0, 1, where=gated, color="#B00020", alpha=0.25, transform=ax.get_xaxis_transform(), label="HT past knee (down-weighted)")
    ax.fill_between(x, 0, 1, where=lowgrad, color="#0072B2", alpha=0.25, transform=ax.get_xaxis_transform(), label="flow: bottom-10% loss (small gradient)")
    ax.plot(x, w / w.max(), color="#B00020", lw=1.2, label="HT gate weight w (scaled)"); ax.plot(x, fl / fl.max(), color="#0072B2", lw=1.2, label="flow loss (scaled)")
    ax.axvline(t, color="black", lw=1.5); ax.set_xlim(0, N - 1); ax.set_ylim(0, 1.05); ax.set_yticks([]); ax.set_xlabel("frame", fontsize=8); ax.tick_params(labelsize=7); ax.legend(fontsize=6.5, ncol=4, frameon=False, loc="upper left")
    fig.suptitle(f"pi0.5 on LIBERO — where the HT loss down-weights (red) and where flow's gradient is smallest (blue)   |   {gated.mean()*100:.0f}% of frames past the HT knee in this episode", fontsize=10)
    fig.savefig(f"{tmpdir}/f_{t:04d}.png", dpi=90); plt.close(fig)
subprocess.run(["ffmpeg", "-y", "-loglevel", "error", "-framerate", str(fps), "-i", f"{tmpdir}/f_%04d.png", "-c:v", "libx264", "-pix_fmt", "yuv420p", "-vf", "scale=trunc(iw/2)*2:trunc(ih/2)*2", out], check=True)
print("wrote", out, N, "frames; gated", int(gated.sum()), "lowgrad", int(lowgrad.sum()))
