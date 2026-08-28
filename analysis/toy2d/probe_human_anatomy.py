"""Failure/dynamics anatomy for the story-proof toy (canonical cell).
P1 drift profile |x-c(y)| vs y (l2 fail/succ vs ht), P2 support distance
vs y, P3 learned lateral law at dogleg and below-funnel slices,
P4 signal-error training curves, P5 train-loss vs noise floor,
P6 ledger + sigma formation. Uses median seeds (l2:7, ht:2)."""
import os

os.environ.setdefault("S_HI", "0.08")
os.environ.setdefault("NEP", "40")
os.environ.setdefault("DOCK_TOL", "0.008")
os.environ.setdefault("STEPS", "25000")

import matplotlib

matplotlib.use("Agg")
import copy
import matplotlib.pyplot as plt
import numpy as np
import torch

import toyhuman as T

X, Y, Yc, AL = T.build_dataset(np.random.RandomState(0))
norm = T.Norm(X, Y)
Xn, Yn, Ycn = norm.nx(X), norm.ny(Y), norm.ny(Yc)
CL = (~AL) & (X[:, 1] > T.Y_DOCK) & (X[:, 1] < T.Y_ALIGN_LO)
Xt = torch.tensor(Xn, device=T.DEV)
Yt = torch.tensor(Yn, device=T.DEV)
MEDIAN_SEED = {"l2": 7, "ht": 2}

# ---------- instrumented retrain (P4-P6) -----------------------------------
def train_instrumented(arm, seed):
    torch.manual_seed(seed)
    net = T.Reg(hetero=(arm == "ht")).to(T.DEV)
    opt = torch.optim.Adam(net.parameters(), lr=1e-3)
    rng = np.random.RandomState(seed)
    hist = {"step": [], "train_loss": [], "sigerr_al": [], "sigerr_cl": [],
            "led_al": [], "sig_ratio": [], "mse_al": [], "mse_cl": []}
    for it in range(T.STEPS):
        idx = rng.randint(0, len(Xt), 256)
        loss = T.loss_fn(arm, net, Xt[idx], Yt[idx])
        opt.zero_grad()
        loss.backward()
        opt.step()
        if it % 500 == 0 or it == T.STEPS - 1:
            with torch.no_grad():
                pred = net(Xt).cpu().numpy()
                mse_lab = ((pred - Yn) ** 2).mean(1)
                sig = np.abs(pred - Ycn).mean(1)
            w = T.weight_of(arm, net, Xt, Yt).cpu().numpy()
            hist["step"].append(it)
            hist["train_loss"].append(float(loss.item()))
            hist["mse_al"].append(float(mse_lab[AL].mean()))
            hist["mse_cl"].append(float(mse_lab[CL].mean()))
            hist["sigerr_al"].append(float(sig[AL].mean()))
            hist["sigerr_cl"].append(float(sig[CL].mean()))
            hist["led_al"].append(float(w[AL].sum() / (w.sum() + 1e-12)))
            if arm == "ht":
                with torch.no_grad():
                    s = T.sigma_of(net, Xt).cpu().numpy()
                hist["sig_ratio"].append(float(s[AL].mean() / (s[CL].mean() + 1e-12)))
            else:
                hist["sig_ratio"].append(np.nan)
    return net, hist


H = {}
for arm in ("l2", "ht"):
    _, H[arm] = train_instrumented(arm, MEDIAN_SEED[arm])
    print(f"{arm} dynamics done", flush=True)

# ---------- rollout anatomy (P1-P2) ----------------------------------------
nets = {a: torch.load(f"human_ck_{a}_med.pt", map_location=T.DEV,
                      weights_only=False) for a in ("l2", "ht")}


def rollout_traj(net, ep):
    p = ep.copy()
    tr = [p.copy()]
    ok = False
    for _ in range(150 // T.H):
        on = torch.tensor(norm.nx(p[None].astype(np.float32)), device=T.DEV)
        with torch.no_grad():
            chunk = norm.uy(net(on)[0].cpu().numpy()).reshape(T.H, 2)
        for a in chunk:
            p = T.wall(p + np.clip(a, -T.CLIP, T.CLIP))
            tr.append(p.copy())
            if np.linalg.norm(p - T.G) < T.DOCK_TOL:
                ok = True
                break
            if p[1] < -0.05:
                break
        if ok or p[1] < -0.05:
            break
    return np.array(tr), ok


rng = np.random.RandomState(999)
eps = [np.array([rng.uniform(-0.2, 0.2), rng.uniform(0.8, 1.0)])
       for _ in range(100)]
TR = {a: [rollout_traj(nets[a], e) for e in eps] for a in ("l2", "ht")}
for a in ("l2", "ht"):
    print(f"{a}: SR {np.mean([ok for _, ok in TR[a]]):.2f}", flush=True)

YBINS = np.linspace(-0.02, 0.5, 27)


def binned(trajs, fn):
    out = np.full((len(trajs), len(YBINS) - 1), np.nan)
    for i, tr in enumerate(trajs):
        yy, vv = tr[:, 1], fn(tr)
        for b in range(len(YBINS) - 1):
            m = (yy >= YBINS[b]) & (yy < YBINS[b + 1])
            if m.any():
                out[i, b] = vv[m].mean()
    return out


def lat_off(tr):
    return np.abs(tr[:, 0] - np.array([T.center(y) for y in tr[:, 1]]))


def sup_dist(tr):
    d = np.sqrt(((tr[:, None, :] - X[None, :, :]) ** 2).sum(-1))
    return d.min(1)


groups = {
    "l2 fail": [tr for tr, ok in TR["l2"] if not ok],
    "l2 succ": [tr for tr, ok in TR["l2"] if ok],
    "ht": [tr for tr, ok in TR["ht"]],
}
GC = {"l2 fail": "#d62728", "l2 succ": "#f2a5a5", "ht": "#9467bd"}
print("group sizes:", {k: len(v) for k, v in groups.items()}, flush=True)

# divergence stats for l2 failures
div_y = []
for tr in groups["l2 fail"]:
    lo = lat_off(tr)
    inb = tr[:, 1] < 0.5
    bad = np.where(inb & (lo > 0.012))[0]
    div_y.append(tr[bad[0], 1] if len(bad) else np.nan)
div_y = np.array(div_y)
print(f"l2 failure first-divergence y: median {np.nanmedian(div_y):.3f}, "
      f"IQR {np.nanpercentile(div_y, 25):.3f}-{np.nanpercentile(div_y, 75):.3f}",
      flush=True)

# miss distance at dock height for failures
md = [np.abs(tr[np.argmin(np.abs(tr[:, 1] - 0.02)), 0])
      for tr in groups["l2 fail"]]
print(f"l2 failure |x| at y~0.02: mean {np.mean(md):.4f} (tol {T.DOCK_TOL})",
      flush=True)

# noise floor
floor = float(((Yn - Ycn) ** 2)[AL].mean())
print(f"align label-noise floor (normalized MSE): {floor:.4f}", flush=True)

# ---------- figure ----------------------------------------------------------
fig = plt.figure(figsize=(13.5, 10.5))
gs = fig.add_gridspec(2, 3, hspace=0.42, wspace=0.32,
                      left=0.06, right=0.98, top=0.90, bottom=0.07)
fig.suptitle("MSE failure anatomy + training dynamics (canonical cell, median seeds)",
             fontsize=12.5, y=0.97)
yc_ = 0.5 * (YBINS[1:] + YBINS[:-1])

axP1 = fig.add_subplot(gs[0, 0])
for g, trs in groups.items():
    if not trs:
        continue
    B = binned(trs, lat_off)
    axP1.plot(np.nanmean(B, 0), yc_, color=GC[g], lw=2, label=g)
    axP1.fill_betweenx(yc_, np.nanmean(B, 0) - np.nanstd(B, 0),
                       np.nanmean(B, 0) + np.nanstd(B, 0), color=GC[g], alpha=0.15)
axP1.axhspan(0.25, 0.5, color="#d62728", alpha=0.06)
axP1.axhspan(0.05, 0.25, color="#2ca02c", alpha=0.06)
axP1.axhline(T.Y_DOCK, color="0.5", lw=0.8, ls=":")
axP1.axvline(T.DOCK_TOL, color="k", lw=0.8, ls=":")
axP1.set_xlabel("|x − center(y)|")
axP1.set_ylabel("height y")
axP1.set_xlim(0, 0.05)
axP1.legend(fontsize=8)
axP1.set_title("P1  drift profile: where failures deviate", fontsize=10, loc="left")

axP2 = fig.add_subplot(gs[0, 1])
for g, trs in groups.items():
    if not trs:
        continue
    B = binned(trs, sup_dist)
    axP2.plot(np.nanmean(B, 0), yc_, color=GC[g], lw=2, label=g)
axP2.axhline(T.Y_DOCK, color="0.5", lw=0.8, ls=":")
axP2.set_xlabel("distance to nearest training state")
axP2.set_ylabel("height y")
axP2.legend(fontsize=8)
axP2.set_title("P2  support distance: off-support at the\nfunnel exit?", fontsize=10,
               loc="left")

axP3 = fig.add_subplot(gs[0, 2])
xs = np.linspace(-0.05, 0.05, 41)
for yq, ls in ((0.12, "-"), (0.03, "--")):
    for a, c in (("l2", "#d62728"), ("ht", "#9467bd")):
        q = np.stack([xs + T.center(yq), np.full_like(xs, yq)], 1).astype(np.float32)
        with torch.no_grad():
            pred = norm.uy(nets[a](torch.tensor(norm.nx(q), device=T.DEV)).cpu().numpy())
        axP3.plot(xs, pred[:, 0], color=c, ls=ls, lw=1.8,
                  label=f"{a} @y={yq}" if True else None)
    ex = [T.expert_action(np.array([x + T.center(yq), yq]))[0] for x in xs]
    axP3.plot(xs, ex, color="k", ls=ls, lw=1.0, alpha=0.6, label=f"expert @y={yq}")
axP3.axvline(0, color="0.8", lw=0.8)
axP3.set_xlabel("lateral offset from center  (x − c(y))")
axP3.set_ylabel("emitted a_x")
axP3.legend(fontsize=6.5, ncol=2)
axP3.set_title("P3  the learned lateral law: dogleg slice (solid)\nand below-funnel slice (dashed)",
               fontsize=10, loc="left")

axP4 = fig.add_subplot(gs[1, 0])
for a, c in (("l2", "#d62728"), ("ht", "#9467bd")):
    axP4.plot(H[a]["step"], H[a]["sigerr_al"], color=c, lw=1.8, label=f"{a} align")
    axP4.plot(H[a]["step"], H[a]["sigerr_cl"], color=c, lw=1.8, ls="--",
              label=f"{a} clean")
axP4.set_yscale("log")
axP4.set_xlabel("training step")
axP4.set_ylabel("error to CLEAN signal (norm.)")
axP4.legend(fontsize=7.5)
axP4.set_title("P4  signal fit: l2's align fit stalls high\n(chasing tails); ht converges",
               fontsize=10, loc="left")

axP5 = fig.add_subplot(gs[1, 1])
for a, c in (("l2", "#d62728"), ("ht", "#9467bd")):
    axP5.plot(H[a]["step"], H[a]["mse_al"], color=c, lw=1.8, label=f"{a} align MSE")
    axP5.plot(H[a]["step"], H[a]["mse_cl"], color=c, lw=1.4, ls="--",
              label=f"{a} clean MSE")
axP5.axhline(floor, color="k", lw=1.2, ls=":")
axP5.text(T.STEPS * 0.55, floor * 1.15, "align noise floor", fontsize=8)
axP5.set_yscale("log")
axP5.set_xlabel("training step")
axP5.set_ylabel("MSE to labels (norm.)")
axP5.legend(fontsize=7.5)
axP5.set_title("P5  loss convergence: l2 plateaus AT the\nirreducible floor (converged, still failing)",
               fontsize=10, loc="left")

axP6 = fig.add_subplot(gs[1, 2])
for a, c in (("l2", "#d62728"), ("ht", "#9467bd")):
    axP6.plot(H[a]["step"], H[a]["led_al"], color=c, lw=1.8, label=f"{a} align grad share")
axP6.set_ylabel("align share of gradient weight")
ax2 = axP6.twinx()
ax2.plot(H["ht"]["step"], H["ht"]["sig_ratio"], color="#2ca02c", lw=1.6)
ax2.set_ylabel("ht sigma ratio (align/clean)", color="#2ca02c")
axP6.set_xlabel("training step")
axP6.legend(fontsize=7.5, loc="center right")
axP6.set_title("P6  formation: ht's sigma map forms early\nand kills the align gradient share",
               fontsize=10, loc="left")

fig.savefig("human_anatomy_fig.png", dpi=140)
print("wrote human_anatomy_fig.png", flush=True)
