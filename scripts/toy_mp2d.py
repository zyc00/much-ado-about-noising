"""2D replica of the MP tool-hang collection (script B structure).

Measured facts reproduced (from tool_hang_full2ins_2000, 30 demos):
  starts spread ~12mm; frame-pickup target T1 randomized (xy spread ~13mm);
  staging/hole target T2 common (<1mm spread); actions = capped proportional
  servo toward the CURRENT phase target; obs-triggered phase switch.

Task (units: mm): start (0, y0), y0 ~ N(0, 12). Phase 1: servo to
T1 = (90, g), g ~ U(-15, 15) (the frame). Phase 2: servo to T2 = (200, 0)
(the hole, fixed). Success: reach x >= 198 with |y| < 5.
Controller: a = clip(kp * (T - p), cap), kp = 0.3, cap = 4 mm/step.
Plant wobble 0.5mm/step on y; labels are the exact servo response.

obs = (x, y, g). Far from the target the servo saturates, so the effective
transverse gain of the DATA field is cap/dist (weak when far, strong near) —
the real script's structure.

Models: L2 | L2+condreg | MIP (two-pass action attraction).
Readouts: learned field maps, directional response spectra (PR), kicked
closed-loop rollouts, SR table (secondary).
"""
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn

torch.manual_seed(0)
np.random.seed(0)
dev = "cuda" if torch.cuda.is_available() else "cpu"

KP, CAP = 0.3, 4.0
T1X, T2 = 90.0, np.array([200.0, 0.0])
SIG_DYN = 0.5
EPS_SUCC = 5.0
N_DEMO = 30


def servo(p, t):
    return np.clip(KP * (t - p), -CAP, CAP)


def collect(g, y0, wobble=True, rng=None):
    p = np.array([0.0, y0])
    phase = 1
    rows = []
    for _ in range(240):
        t = np.array([T1X, g]) if phase == 1 else T2
        a = servo(p, t)
        rows.append([p[0], p[1], g, a[0], a[1], phase])
        p = p + a
        if wobble:
            p[1] += rng.normal(0, SIG_DYN)
        if phase == 1 and np.linalg.norm(p - np.array([T1X, g])) < 3.0:
            phase = 2
        if p[0] >= 198:
            break
    return np.array(rows)


rng = np.random.default_rng(7)
demos = [collect(g, 0.0, wobble=False, rng=rng)
         for g in np.linspace(-15, 15, N_DEMO)]
data = np.concatenate(demos)
S = torch.tensor(data[:, :3], dtype=torch.float32, device=dev)
A = torch.tensor(data[:, 3:5], dtype=torch.float32, device=dev)
S_MU, S_SD = S.mean(0), S.std(0) + 1e-6
A_SD = A.std()
print(f"data: {len(S)} transitions")


def norm_s(s):
    return (s - S_MU) / S_SD


def mlp(inp, out, h=64):
    return nn.Sequential(nn.Linear(inp, h), nn.SiLU(),
                         nn.Linear(h, h), nn.SiLU(), nn.Linear(h, out))


def train_l2(lam_cond=0.0, steps=6000):
    torch.manual_seed(1)
    net = mlp(3, 2).to(dev)
    opt = torch.optim.AdamW(net.parameters(), lr=1e-3, weight_decay=1e-5)
    for _ in range(steps):
        idx = torch.randint(0, len(S), (256,), device=dev)
        s, a = S[idx], A[idx]
        pred = net(norm_s(s))
        loss = ((pred - a) ** 2).mean()
        if lam_cond > 0:
            K = 6
            base = pred
            d2s = []
            for _ in range(K):
                v = torch.randn_like(s)
                v = v / (v.norm(dim=1, keepdim=True) + 1e-9) * 2.0  # 2mm probes
                d2s.append(((net(norm_s(s + v)) - base) ** 2).mean(dim=1))
            D = torch.stack(d2s, 1)
            cv2 = D.var(dim=1) / (D.mean(dim=1) ** 2 + 1e-12)
            loss = loss + lam_cond * torch.relu(cv2 - 1.0).mean()
        opt.zero_grad()
        loss.backward()
        opt.step()

    def fwd(s):
        return net(norm_s(s))
    return fwd


def train_mip(steps=6000, zsig=1.0):
    torch.manual_seed(1)
    net = mlp(5, 2).to(dev)
    opt = torch.optim.AdamW(net.parameters(), lr=1e-3, weight_decay=1e-5)
    for _ in range(steps):
        idx = torch.randint(0, len(S), (256,), device=dev)
        s, a = S[idx], A[idx]
        z = torch.randn_like(a) * zsig * A_SD
        y0 = net(torch.cat([z / A_SD, norm_s(s)], 1))
        y1 = net(torch.cat([y0.detach() / A_SD, norm_s(s)], 1))
        loss = ((y0 - a) ** 2).mean() + ((y1 - a) ** 2).mean()
        opt.zero_grad()
        loss.backward()
        opt.step()

    def fwd(s):
        z = torch.randn(s.shape[0], 2, device=dev) * zsig
        y0 = net(torch.cat([z, norm_s(s)], 1))
        return net(torch.cat([y0 / A_SD, norm_s(s)], 1))
    return fwd


models = {"L2": train_l2(), "L2+condreg": train_l2(lam_cond=3e-3),
          "MIP": train_mip()}
print("trained")


def rollout(f, g, y0, kick=0.0, kick_x=45.0, nmax=260):
    p = np.array([0.0, y0])
    kicked = False
    tr = [p.copy()]
    rr = np.random.default_rng(11)
    sign = 1.0 if rr.random() < 0.5 else -1.0
    while len(tr) < nmax and p[0] < 198 and abs(p[1]) < 80 \
            and -20 < p[0] < 220:
        if not kicked and p[0] >= kick_x:
            p[1] += sign * kick
            kicked = True
        s = torch.tensor([[p[0], p[1], g]], dtype=torch.float32, device=dev)
        with torch.no_grad():
            a = f(s)[0].cpu().numpy()
        p = p + a
        tr.append(p.copy())
    return (p[0] >= 198 and abs(p[1]) < EPS_SUCC), np.array(tr)


# SR table (secondary)
print("\nSR vs kick (mm), 20 episodes:")
kicks = [0, 10, 20, 40]
print("model        " + "  ".join(f"k={k:3d}" for k in kicks))
for name, f in models.items():
    row = []
    rr = np.random.default_rng(2)
    for k in kicks:
        succ = [rollout(f, g, 0.0, kick=k)[0]
                for g in np.linspace(-15, 15, 20)]
        row.append(np.mean(succ))
    print(f"{name:12s} " + "  ".join(f"{v:5.2f}" for v in row))

# ---------- figure 1: task/data ----------
fig, axes = plt.subplots(1, 2, figsize=(13, 4.6))
ax = axes[0]
for d in demos:
    ax.plot(d[:, 0], d[:, 1], lw=1.1,
            color=plt.cm.viridis((d[0, 2] + 15) / 30))
ax.scatter([T1X] * N_DEMO, [dd[0, 2] for dd in demos], marker="s", s=18,
           c=[plt.cm.viridis((dd[0, 2] + 15) / 30) for dd in demos],
           label="frame T1 (randomized)")
ax.plot(*T2, "k*", ms=18, label="hole T2 (fixed)")
ax.set_title("30 demos: spread starts -> per-episode frame -> common hole\n"
             "(single start, per-episode frame anchor, common hole; no noise)",
             fontsize=10.5)
ax.legend(fontsize=9)
ax.set_xlabel("x (mm)")
ax.set_ylabel("y (mm)")
ax = axes[1]
d = demos[3]
sub = d[::7]
ax.quiver(sub[:, 0], sub[:, 1], sub[:, 3], sub[:, 4], color="tab:blue",
          scale=90, width=0.004)
ax.plot(d[:, 0], d[:, 1], "k-", lw=0.7, alpha=0.5)
ax.plot(T1X, d[0, 2], "s", color="tab:green", ms=9)
ax.plot(*T2, "k*", ms=16)
ax.set_title("one demo, recorded actions: servo toward the CURRENT phase\n"
             "target; effective transverse gain = cap/dist (weak far, "
             "strong near)", fontsize=10.5)
ax.set_xlabel("x (mm)")
fig.tight_layout()
fig.savefig("analysis/paper/toy_mp2d_data.png", dpi=150,
            bbox_inches="tight")

# ---------- figure 2: the two mechanisms ----------
fig, axes = plt.subplots(2, 3, figsize=(17, 8.6))
G_FIX = 10.0
# row 1: learned fields at g=+10 (on/off tube)
for j, (name, f) in enumerate(models.items()):
    ax = axes[0, j]
    gx, gy = np.meshgrid(np.linspace(5, 195, 22), np.linspace(-45, 45, 15))
    U = np.zeros_like(gx)
    Vv = np.zeros_like(gy)
    with torch.no_grad():
        for i in range(gx.shape[0]):
            s = torch.tensor([[gx[i, jj], gy[i, jj], G_FIX]
                              for jj in range(gx.shape[1])],
                             dtype=torch.float32, device=dev)
            a = f(s).cpu().numpy()
            U[i], Vv[i] = a[:, 0], a[:, 1]
    mag = np.sqrt(U ** 2 + Vv ** 2)
    q = ax.quiver(gx, gy, U, Vv, np.minimum(mag, 8), cmap="coolwarm",
                  scale=140, width=0.0045)
    dref = collect(G_FIX, 0.0, wobble=False, rng=np.random.default_rng(0))
    ax.plot(dref[:, 0], dref[:, 1], "k-", lw=2, label="demo path (g=+10)")
    ax.plot(*T2, "k*", ms=14)
    ax.set_title(f"{name}: learned action field (color = |a|, cap was 4)",
                 fontsize=10)
    plt.colorbar(q, ax=ax, fraction=0.045)
    if j == 0:
        ax.set_ylabel("y (mm)")

# row 2 left: |a| vs distance off-tube; middle: response spectra; right: rollouts
ax = axes[1, 0]
dref = collect(G_FIX, 0.0, wobble=False, rng=np.random.default_rng(0))
mid = dref[len(dref) // 2, :2]
ds = np.linspace(-40, 40, 41)
for name, f in models.items():
    with torch.no_grad():
        s = torch.tensor([[mid[0], mid[1] + d, G_FIX] for d in ds],
                         dtype=torch.float32, device=dev)
        a = f(s).cpu().numpy()
    ax.plot(ds, np.linalg.norm(a, axis=1), label=name, lw=2)
ax.axhline(CAP, color="k", ls=":", lw=1, label="demo action cap")
ax.set_xlabel("transverse offset from tube (mm), mid-carry")
ax.set_ylabel("|a| (mm/step)")
ax.set_title("MIP mechanism: off-tube action magnitude stays at the\n"
             "demonstrated level; L2 extrapolates", fontsize=10)
ax.legend(fontsize=8.5)

ax = axes[1, 1]
probe = torch.tensor([[mid[0], mid[1] + 15.0, G_FIX]], dtype=torch.float32,
                     device=dev)
width = 0.25
for j, (name, f) in enumerate(models.items()):
    resp = []
    with torch.no_grad():
        base = f(probe)[0]
        for _ in range(64):
            v = torch.randn(1, 3, device=dev)
            v = v / v.norm() * 2.0
            resp.append(float((f(probe + v)[0] - base).norm()))
    resp = np.sort(resp)[::-1]
    pr = (resp.sum() ** 2) / ((resp ** 2).sum() * len(resp)) * len(resp)
    pr = (np.array(resp).sum() ** 2) / ((np.array(resp) ** 2).sum() + 1e-12)
    ax.bar(np.arange(12) + j * width, resp[:12], width=width, label=f"{name}"
           f" (PR {pr:.1f})")
ax.set_xlabel("direction rank (top 12 of 64 probes at 15mm off-tube)")
ax.set_ylabel("|response| to 2mm probe")
ax.set_title("PR mechanism: condreg flattens and bounds the response\n"
             "spectrum; the weak true servo survives as the dominant term",
             fontsize=10)
ax.legend(fontsize=8.5)

ax = axes[1, 2]
for name, f, c in [("L2", models["L2"], "tab:red"),
                   ("L2+condreg", models["L2+condreg"], "tab:blue"),
                   ("MIP", models["MIP"], "tab:green")]:
    for k in [20]:
        _, tr = rollout(f, G_FIX, 0.0, kick=k)
        ax.plot(tr[:, 0], tr[:, 1], color=c, lw=1.8, label=f"{name}")
ax.plot(dref[:, 0], dref[:, 1], "k--", lw=1, alpha=0.6, label="demo path")
ax.plot(*T2, "k*", ms=14)
ax.axhspan(-EPS_SUCC, EPS_SUCC, xmin=0.95, color="gray", alpha=0.4)
ax.set_title("closed loop, 20mm kick at mid-carry", fontsize=10)
ax.set_xlabel("x (mm)")
ax.legend(fontsize=8.5)
fig.suptitle("Mechanisms on the MP-structured toy: bounded off-tube action "
             "(MIP) and flattened response spectrum (condreg)", fontsize=12.5,
             y=1.00)
fig.tight_layout()
fig.savefig("analysis/paper/toy_mp2d_mech.png", dpi=150, bbox_inches="tight")
print("saved analysis/paper/toy_mp2d_data.png analysis/paper/toy_mp2d_mech.png")
print("TOY2-DONE")
