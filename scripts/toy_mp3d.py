"""3D version of the MP-structured toy: anchors diverge in y, the path then
converges in y to a common staging point, and the final phase is a descent
along z with xy held — matching the measured real structure (staging xy
spread <1mm, insertion = z-descent).

Phases (units mm), clean deterministic servo, no noise:
  P1 diverge:  start (0,0,0) -> anchor A = (60, g, 0), g ~ U(-15, 15)
  P2 merge:    A -> staging S = (120, 0, 0)   (common)
  P3 descend:  S -> insertion I = (120, 0, -40), xy held by the servo
Success: z <= -38 with |(x,y) - (120,0)| < 4.
Controller: a = clip(kp (T - p), +-cap), kp=0.3, cap=4.
obs = (x, y, z, g); action = (dx, dy, dz).

In the clean data, xy deviation during P3 is zero everywhere, so the xy
servo of P3 is unidentified off-manifold — kicks during descent probe that
regime; kicks during carry probe the anchor-schedule regime.

Models: L2 | L2+condreg | MIP (two-pass). Results first: SR table for
kicks applied at carry (x=90) and at descent (z=-15).
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
A_X, S_PT, I_PT = 60.0, np.array([120.0, 0.0, 0.0]), np.array([120.0, 0.0, -40.0])
EPS = 4.0
N_DEMO = 30


def servo(p, t):
    return np.clip(KP * (t - p), -CAP, CAP)


def collect(g):
    p = np.zeros(3)
    phase = 1
    rows = []
    for _ in range(400):
        t = (np.array([A_X, g, 0.0]) if phase == 1 else
             S_PT if phase == 2 else I_PT)
        a = servo(p, t)
        rows.append([*p, g, *a, phase])
        p = p + a
        if phase == 1 and np.linalg.norm(p - np.array([A_X, g, 0.0])) < 3:
            phase = 2
        elif phase == 2 and np.linalg.norm(p - S_PT) < 3:
            phase = 3
        if p[2] <= -38:
            break
    return np.array(rows)


demos = [collect(g) for g in np.linspace(-15, 15, N_DEMO)]
data = np.concatenate(demos)
S = torch.tensor(data[:, :4], dtype=torch.float32, device=dev)
A = torch.tensor(data[:, 4:7], dtype=torch.float32, device=dev)
S_MU, S_SD = S.mean(0), S.std(0) + 1e-6
A_SD = A.std()
print(f"data: {len(S)} transitions, {N_DEMO} demos")


def norm_s(s):
    return (s - S_MU) / S_SD


def mlp(inp, out, h=64):
    return nn.Sequential(nn.Linear(inp, h), nn.SiLU(),
                         nn.Linear(h, h), nn.SiLU(), nn.Linear(h, out))


def train_l2(lam_cond=0.0, steps=6000):
    torch.manual_seed(1)
    net = mlp(4, 3).to(dev)
    opt = torch.optim.AdamW(net.parameters(), lr=1e-3, weight_decay=1e-5)
    for _ in range(steps):
        idx = torch.randint(0, len(S), (256,), device=dev)
        s, a = S[idx], A[idx]
        pred = net(norm_s(s))
        loss = ((pred - a) ** 2).mean()
        if lam_cond > 0:
            base = pred
            d2s = []
            for _ in range(6):
                v = torch.randn_like(s)
                v = v / (v.norm(dim=1, keepdim=True) + 1e-9) * 2.0
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
    net = mlp(7, 3).to(dev)
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
        z = torch.randn(s.shape[0], 3, device=dev) * zsig
        y0 = net(torch.cat([z, norm_s(s)], 1))
        return net(torch.cat([y0 / A_SD, norm_s(s)], 1))
    return fwd


models = {"L2": train_l2(), "L2+condreg": train_l2(lam_cond=3e-3),
          "MIP": train_mip()}
print("trained")


def rollout(f, g, kick=0.0, where="carry", nmax=420):
    p = np.zeros(3)
    kicked = False
    tr = [p.copy()]
    sign = 1.0 if (hash((round(g * 10), round(kick))) % 2 == 0) else -1.0
    while (len(tr) < nmax and p[2] > -38 and abs(p[1]) < 80
           and -20 < p[0] < 200 and p[2] < 30):
        trigger = (p[0] >= 90) if where == "carry" else (p[2] <= -15)
        if not kicked and trigger:
            p[1] += sign * kick
            kicked = True
        s = torch.tensor([[*p, g]], dtype=torch.float32, device=dev)
        with torch.no_grad():
            a = f(s)[0].cpu().numpy()
        p = p + a
        tr.append(p.copy())
    succ = (p[2] <= -38 and np.linalg.norm(p[:2] - S_PT[:2]) < EPS)
    return succ, np.array(tr)


print("\nSR vs kick (mm), 20 episodes each; kick at CARRY (x=90):")
kicks = [0, 5, 10, 20]
print("model        " + "  ".join(f"k={k:3d}" for k in kicks))
for name, f in models.items():
    row = [np.mean([rollout(f, g, k, "carry")[0]
                    for g in np.linspace(-15, 15, 20)]) for k in kicks]
    print(f"{name:12s} " + "  ".join(f"{v:5.2f}" for v in row))
print("kick at DESCENT (z=-15):")
for name, f in models.items():
    row = [np.mean([rollout(f, g, k, "descent")[0]
                    for g in np.linspace(-15, 15, 20)]) for k in kicks]
    print(f"{name:12s} " + "  ".join(f"{v:5.2f}" for v in row))

# figures: 3D data + descent-kick rollouts
fig = plt.figure(figsize=(15, 5))
ax = fig.add_subplot(131, projection="3d")
for d in demos:
    ax.plot(d[:, 0], d[:, 1], d[:, 2], lw=1.0,
            color=plt.cm.viridis((d[0, 3] + 15) / 30))
ax.scatter([A_X] * N_DEMO, np.linspace(-15, 15, N_DEMO), 0, marker="s",
           s=14, c="tab:green")
ax.scatter(*I_PT, marker="*", s=120, c="k")
ax.set_title("30 demos: diverge in y (anchors) ->\nmerge -> descend on z "
             "(common insertion)", fontsize=10)
ax.set_xlabel("x")
ax.set_ylabel("y")
ax.set_zlabel("z")
for j, (name, f) in enumerate([("L2+condreg", models["L2+condreg"]),
                               ("L2", models["L2"])]):
    ax = fig.add_subplot(1, 3, 2 + j, projection="3d")
    for k, c in [(0, "tab:green"), (10, "tab:orange")]:
        for g in [-12, 0, 12]:
            _, tr = rollout(f, g, k, "descent")
            ax.plot(tr[:, 0], tr[:, 1], tr[:, 2], color=c, lw=1.4)
    ax.scatter(*I_PT, marker="*", s=120, c="k")
    ax.set_title(f"{name}: descent kicks 0 (green) / 10mm (orange)",
                 fontsize=10)
    ax.set_xlabel("x")
    ax.set_ylabel("y")
    ax.set_zlabel("z")
fig.tight_layout()
fig.savefig("analysis/paper/toy_mp3d.png", dpi=150, bbox_inches="tight")
print("saved analysis/paper/toy_mp3d.png")
print("TOY3D-DONE")

# ---------------- 2x2(+capacity) : {MLP, FiLM} x {L2, MIP} ----------------
class FiLMNet(nn.Module):
    """chiunet-style conditioning: action-input pathway modulated per-layer
    by scale/shift computed from the state."""

    def __init__(self, a_dim=3, s_dim=4, h=64, blocks=2):
        super().__init__()
        self.inp = nn.Linear(a_dim, h)
        self.blocks = nn.ModuleList(nn.Linear(h, h) for _ in range(blocks))
        self.gammas = nn.ModuleList(nn.Sequential(
            nn.Linear(s_dim, h), nn.SiLU(), nn.Linear(h, h))
            for _ in range(blocks))
        self.betas = nn.ModuleList(nn.Sequential(
            nn.Linear(s_dim, h), nn.SiLU(), nn.Linear(h, h))
            for _ in range(blocks))
        self.out = nn.Linear(h, 3)

    def forward(self, a_in, s):
        hdn = torch.nn.functional.silu(self.inp(a_in))
        for blk, gm, bt in zip(self.blocks, self.gammas, self.betas):
            hdn = torch.nn.functional.silu(
                blk(hdn) * torch.tanh(gm(s)) + bt(s))
        return self.out(hdn)


def train_cell(arch, obj, h, blocks=2, steps=8000):
    torch.manual_seed(1)
    if arch == "film":
        net = FiLMNet(h=h, blocks=blocks).to(dev)

        def call(a_in, s):
            return net(a_in, norm_s(s))
    else:
        net = mlp(7, 3, h=h).to(dev)

        def call(a_in, s):
            return net(torch.cat([a_in, norm_s(s)], 1))
    opt = torch.optim.AdamW(net.parameters(), lr=1e-3, weight_decay=1e-5)
    for _ in range(steps):
        idx = torch.randint(0, len(S), (256,), device=dev)
        s, a = S[idx], A[idx]
        if obj == "l2":
            pred = call(torch.zeros_like(a), s)
            loss = ((pred - a) ** 2).mean()
        else:
            z = torch.randn_like(a)
            y0 = call(z, s)
            y1 = call(y0.detach() / A_SD, s)
            loss = ((y0 - a) ** 2).mean() + ((y1 - a) ** 2).mean()
        opt.zero_grad()
        loss.backward()
        opt.step()

    def fwd(s, obj=obj):
        if obj == "l2":
            return call(torch.zeros(s.shape[0], 3, device=dev), s)
        z = torch.randn(s.shape[0], 3, device=dev)
        y0 = call(z, s)
        return call(y0 / A_SD, s)
    return fwd


cells = {}
for arch in ["mlp", "film"]:
    for obj in ["l2", "mip"]:
        for h, hb in [(64, 2), (256, 3)]:
            tag = f"{arch}-{obj}-h{h}"
            cells[tag] = train_cell(arch, obj, h, blocks=hb)
print("\n2x2xCAP SR (carry kicks 0/10/20 | descent kicks 0/10/20):")
for tag, f in cells.items():
    rc = [np.mean([rollout(f, g, k, "carry")[0]
                   for g in np.linspace(-15, 15, 20)]) for k in [0, 10, 20]]
    rd = [np.mean([rollout(f, g, k, "descent")[0]
                   for g in np.linspace(-15, 15, 20)]) for k in [0, 10, 20]]
    print(f"CELL {tag:16s} " + "/".join(f"{v:.2f}" for v in rc) + "  |  "
          + "/".join(f"{v:.2f}" for v in rd))
print("GRID-DONE")

# budget sensitivity + mechanism readout for the h256 pair
for st in [4000, 6000, 8000, 12000]:
    f = train_cell("mlp", "mip", 64, steps=st)
    rc = np.mean([rollout(f, g, 10, "carry")[0]
                  for g in np.linspace(-15, 15, 20)])
    print(f"BUDGET mlp-mip-h64 steps={st}: SR(carry k=10) {rc:.2f}")

fL = train_cell("mlp", "l2", 256, blocks=3, steps=12000)
fM = train_cell("mlp", "mip", 256, blocks=3, steps=12000)
dref = collect(10.0)
mid = dref[np.argmax(dref[:, 5] == 2)]     # first merge-phase row
probe_x, probe_y = 95.0, 0.0
ds_off = np.linspace(-40, 40, 33)
fig2, ax2 = plt.subplots(1, 2, figsize=(11, 4.2))
for name, f, c in [("L2 h256", fL, "tab:red"), ("MIP h256", fM, "tab:green")]:
    mags = []
    with torch.no_grad():
        for d in ds_off:
            s0 = torch.tensor([[probe_x, d, 0.0, 10.0]],
                              dtype=torch.float32, device=dev)
            mags.append(float(f(s0)[0].norm()))
    ax2[0].plot(ds_off, mags, color=c, lw=2, label=name)
    zmags = []
    with torch.no_grad():
        for d in ds_off:
            s0 = torch.tensor([[120.0, d, -20.0, 10.0]],
                              dtype=torch.float32, device=dev)
            zmags.append(float(f(s0)[0].norm()))
    ax2[1].plot(ds_off, zmags, color=c, lw=2, ls="--",
                label=name + " (descent)")
for ax in ax2:
    ax.axhline(CAP, color="k", ls=":", lw=1)
    ax.set_xlabel("transverse offset y (mm)")
    ax.set_ylabel("|a| (mm/step)")
ax2[0].set_title("off-tube |a| at carry (x=95)")
ax2[1].set_title("off-tube |a| at descent (z=-20)")
ax2[0].legend()
ax2[1].legend()
fig2.suptitle("h256 pair: off-support action magnitude", y=1.02)
fig2.tight_layout()
fig2.savefig("analysis/paper/toy_mp3d_mag.png", dpi=150,
             bbox_inches="tight")
with torch.no_grad():
    for d in [10.0, 20.0, 40.0]:
        sC = torch.tensor([[probe_x, d, 0.0, 10.0]], dtype=torch.float32,
                          device=dev)
        aL = fL(sC)[0].cpu().numpy()
        aM = fM(sC)[0].cpu().numpy()
        print(f"MAG carry y={d:.0f}: L2 |a|={np.linalg.norm(aL):.1f} "
              f"a_y={aL[1]:+.1f} | MIP |a|={np.linalg.norm(aM):.1f} "
              f"a_y={aM[1]:+.1f}")
print("MECH-DONE")
