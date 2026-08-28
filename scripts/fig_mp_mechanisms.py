"""One-figure summary of the MP-geometry toy findings.

Panels:
  A  the task (3D): common start -> per-episode anchors -> merge ->
     z-descent to common insertion; clean servo, zero noise.
  B  MIP mechanism: off-support action magnitude (log scale) — L2
     extrapolation grows ~90x the demonstrated cap at 40mm; MIP stays
     within ~16x and proportionate; inset: the training-budget transition
     of the MIP cell (measured SRs).
  C  PR mechanism: unkicked rollouts — L2 (h64) drifts along unidentified
     directions and misses; +condreg holds the corridor (SR 0.20 -> 0.85).
  D  closed-loop summary: SR of the four key cells at takeover and 10mm
     kicks (carry and descent).
"""
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
A_X = 60.0
S_PT = np.array([120.0, 0.0, 0.0])
I_PT = np.array([120.0, 0.0, -40.0])
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
        rows.append([*p, g, *a])
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


def norm_s(s):
    return (s - S_MU) / S_SD


def mlp(inp, out, h):
    return nn.Sequential(nn.Linear(inp, h), nn.SiLU(),
                         nn.Linear(h, h), nn.SiLU(), nn.Linear(h, out))


def train(obj, h, lam_cond=0.0, steps=12000):
    torch.manual_seed(1)
    net = mlp(7, 3, h).to(dev)
    opt = torch.optim.AdamW(net.parameters(), lr=1e-3, weight_decay=1e-5)

    def call(a_in, s):
        return net(torch.cat([a_in, norm_s(s)], 1))
    for _ in range(steps):
        idx = torch.randint(0, len(S), (256,), device=dev)
        s, a = S[idx], A[idx]
        if obj == "l2":
            pred = call(torch.zeros_like(a), s)
            loss = ((pred - a) ** 2).mean()
            if lam_cond > 0:
                d2s = []
                for _ in range(6):
                    v = torch.randn_like(s)
                    v = v / (v.norm(dim=1, keepdim=True) + 1e-9) * 2.0
                    d2s.append(((call(torch.zeros_like(a), s + v)
                                 - pred) ** 2).mean(dim=1))
                D = torch.stack(d2s, 1)
                cv2 = D.var(dim=1) / (D.mean(dim=1) ** 2 + 1e-12)
                loss = loss + lam_cond * torch.relu(cv2 - 1.0).mean()
        else:
            z = torch.randn_like(a)
            y0 = call(z, s)
            y1 = call(y0.detach() / A_SD, s)
            loss = ((y0 - a) ** 2).mean() + ((y1 - a) ** 2).mean()
        opt.zero_grad()
        loss.backward()
        opt.step()

    def fwd(s):
        if obj == "l2":
            return call(torch.zeros(s.shape[0], 3, device=dev), s)
        z = torch.randn(s.shape[0], 3, device=dev)
        y0 = call(z, s)
        return call(y0 / A_SD, s)
    return fwd


print("training cells...")
cells = {
    "L2 (h64)": train("l2", 64, steps=6000),
    "L2+condreg (h64)": train("l2", 64, lam_cond=3e-3, steps=6000),
    "L2 (h256)": train("l2", 256),
    "MIP (h256)": train("mip", 256),
}


def rollout(f, g, kick=0.0, where="carry", nmax=420):
    p = np.zeros(3)
    kicked = False
    tr = [p.copy()]
    sign = 1.0 if (hash((round(g * 10), round(kick))) % 2 == 0) else -1.0
    while (len(tr) < nmax and p[2] > -38 and abs(p[1]) < 80
           and -20 < p[0] < 200 and p[2] < 30):
        trig = (p[0] >= 90) if where == "carry" else (p[2] <= -15)
        if not kicked and kick > 0 and trig:
            p[1] += sign * kick
            kicked = True
        s = torch.tensor([[*p, g]], dtype=torch.float32, device=dev)
        with torch.no_grad():
            a = f(s)[0].cpu().numpy()
        p = p + a
        tr.append(p.copy())
    succ = (p[2] <= -38 and np.linalg.norm(p[:2] - S_PT[:2]) < EPS)
    return succ, np.array(tr)


def sr(f, kick, where):
    return np.mean([rollout(f, g, kick, where)[0]
                    for g in np.linspace(-15, 15, 20)])


print("evaluating...")
SRD = {name: [sr(f, 0, "carry"), sr(f, 10, "carry"), sr(f, 10, "descent")]
       for name, f in cells.items()}
for k, v in SRD.items():
    print("SR", k, [round(x, 2) for x in v])

fig = plt.figure(figsize=(21, 9.5))

# A task
ax = fig.add_subplot(231, projection="3d")
for d in demos:
    ax.plot(d[:, 0], d[:, 1], d[:, 2], lw=1.0,
            color=plt.cm.viridis((d[0, 3] + 15) / 30))
ax.scatter([A_X] * N_DEMO, np.linspace(-15, 15, N_DEMO), 0, marker="s",
           s=16, c="tab:green", label="per-episode anchor")
ax.scatter(*I_PT, marker="*", s=160, c="k", label="common insertion")
ax.view_init(elev=22, azim=-60)
ax.set_title("A  task: common start $\\to$ anchors $\\to$ merge $\\to$ "
             "z-descent\nclean capped servo, zero noise", fontsize=11)
ax.set_xlabel("x")
ax.set_ylabel("y")
ax.set_zlabel("z")
ax.legend(fontsize=8.5, loc="upper left")

# B MIP mechanism: |a| + budget inset
ax = fig.add_subplot(232)
offs = np.linspace(-40, 40, 41)
for name, c in [("L2 (h256)", "tab:red"), ("MIP (h256)", "tab:green")]:
    mags = []
    with torch.no_grad():
        for d in offs:
            s0 = torch.tensor([[95.0, d, 0.0, 10.0]], dtype=torch.float32,
                              device=dev)
            mags.append(float(cells[name](s0)[0].norm()))
    ax.semilogy(offs, mags, color=c, lw=2.5, label=name)
ax.axhline(CAP, color="k", ls=":", lw=1.5, label="demonstrated cap (4)")
ax.fill_betweenx([0.5, 500], -15, 15, color="tab:orange", alpha=0.12)
ax.set_ylim(1, 500)
ax.set_xlabel("transverse offset (mm), mid-carry probe")
ax.set_ylabel("|a| (mm/step, log)")
ax.set_title("B  why MIP helps: off-support |a| stays near the\n"
             "demonstrated scale; L2 reaches ~100x the cap", fontsize=11)
axi = ax.inset_axes([0.60, 0.13, 0.37, 0.30])
axi.plot([4, 6, 8, 12], [0.10, 0.00, 0.95, 1.00], "o-", color="tab:green")
axi.set_title("MIP SR vs training steps (k):\nrequires capacity/budget",
              fontsize=8)
axi.set_ylim(-0.05, 1.1)
axi.grid(alpha=0.3)
ax.legend(fontsize=8.5, loc="upper left")

# C summary bars
ax = fig.add_subplot(233)
names = list(SRD.keys())
xpos = np.arange(len(names))
wd = 0.25
for i, (lab, c) in enumerate(zip(
        ["takeover", "carry kick 10mm", "descent kick 10mm"],
        ["tab:gray", "tab:orange", "tab:purple"])):
    ax.bar(xpos + (i - 1) * wd, [SRD[n][i] for n in names], wd, label=lab,
           color=c)
ax.set_xticks(xpos)
ax.set_xticklabels([n.replace(" ", "\n") for n in names], fontsize=8.5)
ax.set_ylabel("SR")
ax.set_ylim(0, 1.12)
ax.legend(fontsize=8.5)
ax.set_title("C  closed-loop summary: two repair routes", fontsize=11)
ax.grid(axis="y", alpha=0.3)

# D drift failure in the z view (where it actually happens)
ax = fig.add_subplot(234)
for name, c in [("L2 (h64)", "tab:red"), ("L2+condreg (h64)", "tab:blue")]:
    for g in np.linspace(-15, 15, 8):
        _, tr = rollout(cells[name], g, 0.0)
        ax.plot(tr[:, 0], tr[:, 2], color=c, lw=1.2, alpha=0.8)
        if tr[-1, 2] >= 30:
            ax.plot(tr[-1, 0], tr[-1, 2], "x", color=c, ms=9, mew=2.5)
ax.plot([120], [-40], "k*", ms=16)
ax.axhline(0, color="k", lw=0.8, alpha=0.5)
ax.set_xlabel("x (mm)")
ax.set_ylabel("z (mm)")
ax.set_title("D  why condreg helps — unkicked rollouts, (x, z) view:\n"
             "L2 (red) escapes UPWARD in z (the unidentified direction), "
             f"SR {SRD['L2 (h64)'][0]:.2f};\n+condreg (blue) holds z and "
             f"descends, SR {SRD['L2+condreg (h64)'][0]:.2f}", fontsize=10.5)

# E overshoot: kicked y(t)
ax = fig.add_subplot(235)
for name, c in [("L2 (h256)", "tab:red"), ("MIP (h256)", "tab:green")]:
    _, tr = rollout(cells[name], 10.0, 10.0, "carry")
    i0 = np.argmax(np.abs(np.diff(tr[:, 1])) > 8)   # kick step
    t = np.arange(len(tr)) - i0
    ax.plot(t, tr[:, 1], color=c, lw=2, label=name)
ax.axhline(0, color="k", lw=0.8, alpha=0.5)
ax.set_xlim(-5, 40)
ax.set_xlabel("steps since 10mm kick (carry)")
ax.set_ylabel("y (mm)")
ax.set_title("E  overshoot: L2's ~6x-cap response overshoots and\n"
             "oscillates; MIP returns proportionately", fontsize=11)
ax.legend(fontsize=9)

# F kicked spatial view
ax = fig.add_subplot(236)
for name, c in [("L2 (h256)", "tab:red"), ("MIP (h256)", "tab:green")]:
    for g in [-10, 10]:
        _, tr = rollout(cells[name], g, 10.0, "carry")
        ax.plot(tr[:, 0], tr[:, 1], color=c, lw=1.6, alpha=0.85)
        if not (tr[-1, 2] <= -38 and
                np.linalg.norm(tr[-1, :2] - S_PT[:2]) < EPS):
            ax.plot(tr[-1, 0], tr[-1, 1], "x", color=c, ms=10, mew=2.5)
ax.plot([120], [0], "k*", ms=16)
ax.set_xlabel("x (mm)")
ax.set_ylabel("y (mm)")
ax.set_title("F  kicked rollouts (10mm at carry), (x, y):\n"
             "L2 oscillates and exits; MIP settles and inserts", fontsize=11)

fig.suptitle("Script-data mechanisms on the MP-geometry toy: the failure is "
             "the unidentified off-support response; condreg and MIP repair "
             "it by different routes", fontsize=13.5, y=0.995)
fig.tight_layout()
out = "analysis/paper/mp_mechanisms_fig.png"
fig.savefig(out, dpi=150, bbox_inches="tight")
print("saved", out)

# ---- why does condreg stop the drift? channel decomposition ----
def train_shellfit(h=64, steps=6000):
    """control: fit the labels in the 2mm shell around the data (input-
    perturbation smoothing), WITHOUT the CV^2 spectral term."""
    torch.manual_seed(1)
    net = mlp(7, 3, h).to(dev)
    opt = torch.optim.AdamW(net.parameters(), lr=1e-3, weight_decay=1e-5)

    def call(a_in, s):
        return net(torch.cat([a_in, norm_s(s)], 1))
    for _ in range(steps):
        idx = torch.randint(0, len(S), (256,), device=dev)
        s, a = S[idx], A[idx]
        pred = call(torch.zeros_like(a), s)
        loss = ((pred - a) ** 2).mean()
        for _ in range(2):
            v = torch.randn_like(s)
            v = v / (v.norm(dim=1, keepdim=True) + 1e-9) * 2.0
            loss = loss + 0.5 * ((call(torch.zeros_like(a), s + v)
                                  - a) ** 2).mean()
        opt.zero_grad()
        loss.backward()
        opt.step()

    def fwd(s):
        return call(torch.zeros(s.shape[0], 3, device=dev), s)
    return fwd


probe_models = {"L2 (h64)": cells["L2 (h64)"],
                "L2+condreg (h64)": cells["L2+condreg (h64)"],
                "shell-fit (h64)": train_shellfit()}
print("\nz-response anatomy at mid-carry (x=95, y=10, g=10):")
print(f"{'model':18s} a_z(z=0)   da_z/dz @z=0/+3/+8   SR(takeover)")
for name, f in probe_models.items():
    row = []
    for z0 in [0.0, 3.0, 8.0]:
        sp = torch.tensor([[95.0, 10.0, z0, 10.0]], dtype=torch.float32,
                          device=dev, requires_grad=True)
        az = f(sp)[0, 2]
        gz = float(torch.autograd.grad(az, sp)[0][0, 2])
        row.append(gz)
        if z0 == 0.0:
            bias = float(az)
    srv = np.mean([rollout(f, g, 0.0)[0]
                   for g in np.linspace(-15, 15, 20)])
    print(f"{name:18s} {bias:+.3f}     "
          + "  ".join(f"{g:+.3f}" for g in row) + f"     {srv:.2f}")
print("DECOMP-DONE")

# ---- rank / direction / inter-extrapolation analysis (user request) ----
import numpy.linalg as la


def jac_at(f, state):
    sp = torch.tensor([state], dtype=torch.float32, device=dev,
                      requires_grad=True)
    J = torch.autograd.functional.jacobian(
        lambda q: f(q)[0], sp, vectorize=True)[..., 0, :].squeeze(0)
    return J.detach().cpu().numpy()          # (3 out, 4 in)


def path_y(x, g):
    # nominal path y at progress x for anchor g (piecewise linear approx)
    if x <= A_X:
        return g * x / A_X
    if x <= 120:
        return g * (1 - (x - A_X) / (120 - A_X))
    return 0.0


AN = {"L2": cells["L2 (h64)"], "condreg": cells["L2+condreg (h64)"],
      "shellfit": probe_models["shell-fit (h64)"]}
G0 = 10.0
print("\n=== Jacobian at ON-PATH carry state (x=85, y=path, z=0) ===")
st = [85.0, path_y(85.0, G0), 0.0, G0]
print(f"state {[round(v,1) for v in st]}")
for name, f in AN.items():
    J = jac_at(f, st)
    U, sv, Vt = la.svd(J)
    pr = (sv.sum() ** 2) / ((sv ** 2).sum() + 1e-12)
    print(f"{name:9s} sv={np.round(sv, 2)} PR={pr:.2f}")
    print(f"          dz-column (da/dz) = {np.round(J[:, 2], 3)}  "
          f"(ax,ay,az responses to z)")
    print(f"          top-sv input dir v1 = {np.round(Vt[0], 2)} "
          f"(x,y,z,g) -> out u1 = {np.round(U[:, 0], 2)}")

print("\n=== a_z(z) inter/extrapolation at on-path (x=85) and (x=105) ===")
for xq in [85.0, 105.0]:
    print(f"x={xq:.0f}: z:      -10    -5     -2      0     +2     +5    +10")
    for name, f in AN.items():
        vals = []
        for z0 in [-10, -5, -2, 0, 2, 5, 10]:
            sp = torch.tensor([[xq, path_y(xq, G0), z0, G0]],
                              dtype=torch.float32, device=dev)
            with torch.no_grad():
                vals.append(float(f(sp)[0, 2]))
        print(f"  {name:9s} a_z: " + " ".join(f"{v:+.2f}" for v in vals))

print("\n=== on-path a_z(x) profile (z=0): the cross-phase blending test ===")
xs_q = [65, 75, 85, 95, 105, 112, 118]
print("x:        " + "  ".join(f"{x:5d}" for x in xs_q))
for name, f in AN.items():
    vals = []
    for xq in xs_q:
        sp = torch.tensor([[xq, path_y(xq, G0), 0.0, G0]],
                          dtype=torch.float32, device=dev)
        with torch.no_grad():
            vals.append(float(f(sp)[0, 2]))
    print(f"{name:9s} " + "  ".join(f"{v:+.2f}" for v in vals))
print("ANALYSIS-DONE")

# ---- trajectory-conditioned: where do the executed paths separate? ----
print("\n=== executed rollouts, g=+10: state and a_z along the path ===")
print("step:      5     10     15     20     25     30     35")
for name in ["L2", "condreg"]:
    f = AN[name]
    _, tr = rollout(f, G0, 0.0)
    zs, ys, azs = [], [], []
    for i in [5, 10, 15, 20, 25, 30, 35]:
        i = min(i, len(tr) - 1)
        p = tr[i]
        sp = torch.tensor([[p[0], p[1], p[2], G0]], dtype=torch.float32,
                          device=dev)
        with torch.no_grad():
            a = f(sp)[0].cpu().numpy()
        zs.append(p[2])
        ys.append(p[1])
        azs.append(a[2])
    print(f"{name:9s} z:  " + " ".join(f"{v:+5.1f}" for v in zs))
    print(f"{name:9s} y:  " + " ".join(f"{v:+5.1f}" for v in ys))
    print(f"{name:9s} az: " + " ".join(f"{v:+5.2f}" for v in azs))

print("\n=== a_z as a function of y (the coupling): x=70, z=0, g=+10 ===")
print("y:        -5     0     +5    +8   +11    +14")
for name in ["L2", "condreg", "shellfit"]:
    f = AN[name]
    vals = []
    for yq in [-5, 0, 5, 8, 11, 14]:
        sp = torch.tensor([[70.0, yq, 0.0, G0]], dtype=torch.float32,
                          device=dev)
        with torch.no_grad():
            vals.append(float(f(sp)[0, 2]))
    print(f"{name:9s} a_z: " + " ".join(f"{v:+.2f}" for v in vals))
print("TRAJ-DONE")

# ---- seed replication: is the benign side systematic? ----
def train_seeded(obj, h, lam_cond, seed, steps=6000, shell=False):
    torch.manual_seed(seed)
    net = mlp(7, 3, h).to(dev)
    opt = torch.optim.AdamW(net.parameters(), lr=1e-3, weight_decay=1e-5)

    def call(a_in, s):
        return net(torch.cat([a_in, norm_s(s)], 1))
    for _ in range(steps):
        idx = torch.randint(0, len(S), (256,), device=dev)
        s, a = S[idx], A[idx]
        pred = call(torch.zeros_like(a), s)
        loss = ((pred - a) ** 2).mean()
        if shell:
            for _ in range(2):
                v = torch.randn_like(s)
                v = v / (v.norm(dim=1, keepdim=True) + 1e-9) * 2.0
                loss = loss + 0.5 * ((call(torch.zeros_like(a), s + v)
                                      - a) ** 2).mean()
        elif lam_cond > 0:
            d2s = []
            for _ in range(6):
                v = torch.randn_like(s)
                v = v / (v.norm(dim=1, keepdim=True) + 1e-9) * 2.0
                d2s.append(((call(torch.zeros_like(a), s + v)
                             - pred) ** 2).mean(dim=1))
            D = torch.stack(d2s, 1)
            cv2 = D.var(dim=1) / (D.mean(dim=1) ** 2 + 1e-12)
            loss = loss + lam_cond * torch.relu(cv2 - 1.0).mean()
        opt.zero_grad()
        loss.backward()
        opt.step()

    def fwd(s):
        return call(torch.zeros(s.shape[0], 3, device=dev), s)
    return fwd


print("\n=== seed replication: takeover SR (20 eps) per training seed ===")
print("seed:        1     2     3     4     5")
for label, kwargs in [("L2", dict(lam_cond=0.0)),
                      ("condreg", dict(lam_cond=3e-3)),
                      ("shellfit", dict(lam_cond=0.0, shell=True))]:
    row = []
    for sd in [1, 2, 3, 4, 5]:
        f = train_seeded("l2", 64, seed=sd, **kwargs)
        row.append(np.mean([rollout(f, g, 0.0)[0]
                            for g in np.linspace(-15, 15, 20)]))
    print(f"{label:9s} " + "  ".join(f"{v:4.2f}" for v in row))
print("SEEDS-DONE")

# ---- 20-seed bias-distribution test: why does condreg tilt down? ----
def early_bias(f, g=10.0):
    """mean a_z over the early ascent region of the nominal path."""
    vals = []
    for xq in [10, 18, 26, 34, 42]:
        sp = torch.tensor([[xq, path_y(xq, g), 0.0, g]],
                          dtype=torch.float32, device=dev)
        with torch.no_grad():
            vals.append(float(f(sp)[0, 2]))
    return float(np.mean(vals))


print("\n=== 20 seeds x {L2, condreg}: early-region b_z vs takeover SR ===")
rows = {"L2": [], "condreg": []}
for sd in range(1, 21):
    for label, lam in [("L2", 0.0), ("condreg", 3e-3)]:
        f = train_seeded("l2", 64, lam_cond=lam, seed=sd, steps=6000)
        b = early_bias(f)
        srv = np.mean([rollout(f, g, 0.0)[0]
                       for g in np.linspace(-15, 15, 10)])
        rows[label].append((b, srv))
        print(f"SEED {sd:2d} {label:8s} b_z={b:+.3f} SR={srv:.2f}",
              flush=True)
for label, v in rows.items():
    bs = np.array([x[0] for x in v])
    srs = np.array([x[1] for x in v])
    frac_dn = np.mean(bs < 0)
    print(f"SUMMARY {label:8s} b_z mean {bs.mean():+.3f} sd {bs.std():.3f} "
          f"frac(b_z<0) {frac_dn:.2f}  SR mean {srs.mean():.2f}  "
          f"corr(b_z<0, SR) {np.corrcoef(bs < 0, srs)[0, 1]:.2f}")
print("DIST-DONE")

# ---- the same scrutiny for MIP: seed robustness + mechanism signature ----
def train_mip_seeded(seed, h=256, steps=12000):
    torch.manual_seed(seed)
    net = mlp(7, 3, h).to(dev)
    opt = torch.optim.AdamW(net.parameters(), lr=1e-3, weight_decay=1e-5)

    def call(a_in, s):
        return net(torch.cat([a_in, norm_s(s)], 1))
    for _ in range(steps):
        idx = torch.randint(0, len(S), (256,), device=dev)
        s, a = S[idx], A[idx]
        z = torch.randn_like(a)
        y0 = call(z, s)
        y1 = call(y0.detach() / A_SD, s)
        loss = ((y0 - a) ** 2).mean() + ((y1 - a) ** 2).mean()
        opt.zero_grad()
        loss.backward()
        opt.step()

    def fwd(s):
        z = torch.randn(s.shape[0], 3, device=dev)
        y0 = call(z, s)
        return call(y0 / A_SD, s)
    return fwd


print("\n=== MIP h256: 10 seeds — SR, boundedness, knife-edge structure ===")
print("seed   SR(k0) SR(carry10) |a|@+25mm  |a|@-25mm  a_z(z=+5)  b_z(early)")
stats = []
for sd in range(1, 11):
    f = train_mip_seeded(sd)
    sr0 = np.mean([rollout(f, g, 0.0)[0]
                   for g in np.linspace(-15, 15, 10)])
    srk = np.mean([rollout(f, g, 10.0, "carry")[0]
                   for g in np.linspace(-15, 15, 10)])
    with torch.no_grad():
        ap = float(f(torch.tensor([[95.0, 25.0, 0.0, 10.0]],
                    dtype=torch.float32, device=dev))[0].norm())
        am = float(f(torch.tensor([[95.0, -25.0, 0.0, 10.0]],
                    dtype=torch.float32, device=dev))[0].norm())
        az5 = float(f(torch.tensor([[95.0, path_y(95.0, 10.0), 5.0, 10.0]],
                     dtype=torch.float32, device=dev))[0, 2])
    bz = early_bias(f)
    stats.append((sr0, srk, ap, am, az5, bz))
    print(f"{sd:4d}   {sr0:4.2f}   {srk:4.2f}      {ap:6.1f}    {am:6.1f}"
          f"     {az5:+5.2f}     {bz:+.3f}")
arr = np.array(stats)
print(f"MEAN   {arr[:,0].mean():4.2f}   {arr[:,1].mean():4.2f}      "
      f"{arr[:,2].mean():6.1f}    {arr[:,3].mean():6.1f}     "
      f"{arr[:,4].mean():+5.2f}     {arr[:,5].mean():+.3f}")
print(f"reference L2 h256 (seed1): |a|@+/-25mm ~ 100-160; "
      f"L2 20-seed SR(k0) mean 0.36")
print("MIPSEEDS-DONE")

# ---- the z axis specifically: field, excursions, z-kicks ----
print("\n=== a_z(z) at on-path mid-carry (x=95), MIP seeds vs L2 ===")
print("z:          0     +2     +5    +10    +15    +25")
fL2 = cells["L2 (h256)"]
vals = []
for z0 in [0, 2, 5, 10, 15, 25]:
    sp = torch.tensor([[95.0, path_y(95.0, 10.0), z0, 10.0]],
                      dtype=torch.float32, device=dev)
    with torch.no_grad():
        vals.append(float(fL2(sp)[0, 2]))
print("L2 h256   " + "  ".join(f"{v:+5.2f}" for v in vals))
for sd in [1, 2, 3]:
    f = train_mip_seeded(sd)
    vals = []
    for z0 in [0, 2, 5, 10, 15, 25]:
        sp = torch.tensor([[95.0, path_y(95.0, 10.0), z0, 10.0]],
                          dtype=torch.float32, device=dev)
        with torch.no_grad():
            vals.append(float(f(sp)[0, 2]))
    print(f"MIP s{sd}    " + "  ".join(f"{v:+5.2f}" for v in vals))

    # z-excursion during unkicked rollouts + z-kick test
    maxz = []
    for g in np.linspace(-15, 15, 10):
        _, tr = rollout(f, g, 0.0)
        car = tr[tr[:, 0] < 118]
        maxz.append(np.abs(car[:, 2]).max() if len(car) else 0)
    # z-kick: displace z by +10 at mid-carry
    def zkick_roll(fq, g, kick=10.0, nmax=420):
        p = np.zeros(3)
        kicked = False
        while (p[2] > -38 and abs(p[1]) < 80 and -20 < p[0] < 200
               and p[2] < 30 and nmax > 0):
            nmax -= 1
            if not kicked and p[0] >= 90:
                p[2] += kick
                kicked = True
            sp = torch.tensor([[*p, g]], dtype=torch.float32, device=dev)
            with torch.no_grad():
                a = fq(sp)[0].cpu().numpy()
            p = p + a
        return (p[2] <= -38 and np.linalg.norm(p[:2] - S_PT[:2]) < EPS)
    srz = np.mean([zkick_roll(f, g) for g in np.linspace(-15, 15, 10)])
    print(f"          max|z| carry (unkicked): {np.mean(maxz):.2f}mm | "
          f"SR z-kick +10mm: {srz:.2f}")
srzL = np.mean([  # L2 z-kick reference
    (lambda ok: ok)(False) for _ in [0]])
def zkick_roll2(fq, g, kick=10.0, nmax=420):
    p = np.zeros(3)
    kicked = False
    while (p[2] > -38 and abs(p[1]) < 80 and -20 < p[0] < 200
           and p[2] < 30 and nmax > 0):
        nmax -= 1
        if not kicked and p[0] >= 90:
            p[2] += kick
            kicked = True
        sp = torch.tensor([[*p, g]], dtype=torch.float32, device=dev)
        with torch.no_grad():
            a = fq(sp)[0].cpu().numpy()
        p = p + a
    return (p[2] <= -38 and np.linalg.norm(p[:2] - S_PT[:2]) < EPS)
srzL = np.mean([zkick_roll2(fL2, g) for g in np.linspace(-15, 15, 10)])
print(f"L2 h256   SR z-kick +10mm: {srzL:.2f}")
print("ZAXIS-DONE")

# ---- panel-D-style figure for the z-kick experiment ----
def roll_zkick(fq, g, kick, nmax=420):
    p = np.zeros(3)
    kicked = False
    tr = [p.copy()]
    while (p[2] > -38 and abs(p[1]) < 80 and -20 < p[0] < 200
           and p[2] < 30 and len(tr) < nmax):
        if not kicked and kick > 0 and p[0] >= 90:
            p[2] += kick
            kicked = True
        sp = torch.tensor([[*p, g]], dtype=torch.float32, device=dev)
        with torch.no_grad():
            a = fq(sp)[0].cpu().numpy()
        p = p + a
        tr.append(p.copy())
    succ = (p[2] <= -38 and np.linalg.norm(p[:2] - S_PT[:2]) < EPS)
    return succ, np.array(tr)


fM2 = train_mip_seeded(2)
show = [("L2 (h256)", cells["L2 (h256)"]),
        ("MIP (h256, seed 2)", fM2),
        ("shell-fit (h64)", probe_models["shell-fit (h64)"])]
fig, axes = plt.subplots(1, 3, figsize=(17, 4.8), sharey=True)
for ax, (name, f) in zip(axes, show):
    nsucc = {0.0: 0, 10.0: 0}
    ntot = {0.0: 0, 10.0: 0}
    for kick, c in [(0.0, "tab:green"), (10.0, "tab:orange")]:
        for g in np.linspace(-15, 15, 8):
            succ, tr = roll_zkick(f, g, kick)
            ntot[kick] += 1
            nsucc[kick] += succ
            ax.plot(tr[:, 0], tr[:, 2], color=c, lw=1.3, alpha=0.8)
            if not succ:
                ax.plot(tr[-1, 0], tr[-1, 2], "x", color=c, ms=10, mew=2.5)
    ax.plot([120], [-40], "k*", ms=16)
    ax.axhline(0, color="k", lw=0.7, alpha=0.5)
    ax.set_xlabel("x (mm)")
    ax.set_title(f"{name}\nunkicked SR {nsucc[0.0]}/{ntot[0.0]}   "
                 f"z-kick +10mm SR {nsucc[10.0]}/{ntot[10.0]}", fontsize=11)
axes[0].set_ylabel("z (mm)")
fig.suptitle("z-kick test, (x, z) view: +10mm vertical displacement at "
             "mid-carry (orange) vs unkicked (green); X = failure",
             fontsize=12.5, y=1.02)
fig.tight_layout()
fig.savefig("analysis/paper/zkick_view.png", dpi=150, bbox_inches="tight")
print("saved analysis/paper/zkick_view.png")
print("ZVIEW-DONE")

# ---- basin capture range in z: a_z near the descent longitude ----
print("\n=== a_z(x, z): the basin wedge (negative = capture) ===")
print("            z=0    z=+10  z=+20  z=+30")
fM2b = train_mip_seeded(2)
for name, f in [("L2 h256", cells["L2 (h256)"]), ("MIP s2", fM2b)]:
    for xq in [105.0, 112.0, 118.0]:
        row = []
        for z0 in [0.0, 10.0, 20.0, 30.0]:
            sp = torch.tensor([[xq, 0.0, z0, 10.0]], dtype=torch.float32,
                              device=dev)
            with torch.no_grad():
                row.append(float(f(sp)[0, 2]))
        print(f"{name:8s} x={xq:3.0f} " + "  ".join(f"{v:+5.1f}" for v in row))
print("BASIN-DONE")
