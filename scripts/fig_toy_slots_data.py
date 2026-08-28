"""Data-level view of the three-slot witness: ZERO injection — every
label is an executed aim from a successful demonstration; the variance is
between-operator slot choice."""
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

K, FWD, XG = 0.5, 4.0, 160.0
SLOTS_U = [0.0, -0.30, -0.55]
SLOTS_MM = [20 * s for s in SLOTS_U]
TOL = 1.0
COMP = [(0.60, 0.000, 0.004, "tab:green", "center slot (60%, tight)"),
        (0.28, -0.300, 0.015, "#c0392b", "low slot (28%)"),
        (0.12, -0.550, 0.025, "tab:orange", "lower slot (12%, sloppier)")]
rng = np.random.RandomState(8)


def demo(rng):
    u = rng.rand()
    acc = 0
    for p, mu, s, c, _ in COMP:
        acc += p
        if u < acc:
            aim = rng.normal(mu, s)
            col = c
            break
    x, y = 0.0, rng.uniform(-9, 9)
    tr = [(x, y)]
    while x < XG:
        y = y + (-K * (y - aim * 20))
        x += FWD
        tr.append((x, y))
    return np.array(tr), col, aim


fig = plt.figure(figsize=(19.5, 10))
gs = fig.add_gridspec(2, 3, hspace=0.42, wspace=0.3,
                      height_ratios=[1.25, 1])

ax = fig.add_subplot(gs[0, :])
xs = np.linspace(0, 172, 200)
hw = 4.2 + 9.8 * (1.0 - np.clip(xs / 130.0, 0.0, 1.0))
ax.fill_between(xs, -hw - 9, hw, color="#e8edf4")
ok = 0
for _ in range(30):
    tr, col, aim = demo(rng)
    ax.plot(tr[:, 0], tr[:, 1], color=col, lw=1.2, alpha=0.8)
    ok += any(abs(tr[-1, 1] - sm) < TOL for sm in SLOTS_MM)
for sm, (p, mu, s, c, lab) in zip(SLOTS_MM, COMP):
    ax.add_patch(plt.Rectangle((XG - 0.8, sm - TOL), 1.6, 2 * TOL,
                               color=c, alpha=0.85))
    ax.text(XG + 2.5, sm - 0.5, lab, fontsize=9.5, color=c)
for wz in ((SLOTS_MM[1] + TOL, SLOTS_MM[0] - TOL),
           (SLOTS_MM[2] + TOL, SLOTS_MM[1] - TOL)):
    ax.add_patch(plt.Rectangle((XG - 0.8, wz[0]), 1.6, wz[1] - wz[0],
                               color="k", alpha=0.35))
ax.text(XG + 2.5, -3.9, "wall (failure zone)", fontsize=8.5, color="k")
ax.axvline(XG, color="k", lw=1)
ax.set_xlim(-3, 196)
ax.set_ylim(-16, 14)
ax.set_xlabel("x (mm)")
ax.set_ylabel("y (mm)")
ax.set_title(f"A   The demonstrations: three valid slots, three operator "
             f"aim styles — every action executed, every demo successful "
             f"(this sample: {ok}/30; population 0.995). No noise channel "
             "exists anywhere in the generator.", fontsize=11.5,
             loc="left")

ax = fig.add_subplot(gs[1, 0])
n = 400000
u = rng.rand(n)
res = np.empty(n)
acc = 0
for p, mu, s, c, _ in COMP:
    m = (u >= acc) & (u < acc + p)
    res[m] = rng.normal(mu, s, m.sum())
    acc += p
ax.hist(res, bins=240, range=(-0.75, 0.12), density=True,
        color="#9aa7b5", alpha=0.85)
ax.set_yscale("log")
for xv, c, lw in ((-0.1497, "tab:blue", 1.8), (-0.0948, "tab:red", 1.8),
                  (-0.061, "#7fbf7f", 1.6), (-0.000, "tab:green", 1.8)):
    ax.axvline(xv, color=c, ls="--", lw=lw)
for sm in SLOTS_U:
    ax.axvspan(sm - 0.05, sm + 0.05, color="gray", alpha=0.12)
ax.annotate("L2/HG = mean $-$3.0 mm\n(wall)", xy=(-0.15, 2.5),
            xytext=(-0.62, 6), fontsize=8.5, color="tab:blue",
            arrowprops=dict(arrowstyle="->", color="tab:blue", lw=1))
ax.annotate("MIP $-$1.9 mm (wall:\nsoft average of two slots)",
            xy=(-0.095, 1.0), xytext=(-0.6, 0.35), fontsize=8.5,
            color="tab:red",
            arrowprops=dict(arrowstyle="->", color="tab:red", lw=1))
ax.annotate("HT $\\nu$=2: $-$1.2 mm (wall edge)", xy=(-0.061, 6),
            xytext=(-0.6, 30), fontsize=8.5, color="#7fbf7f",
            arrowprops=dict(arrowstyle="->", color="#7fbf7f", lw=1))
ax.annotate("HT $\\nu$=0.5: 0.00 mm\n(center slot)", xy=(0.0, 30),
            xytext=(0.02, 3.5), fontsize=8.5, color="tab:green",
            arrowprops=dict(arrowstyle="->", color="tab:green", lw=1))
ax.set_xlabel("aim label (normalized units); gray bands = slots")
ax.set_title("B   The label mixture at pre-fork states and\nwhere each "
             "estimator lands (population)", fontsize=11, loc="left")

ax = fig.add_subplot(gs[1, 1])
nus = np.linspace(0.3, 3.0, 22)
locs = []
for nu in nus:
    mu, s = np.median(res), 0.02
    for _ in range(300):
        r = (res - mu) / s
        w = (nu + 1) / (nu + r ** 2)
        mu = (w * res).sum() / w.sum()
        s = np.sqrt(max((w * (res - mu) ** 2).sum() / len(res), 1e-12))
    locs.append(mu * 20)
ax.plot(nus, locs, "-o", color="tab:green", ms=4)
for sm, (p, mu0, s0, c, _) in zip(SLOTS_MM, COMP):
    ax.axhspan(sm - TOL, sm + TOL, color=c, alpha=0.15)
ax.axvline(2.0, color="gray", ls=":", lw=1.2)
ax.text(2.05, -2.6, "fixed $\\nu$=2", fontsize=8.5, color="gray")
ax.set_xlabel("Student-t $\\nu$")
ax.set_ylabel("HT dock position (mm)")
ax.set_title("C   HT location vs $\\nu$: the center slot is\nselected for "
             "$\\nu \\lesssim 1$ (far mass 0.40)", fontsize=11, loc="left")
ax.grid(alpha=0.3)

ax = fig.add_subplot(gs[1, 2])
ax.axis("off")
txt = (
    "Why this is injection-free:\n"
    "the only variance is WHICH slot each\n"
    "operator aims at (and small per-operator\n"
    "aim spread, all inside slot tolerance).\n"
    "Every label was executed; every episode\n"
    "succeeded.\n\n"
    "The fork's failure zones (walls) make\n"
    "convex combinations of valid aims invalid:\n"
    "  L2/HG: global mean $\\to$ wall\n"
    "  MIP:   posterior = SOFT average of the\n"
    "         two nearest slots (0.68/0.32)\n"
    "         $\\to$ wall — commitment fails by\n"
    "         averaging, not by mode choice\n"
    "  HT:    redescending location locks the\n"
    "         tight majority slot; needs\n"
    "         $\\nu \\lesssim 1$ (learned-$\\nu$ finds it)\n\n"
    "Trained 8-seed run in progress.")
ax.text(0.02, 0.98, txt, fontsize=10.5, va="top", family="serif")
ax.set_title("D   Construction notes", fontsize=11, loc="left")

fig.suptitle("Three-slot witness, data-level view: zero injected noise — "
             "operator aim choice is the only variance; only the "
             "heavy-hypothesis location (HT $\\nu\\approx$0.5) selects a "
             "valid slot", fontsize=14, y=0.99)
fig.savefig("analysis/paper/toy_slots_data.png", dpi=140,
            bbox_inches="tight")
print("saved")
