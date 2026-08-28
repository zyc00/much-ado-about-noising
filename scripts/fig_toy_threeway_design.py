"""Design + population-prediction figure for the three-branch witness
(user construction): dominant bottom branch (MIP commits), ultra-tight
middle = true action (learned-nu HT locks), heavy top counterweight
(zero mean; drags L2's finite-sample estimate). No trained models yet."""
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

P_BOT, MU_BOT, S_BOT = 0.48, -0.060, 0.015
P_MID, MU_MID, S_MID = 0.424, 0.000, 0.003
P_TOP, S_TOP = 0.096, 0.060
MU_TOP = -(P_BOT * MU_BOT) / P_TOP
SA = 0.10
TOL_RES = 0.025          # 0.5 mm dock tolerance in residual units (x20 mm)
rng = np.random.RandomState(0)


def draw(n):
    u = rng.rand(n)
    out = np.empty(n)
    b = u < P_BOT
    m = (u >= P_BOT) & (u < P_BOT + P_MID)
    t = ~b & ~m
    out[b] = rng.normal(MU_BOT, S_BOT, b.sum())
    out[m] = rng.normal(MU_MID, S_MID, m.sum())
    out[t] = MU_TOP + np.clip(rng.standard_t(1.05, t.sum()) * S_TOP,
                              -0.6, 0.6)
    return out


def ht_loc(big, nu):
    mu, s = np.median(big), 0.02
    for _ in range(400):
        r = (big - mu) / s
        w = (nu + 1) / (nu + r ** 2)
        mu = (w * big).sum() / w.sum()
        s = np.sqrt(max((w * (big - mu) ** 2).sum() / len(big), 1e-12))
    return mu, s


big = draw(400000)
comp = np.array([MU_BOT, MU_MID, MU_TOP])
pr = np.array([P_BOT, P_MID, P_TOP])
w = pr * np.exp(-comp ** 2 / (2 * SA ** 2))
mip_post = (w * comp).sum() / w.sum()
loc2, s2 = ht_loc(big, 2.0)
loc05, s05 = ht_loc(big, 0.5)

fig = plt.figure(figsize=(19.5, 10.5))
gs = fig.add_gridspec(2, 3, hspace=0.45, wspace=0.3)

# A task + executed nuisance arrows colored by branch
ax = fig.add_subplot(gs[0, 0])
xs = np.linspace(0, 165, 200)
hw = 5.0 + 9.0 * (1.0 - np.clip(xs / 130.0, 0.0, 1.0))
ax.fill_between(xs, -hw, hw, color="#dbe4ee", alpha=0.8)
y = -8.0
xp = 0.0
K = 0.5
cols = {0: "#c0392b", 1: "tab:green", 2: "tab:orange"}
rngA = np.random.RandomState(5)
for k in range(28):
    u = rngA.rand()
    if u < P_BOT:
        r, c = rngA.normal(MU_BOT, S_BOT), cols[0]
    elif u < P_BOT + P_MID:
        r, c = rngA.normal(MU_MID, S_MID), cols[1]
    else:
        r = MU_TOP + np.clip(rngA.standard_t(1.05) * S_TOP, -0.6, 0.6)
        c = cols[2]
    a = -K * y + r * 20
    ax.annotate("", xy=(xp + 4, y + a * 0.8), xytext=(xp, y),
                arrowprops=dict(arrowstyle="->", color=c, lw=1.4,
                                alpha=0.9))
    y = y + (-K * y + r * 20 * 0.3)
    xp += 5.5
    if xp > 150:
        break
ax.axvline(160, color="k", lw=1)
ax.add_patch(plt.Rectangle((159.2, -0.5), 1.6, 1.0, color="tab:green",
                           alpha=0.7))
ax.set_xlim(-3, 172)
ax.set_ylim(-14, 14)
ax.set_xlabel("x (mm)")
ax.set_ylabel("y (mm)")
ax.set_title("A   Executed 3-branch nuisance\n(red bottom 48% / green "
             "middle 42% / orange top 10%)", fontsize=10.5, loc="left")

# B the label distribution + every estimator's target
ax = fig.add_subplot(gs[0, 1:])
xs = np.linspace(-0.15, 0.45, 3000)


def gpdf(x, mu, s):
    return np.exp(-(x - mu) ** 2 / (2 * s * s)) / (s * np.sqrt(2 * np.pi))


dens = (P_BOT * gpdf(xs, MU_BOT, S_BOT) + P_MID * gpdf(xs, MU_MID, S_MID)
        + P_TOP * 1 / (np.pi * S_TOP * (1 + ((xs - MU_TOP) / S_TOP) ** 2)))
ax.fill_between(xs, dens, color="#f0f0f0")
ax.plot(xs, P_BOT * gpdf(xs, MU_BOT, S_BOT), color="#c0392b", lw=2,
        label="bottom branch (48%, wide)")
ax.plot(xs, P_MID * gpdf(xs, MU_MID, S_MID), color="tab:green", lw=2,
        label="middle = true action (42%, ultra-tight)")
ax.plot(xs, P_TOP / (np.pi * S_TOP * (1 + ((xs - MU_TOP) / S_TOP) ** 2)),
        color="tab:orange", lw=2, label="top counterweight (10%, heavy)")
ax.axvspan(-TOL_RES, TOL_RES, color="tab:green", alpha=0.12)
ax.set_yscale("log")
ax.set_ylim(0.05, 80)
marks = [("L2 = population mean = 0.000\n(but n=30 scatter std 0.022)",
          0.0, "tab:blue", 38),
         (f"MIP posterior {mip_post:+.3f}\n(dock {mip_post*20:+.2f} mm: "
          "FAILS)", mip_post, "#c0392b", 34),
         (f"HT $\\nu$=2: {loc2:+.3f} (FAILS)", loc2, "gray", 18),
         (f"HT $\\nu$=0.5: {loc05:+.3f}\n(dock {loc05*20:+.2f} mm: DOCKS)",
          loc05, "tab:green", 8)]
for txt, xv, c, yv in marks:
    ax.axvline(xv, color=c, lw=1.6, ls="--", alpha=0.85)
    ax.annotate(txt, xy=(xv, yv), xytext=(xv + 0.05, yv), fontsize=9,
                color=c, arrowprops=dict(arrowstyle="->", color=c, lw=1))
ax.legend(fontsize=9, loc="upper right")
ax.set_xlabel("lateral action residual (normalized units); shaded = dock "
              "tolerance $\\pm$0.025")
ax.set_title("B   The label distribution at every state, and where each "
             "estimator lands (population)", fontsize=10.5, loc="left")

# C kernel-shape separation
ax = fig.add_subplot(gs[1, 0])
xs = np.linspace(-0.12, 0.12, 800)
ax.plot(xs, np.exp(-xs ** 2 / (2 * SA ** 2)), color="#c0392b", lw=2,
        label=f"MIP anchor kernel (fixed $\\sigma_a$={SA})")
nu = 0.5
r = xs / s05
ax.plot(xs, ((nu + 1) * np.abs(r) / (nu + r ** 2)) /
        ((nu + 1) / (2 * np.sqrt(nu))), color="tab:green", lw=2,
        label=f"HT influence (fitted $\\sigma$={s05:.3f}, redescending)")
for xv, lab in ((MU_BOT, "bottom"), (0.0, "middle")):
    ax.axvline(xv, color="k", lw=0.8, ls=":")
    ax.text(xv + 0.003, 1.03, lab, fontsize=8.5)
ax.set_ylim(0, 1.12)
ax.set_xlabel("residual")
ax.set_ylabel("relative weight")
ax.legend(fontsize=8.5, loc="center right")
ax.set_title("C   Why they disagree: at the bottom branch the\nMIP kernel "
             "is still 0.84; HT's influence has redescended", fontsize=10.5,
             loc="left")

# D HT location vs nu
ax = fig.add_subplot(gs[1, 1])
nus = np.linspace(0.3, 3.0, 24)
locs = [ht_loc(big, nu)[0] for nu in nus]
ax.plot(nus, np.array(locs) * 20, "-o", color="tab:green", ms=4)
ax.axhspan(-0.5, 0.5, color="tab:green", alpha=0.12)
ax.axhline(mip_post * 20, color="#c0392b", ls="--", lw=1.4,
           label=f"MIP posterior ({mip_post*20:+.2f} mm)")
ax.axvline(2.0, color="gray", ls=":", lw=1.2)
ax.text(2.03, -0.1, "fixed $\\nu$=2", fontsize=8.5, color="gray")
ax.set_xlabel("Student-t $\\nu$")
ax.set_ylabel("HT dock offset (mm)")
ax.legend(fontsize=8.5)
ax.set_title("D   HT's location vs $\\nu$: the middle is selected only "
             "for\n$\\nu \\lesssim 0.7$ — the learned-$\\nu$ arm is the "
             "witness", fontsize=10.5, loc="left")
ax.grid(alpha=0.3)

# E finite-sample L2
ax = fig.add_subplot(gs[1, 2])
means = np.array([draw(30).mean() for _ in range(4000)]) * 20
ax.hist(means, bins=90, color="tab:blue", alpha=0.75)
ax.axvspan(-0.5, 0.5, color="tab:green", alpha=0.15)
frac = np.mean(np.abs(means) > 0.5)
ax.set_xlim(-3, 3)
ax.set_xlabel("L2 dock offset from n=30 labels/state (mm)")
ax.set_title(f"E   L2's finite-sample scatter: {frac:.0%} of states\n"
             "land outside the dock tolerance (heavy top)", fontsize=10.5,
             loc="left")

fig.suptitle("Three-branch witness (design + population prediction, no "
             "trained models yet): dominant bottom commits MIP; the heavy "
             "top drags L2's finite-sample mean; only the matched "
             "(learned-$\\nu$) HT locks the true middle",
             fontsize=14, y=0.99)
fig.savefig("analysis/paper/toy_threeway_nfl_design.png", dpi=140,
            bbox_inches="tight")
print("saved")
