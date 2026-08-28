"""Data-level visualization of the three-branch witness, realized as
operator styles: temporally-correlated stretches (visible branches
mid-funnel), all recentering before the tight dock (expert docks), with
the 3-component label mixture at gate-region states deciding the
estimators."""
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

K = 0.5
FWD = 4.0
X_GOAL = 160.0
X_CALM = 128.0
MU_BOT, S_BOT = -0.060, 0.015
MU_MID, S_MID = 0.000, 0.003
P_BOT, P_MID, P_TOP = 0.48, 0.424, 0.096
MU_TOP = -(P_BOT * MU_BOT) / P_TOP
S_TOP = 0.060
ACT = 20.0     # residual unit -> mm equilibrium (ACTION_MM / K_SERVO)


def style_seq(rng, n):
    """Markov style stretches: dense bottom / tight middle / heavy top."""
    out = np.empty(n, dtype=int)
    s = rng.choice(3, p=[P_BOT, P_MID, P_TOP])
    for t in range(n):
        if rng.rand() < 0.12:
            s = rng.choice(3, p=[P_BOT, P_MID, P_TOP])
        out[t] = s
    return out


def demo(rng):
    x, y = 0.0, rng.uniform(-8, 8)
    n = int(X_GOAL / FWD) + 2
    st = style_seq(rng, n)
    tr = [(x, y)]
    res = []
    for t in range(n):
        if x >= X_CALM:
            r = rng.normal(0, 0.004)
        elif st[t] == 0:
            r = rng.normal(MU_BOT, S_BOT)
        elif st[t] == 1:
            r = rng.normal(MU_MID, S_MID)
        else:
            r = MU_TOP + np.clip(rng.standard_t(1.05) * S_TOP, -0.6, 0.6)
        res.append((x, r))
        y = y + (-K * y + r * 10.0)
        hw_here = 5.0 + 9.0 * (1.0 - min(max(x / 130.0, 0.0), 1.0))
        y = float(np.clip(y, -hw_here + 0.3, hw_here - 0.3))
        x += FWD
        tr.append((x, y))
        if x >= X_GOAL:
            break
    return np.array(tr), st, res


rng = np.random.RandomState(4)
fig = plt.figure(figsize=(19.5, 10))
gs = fig.add_gridspec(2, 3, hspace=0.42, wspace=0.3,
                      height_ratios=[1.25, 1])

ax = fig.add_subplot(gs[0, :])
xs = np.linspace(0, 172, 200)
hw = 5.0 + 9.0 * (1.0 - np.clip(xs / 130.0, 0.0, 1.0))
ax.fill_between(xs, -hw, hw, color="#e8edf4")
cols = {0: "#c0392b", 1: "tab:green", 2: "tab:orange"}
docks = []
for _ in range(26):
    tr, st, _ = demo(rng)
    for t in range(len(tr) - 1):
        c = cols[st[t]] if tr[t][0] < X_CALM else "dimgray"
        ax.plot(tr[t:t + 2, 0], tr[t:t + 2, 1], color=c, lw=1.3,
                alpha=0.75)
    docks.append(tr[-1, 1])
ax.axvline(X_CALM, color="k", ls=":", lw=1.2)
ax.text(X_CALM + 1, 11.5, "recenter before docking\n(matches human phase "
        "structure)", fontsize=9)
ax.axvline(X_GOAL, color="k", lw=1)
ax.add_patch(plt.Rectangle((X_GOAL - 0.7, -0.5), 1.4, 1.0,
                           color="tab:green", alpha=0.8))
ax.text(X_GOAL + 2, -0.6, f"dock |y|<0.5 mm\nexpert: "
        f"{np.mean(np.abs(docks) < 0.5):.0%} dock", fontsize=9.5)
ax.set_xlim(-3, 178)
ax.set_ylim(-13.5, 13.5)
ax.set_xlabel("x (mm)")
ax.set_ylabel("y (mm)")
ax.set_title("A   The demonstrations: two robust branches + one noisy "
             "branch, braided by operator-style stretches "
             "(red = dense bottom style 48%, green = tight middle 42%, "
             "orange = heavy noisy style 10%); every demo recenters and "
             "docks", fontsize=11.5, loc="left")

axz = fig.add_subplot(gs[1, 0])
rng2 = np.random.RandomState(11)
for _ in range(40):
    tr, st, _ = demo(rng2)
    m = (tr[:, 0] > 40) & (tr[:, 0] < 125)
    stm = st[:len(tr) - 1]
    for t in range(len(tr) - 1):
        if not (40 < tr[t][0] < 125):
            continue
        axz.plot(tr[t:t + 2, 0], tr[t:t + 2, 1], color=cols[stm[t]],
                 lw=1.1, alpha=0.6)
axz.axhline(MU_BOT * ACT, color="#c0392b", ls="--", lw=1.6)
axz.text(42, MU_BOT * ACT - 0.55, "bottom band: equilibrium "
         f"{MU_BOT*ACT:+.1f} mm (densest)", fontsize=8.5, color="#c0392b")
axz.axhline(0, color="tab:green", ls="--", lw=1.6)
axz.text(42, 0.15, "middle band: the true action", fontsize=8.5,
         color="tab:green")
axz.axhline(MU_TOP * ACT, color="tab:orange", ls="--", lw=1.6)
axz.text(42, MU_TOP * ACT + 0.3, f"noisy band around {MU_TOP*ACT:+.1f} mm "
         "(heavy)", fontsize=8.5, color="tab:orange")
axz.set_ylim(-4, 9)
axz.set_xlabel("x (mm)")
axz.set_ylabel("y (mm)")
axz.set_title("B   Mid-funnel zoom: the three visible bands", fontsize=11,
              loc="left")

ax = fig.add_subplot(gs[1, 1])
res = []
rng3 = np.random.RandomState(21)
for _ in range(600):
    _, st, rr = demo(rng3)
    res += [r for x, r in rr if 90 < x < X_CALM]
res = np.array(res)
ax.hist(res, bins=140, range=(-0.15, 0.45), density=True,
        color="#9aa7b5", alpha=0.8)
ax.set_yscale("log")
for xv, c, lab in ((-0.029, "#c0392b", "MIP $-$0.57 mm: miss"),
                   (0.0, "tab:blue", "L2 popn 0 (n=30 std 0.44 mm)"),
                   (-0.001, "tab:green", "HT learned-$\\nu$: dock")):
    ax.axvline(xv, color=c, ls="--", lw=1.6)
ax.axvspan(-0.025, 0.025, color="tab:green", alpha=0.15)
ax.set_xlabel("lateral action residual at pre-boundary states")
ax.set_title("C   Label mixture at the states that decide the last\n"
             "committed chunk — same estimator math as the design fig",
             fontsize=11, loc="left")
ax.legend(["MIP commits bottom: $-$0.57 mm miss",
           "L2: popn-correct, finite-sample scatter",
           "HT (learned $\\nu\\leq$0.7): locks middle"], fontsize=8,
          loc="upper right")

ax = fig.add_subplot(gs[1, 2])
ax.axis("off")
txt = (
    "Why the branches must be styles, not routes:\n"
    "a demonstrated ROUTE that docks is safe to\n"
    "commit to — route-branches cannot defeat MIP.\n\n"
    "Style stretches give the visible branches\n"
    "mid-funnel; every demo recenters before the\n"
    "gate (oracle docks). The decisive labels are\n"
    "the pre-boundary chunk states (panel C):\n"
    "  MIP:  mass$\\times$Gaussian kernel $\\to$ dense bottom\n"
    "  L2:   population mean 0, heavy-top scatter\n"
    "  HT:   top counterweight cancels the bottom's\n"
    "        pull; tight middle captures the scale\n"
    "        (needs learned $\\nu \\lesssim 0.7$)\n\n"
    "Trained pilot pending; population predictions\n"
    "from the design figure apply unchanged.")
ax.text(0.02, 0.98, txt, fontsize=10.5, va="top", family="serif")
ax.set_title("D   Construction notes", fontsize=11, loc="left")

fig.suptitle("Three-branch witness, data-level view: 2 robust branches + "
             "1 noisy branch (operator styles), expert docks; the noisy "
             "branch is the counterweight that lets HT select the middle "
             "while the dense bottom captures MIP", fontsize=14, y=0.99)
fig.savefig("analysis/paper/toy_threeway_nfl_data.png", dpi=140,
            bbox_inches="tight")
print("saved; expert dock rate printed in panel A")
