"""Data-level visualization of the midlock witness: dominant tight middle
style, minority bottom style, and rare one-sided RECORDED glitches that
place the label mean on the bottom branch. Expert executes styles +
recenters -> docks; the glitches are log corruption, never executed."""
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

K = 0.5
FWD = 4.0
X_GOAL = 160.0
X_CALM = 128.0
P_MID, S_MID = 0.72, 0.003
P_BOT, MU_BOT, S_BOT = 0.22, -0.170, 0.015
P_TOP, MU_TOP, S_TOP = 0.06, -2.2117, 0.10
ACT = 20.0


def style_seq(rng, n):
    out = np.empty(n, dtype=int)
    s = rng.choice(2, p=[P_MID / (P_MID + P_BOT), P_BOT / (P_MID + P_BOT)])
    for t in range(n):
        if rng.rand() < 0.10:
            s = rng.choice(2, p=[P_MID / (P_MID + P_BOT),
                                 P_BOT / (P_MID + P_BOT)])
        out[t] = s
    return out


def demo(rng):
    x, y = 0.0, rng.uniform(-8, 8)
    n = int(X_GOAL / FWD) + 2
    st = style_seq(rng, n)
    tr = [(x, y)]
    glitches = []
    for t in range(n):
        if x >= X_CALM:
            r = rng.normal(0, 0.004)
        elif st[t] == 0:
            r = rng.normal(0.0, S_MID)
        else:
            r = rng.normal(MU_BOT, S_BOT)
        if x < X_CALM and rng.rand() < P_TOP:
            glitches.append((x, y, MU_TOP + np.clip(
                rng.standard_t(1.05) * S_TOP, -1, 1)))
        y = y + (-K * y + r * 10.0)
        hw = 5.0 + 9.0 * (1.0 - min(max(x / 130.0, 0.0), 1.0))
        y = float(np.clip(y, -hw + 0.3, hw - 0.3))
        x += FWD
        tr.append((x, y))
        if x >= X_GOAL:
            break
    return np.array(tr), st, glitches


rng = np.random.RandomState(6)
fig = plt.figure(figsize=(19.5, 10))
gs = fig.add_gridspec(2, 3, hspace=0.42, wspace=0.3,
                      height_ratios=[1.25, 1])

ax = fig.add_subplot(gs[0, :])
xs = np.linspace(0, 172, 200)
hw = 5.0 + 9.0 * (1.0 - np.clip(xs / 130.0, 0.0, 1.0))
ax.fill_between(xs, -hw, hw, color="#e8edf4")
docks = []
all_gl = []
for _ in range(26):
    tr, st, gl = demo(rng)
    for t in range(len(tr) - 1):
        c = ("tab:green" if st[t] == 0 else "#c0392b") \
            if tr[t][0] < X_CALM else "dimgray"
        ax.plot(tr[t:t + 2, 0], tr[t:t + 2, 1], color=c, lw=1.2,
                alpha=0.75)
    docks.append(tr[-1, 1])
    all_gl += gl
for x0, y0, g in all_gl[:22]:
    ax.annotate("", xy=(x0 + 3, y0 + g * 10 * 0.35), xytext=(x0, y0),
                arrowprops=dict(arrowstyle="->", color="tab:orange",
                                lw=1.3, alpha=0.85))
x0, y0, g = all_gl[3]
ax.annotate("recorded glitch: rest-pose snap\n(in the log only — never "
            "executed)", xy=(x0 + 3, y0 + g * 10 * 0.35),
            xytext=(x0 + 14, y0 - 7.5), fontsize=9, color="tab:orange",
            arrowprops=dict(arrowstyle="->", color="tab:orange", lw=0.9))
ax.axvline(X_CALM, color="k", ls=":", lw=1.2)
ax.text(X_CALM + 1, 11.3, "recenter before docking", fontsize=9)
ax.axvline(X_GOAL, color="k", lw=1)
ax.add_patch(plt.Rectangle((X_GOAL - 0.7, -0.5), 1.4, 1.0,
                           color="tab:green", alpha=0.8))
ax.text(X_GOAL + 2, -0.6, f"dock |y|<0.5 mm\nexpert: "
        f"{np.mean(np.abs(docks) < 0.5):.0%} dock", fontsize=9.5)
ax.set_xlim(-3, 178)
ax.set_ylim(-13.5, 13.5)
ax.set_xlabel("x (mm)")
ax.set_ylabel("y (mm)")
ax.set_title("A   Midlock demonstrations: dominant middle style (green, "
             "72%), minority bottom style (red, 22%, equilibrium "
             "$-$3.4 mm), and rare one-sided recorded glitches (orange "
             "arrows, 6%); every executed trajectory recenters and docks",
             fontsize=11.5, loc="left")

axz = fig.add_subplot(gs[1, 0])
rng2 = np.random.RandomState(13)
for _ in range(40):
    tr, st, _ = demo(rng2)
    stm = st[:len(tr) - 1]
    for t in range(len(tr) - 1):
        if not (40 < tr[t][0] < 125):
            continue
        axz.plot(tr[t:t + 2, 0], tr[t:t + 2, 1],
                 color="tab:green" if stm[t] == 0 else "#c0392b",
                 lw=1.1, alpha=0.6)
axz.axhline(0, color="tab:green", ls="--", lw=1.6)
axz.text(42, 0.25, "middle band: the true action (72%)", fontsize=8.5,
         color="tab:green")
axz.axhline(MU_BOT * ACT, color="#c0392b", ls="--", lw=1.6)
axz.text(42, MU_BOT * ACT - 0.8, "bottom band: equilibrium "
         f"{MU_BOT*ACT:+.1f} mm (22%)", fontsize=8.5, color="#c0392b")
axz.set_ylim(-6, 3)
axz.set_xlabel("x (mm)")
axz.set_ylabel("y (mm)")
axz.set_title("B   Mid-funnel zoom: two executed bands\n(glitches are not "
              "trajectories)", fontsize=11, loc="left")

ax = fig.add_subplot(gs[1, 1])
rng3 = np.random.RandomState(21)
n = 400000
u = rng3.rand(n)
res = np.empty(n)
m = u < P_MID
b = (u >= P_MID) & (u < P_MID + P_BOT)
t = ~m & ~b
res[m] = rng3.normal(0, S_MID, m.sum())
res[b] = rng3.normal(MU_BOT, S_BOT, b.sum())
res[t] = MU_TOP + np.clip(rng3.standard_t(1.05, t.sum()) * S_TOP, -1, 1)
ax.hist(res, bins=200, range=(-2.6, 0.3), density=True, color="#9aa7b5",
        alpha=0.85)
ax.set_yscale("log")
for xv, c, lab in ((-0.1696, "tab:blue", None), (-0.0957, "tab:red", None),
                   (-0.0002, "tab:green", None)):
    ax.axvline(xv, color=c, ls="--", lw=1.7)
ax.axvspan(-0.025, 0.025, color="tab:green", alpha=0.18)
ax.annotate("L2 / HG / MIP step 1 = mean\n= ON the bottom branch "
            "($-$3.4 mm)", xy=(-0.17, 3), xytext=(-1.6, 8), fontsize=8.5,
            color="tab:blue",
            arrowprops=dict(arrowstyle="->", color="tab:blue", lw=1))
ax.annotate("MIP full $-$1.9 mm", xy=(-0.096, 1.2), xytext=(-1.1, 0.9),
            fontsize=8.5, color="tab:red",
            arrowprops=dict(arrowstyle="->", color="tab:red", lw=1))
ax.annotate("HT $\\nu$=2: 0.00 mm (docks)", xy=(-0.0002, 20),
            xytext=(-1.35, 40), fontsize=8.5, color="tab:green",
            arrowprops=dict(arrowstyle="->", color="tab:green", lw=1))
ax.set_xlabel("lateral action residual (normalized units)")
ax.set_title("C   The label mixture and where each estimator lands\n"
             "(one-sided glitch cluster at $-$2.2 drags the mean)",
             fontsize=11, loc="left")

ax = fig.add_subplot(gs[1, 2])
ax.axis("off")
txt = (
    "Design rules (both must hold):\n\n"
    "1. Far mass < 1/($\\nu$+1) = 1/3 for $\\nu$=2:\n"
    "   bottom 22% + glitches 6% = 28% —\n"
    "   the t-scale collapses onto the tight\n"
    "   middle and redescends both far\n"
    "   components $\\Rightarrow$ HT locks the middle.\n\n"
    "2. The label MEAN is MIP's anchor:\n"
    "   the one-sided glitch tail places the\n"
    "   mean exactly on the bottom branch,\n"
    "   so L2/HG/step-1 commit there and\n"
    "   MIP's denoiser lands between\n"
    "   ($-$1.9 mm) — all outside tolerance.\n\n"
    "Trained result (8/8 seeds): only the\n"
    "HT family docks (0.02 mm)." )
ax.text(0.02, 0.98, txt, fontsize=10.5, va="top", family="serif")
ax.set_title("D   Construction notes", fontsize=11, loc="left")

fig.suptitle("Midlock witness, data-level view: two executed styles + "
             "one-sided recorded glitches; the mean sits on the bottom "
             "branch, and only HT $\\nu$=2 recovers the middle",
             fontsize=14, y=0.99)
fig.savefig("analysis/paper/toy_midlock_data.png", dpi=140,
            bbox_inches="tight")
print("saved; expert dock rate in panel A")
