"""Schematic: why the anchor structure (MIP) keeps rollouts near the support.

Same 2D construction as the rank schematic: support = sine curve, demonstrated
action = unit tangent, both policies identical on-support. The panels differ
in the OFF-SUPPORT MAGNITUDE behavior of the action field:
  A  L2: the whole action passes through the state-conditioned pathway;
     off-support extrapolation grows in magnitude and rotates (misdirection
     grows with distance).  a(x) = (1 + 2.2 d) * R(2.5 d * w(s)) t(s)
  B  MIP: action = anchor (state-independent at fixed progress, equals the
     demonstrated action) + small bounded correction. Off-support the
     magnitude stays at the demonstrated level.  a(x) ~= t(s)
  C  measured closed-loop values (twofactor N=200, mp200 checkpoints).
"""
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

def curve(s):
    return np.stack([s, 0.35 * np.sin(2 * s)], -1)

def tangent(s):
    t = np.stack([np.ones_like(s), 0.7 * np.cos(2 * s)], -1)
    return t / np.linalg.norm(t, axis=-1, keepdims=True)

def normal(s):
    t = tangent(s)
    return np.stack([-t[..., 1], t[..., 0]], -1)

S = np.linspace(0, 3.2, 400)
C = curve(S)

def project(x):
    d2 = ((C - x) ** 2).sum(-1)
    i = int(np.argmin(d2))
    s = S[i]
    n = normal(np.array([s]))[0]
    dn = float((x - C[i]) @ n)
    return s, dn, n

def rot(a, th):
    c, s_ = np.cos(th), np.sin(th)
    return np.array([c * a[0] - s_ * a[1], s_ * a[0] + c * a[1]])

def field_l2(x):
    s, dn, n = project(x)
    t = tangent(np.array([s]))[0]
    mag = 1.0 + 2.2 * abs(dn)                  # magnitude grows off-support
    th = 2.5 * dn * np.sin(3 * s + 1.0)        # misdirection grows, sign varies
    return mag * rot(t, th)

def field_mip(x):
    s, dn, n = project(x)
    return tangent(np.array([s]))[0]           # anchor: demonstrated magnitude

def rollout(field, x0, eta=0.05, nsteps=140, dmax=0.30):
    # terminate at deep band exit: in the real task the episode fails during
    # the excursion (SR|cross4 = 15-27%); re-entry in 2D is a low-dimension
    # artifact and is not rendered.
    xs = [np.array(x0, float)]
    rng = np.random.default_rng(5)
    failed = False
    for _ in range(nsteps):
        x = xs[-1]
        if x[0] > 3.05:
            break
        _, dn, _ = project(x)
        if abs(dn) > dmax:
            failed = True
            break
        xs.append(x + eta * field(x) + rng.normal(0, 0.004, 2))
    return np.array(xs), failed

def field_rank(x, k=2.2):
    s, dn, n = project(x)
    a = tangent(np.array([s]))[0] - k * dn * n
    return a / max(np.linalg.norm(a), 1e-9)


fig, axes = plt.subplots(1, 3, figsize=(15.5, 4.7))

for ax, field, title, note in [
    (axes[0], field_l2,
     "A  L2 regression: whole action through the state-conditioned\n"
     "pathway; off-support magnitude grows, direction rotates",
     "deviation is amplified: deep exit\nis reached quickly\n"
     "(measured: deep-exit rate 0.61/episode,\nSR|cross4 = 15$-$27%)"),
    (axes[1], field_rank,
     "B  rank constraint: response along the deviation\n"
     "direction exists and points to the support",
     "small deviations are corrected\nimmediately\n"
     "(measured: holds 3$-$6mm at +8 steps;\nband return fraction 0.64)"),
    (axes[2], field_mip,
     "C  MIP anchor: bounded response at the\n"
     "demonstrated magnitude",
     "no amplification: deviation grows only\nat the noise rate\n"
     "(measured: outward tail +0.088\nvs +0.16$-$0.23; deep-exit 0.22/ep)"),
]:
    for w, alpha in [(0.24, 0.10), (0.12, 0.12)]:
        ax.fill_between(S, C[:, 1] - w, C[:, 1] + w,
                        color="tab:orange", alpha=alpha, lw=0)
    ax.plot(C[:, 0], C[:, 1], "k-", lw=2.2, label="demonstration support")
    gx, gy = np.meshgrid(np.linspace(0.1, 3.1, 15),
                         np.linspace(-0.75, 0.95, 10))
    U = np.zeros_like(gx)
    V = np.zeros_like(gy)
    for i in range(gx.shape[0]):
        for j in range(gx.shape[1]):
            a = field(np.array([gx[i, j], gy[i, j]]))
            U[i, j], V[i, j] = a
    ax.quiver(gx, gy, U, V, color="tab:blue", alpha=0.45,
              scale=30, width=0.0035)
    x0 = [0.3, 0.40]
    tr, failed = rollout(field, x0)
    ax.plot(tr[:, 0], tr[:, 1], "r-", lw=2.0, label="closed-loop rollout")
    ax.plot(*x0, "r*", ms=14, label="start (off-support)")
    if failed:
        ax.plot(tr[-1, 0], tr[-1, 1], "rx", ms=16, mew=3.5,
                label="deep exit = episode failure")
    ax.set_title(title, fontsize=10.5)
    ax.set_xlim(0, 3.2)
    ax.set_ylim(-0.85, 1.05)
    ax.set_xticks([])
    ax.set_yticks([])
    ax.legend(loc="lower right", fontsize=8, framealpha=0.9)
    ax.annotate(note, xy=(0.12, -0.78), fontsize=8.6, color="darkred")

fig.suptitle("Off-support response and closed-loop deviation: amplifying "
             "(L2) vs corrective (rank constraint) vs bounded (MIP anchor)",
             fontsize=12, y=1.00)
fig.tight_layout()
out = "analysis/paper/anchor_mechanism_schematic.png"
fig.savefig(out, dpi=160, bbox_inches="tight")
print("saved", out)
