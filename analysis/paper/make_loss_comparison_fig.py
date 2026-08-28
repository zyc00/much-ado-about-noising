"""Three-panel comparison of MSE / Cauchy (t, nu=1) / our Student-t (nu=2):
loss rho(r), influence psi(r) = drho/dr, and implied noise density.
Unit scale sigma = 1; the heteroscedastic head only rescales r.
"""
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

C = {"mse": "#4477AA", "cauchy": "#EE6677", "t2": "#228833"}
r = np.linspace(-6, 6, 1201)


def rho_t(r, nu):
    return (nu + 1) / 2 * np.log1p(r**2 / nu)


def psi_t(r, nu):
    return (nu + 1) * r / (nu + r**2)


fig, axes = plt.subplots(1, 3, figsize=(10.5, 3.0))

ax = axes[0]
ax.plot(r, r**2 / 2, color=C["mse"], lw=2,
        label=r"MSE: $\frac{1}{2}r^2$")
ax.plot(r, rho_t(r, 1), color=C["cauchy"], lw=2,
        label=r"Cauchy ($t_{\nu=1}$): $\log(1+r^2)$")
ax.plot(r, rho_t(r, 2), color=C["t2"], lw=2,
        label=r"Student-$t$ $\nu{=}2$: $\frac{3}{2}\log(1+\frac{r^2}{2})$")
ax.set_ylim(0, 8)
ax.set_xlabel(r"residual $r/\sigma$")
ax.set_ylabel(r"loss $\rho(r)$")
ax.set_title("(a) loss", fontsize=10)
ax.legend(fontsize=7.5, frameon=False)

ax = axes[1]
ax.plot(r, r, color=C["mse"], lw=2, label=r"$\psi=r$ (unbounded)")
ax.plot(r, psi_t(r, 1), color=C["cauchy"], lw=2,
        label=r"$\psi=\frac{2r}{1+r^2}$")
ax.plot(r, psi_t(r, 2), color=C["t2"], lw=2,
        label=r"$\psi=\frac{3r}{2+r^2}$")
ax.set_ylim(-3.2, 3.2)
ax.set_xlabel(r"residual $r/\sigma$")
ax.set_ylabel(r"influence $\psi(r)=\rho'(r)$")
ax.set_title("(b) influence (gradient weight)", fontsize=10)
ax.axhline(0, color="0.8", lw=0.6, zorder=0)
ax.legend(fontsize=7.5, frameon=False, loc="upper left")

ax = axes[2]
dens_g = np.exp(-(r**2) / 2) / np.sqrt(2 * np.pi)
dens_c = 1 / (np.pi * (1 + r**2))
dens_t2 = (1 + r**2 / 2) ** (-1.5) / (2 * np.sqrt(2))
ax.semilogy(r, dens_g, color=C["mse"], lw=2, label="Gaussian")
ax.semilogy(r, dens_c, color=C["cauchy"], lw=2, label="Cauchy")
ax.semilogy(r, dens_t2, color=C["t2"], lw=2, label=r"$t_{\nu=2}$")
ax.set_ylim(1e-8, 1)
ax.set_xlabel(r"$r/\sigma$")
ax.set_ylabel(r"implied noise density $\propto e^{-\rho}$")
ax.set_title("(c) implied noise model (log scale)", fontsize=10)
ax.legend(fontsize=7.5, frameon=False)

for ax in axes:
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.tick_params(labelsize=8)

fig.tight_layout()
fig.savefig("analysis/paper/loss_comparison_fig.png", dpi=220)
fig.savefig("analysis/paper/loss_comparison_fig.pdf")
print("saved analysis/paper/loss_comparison_fig.{png,pdf}")
