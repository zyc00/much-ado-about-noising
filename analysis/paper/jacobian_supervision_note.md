# How each objective supervises the Jacobian
### MSE vs flow/MIP/score-based vs hetero-Gaussian/Student-t — derivations, with each claim tagged [derived] / [measured] / [open]

Notation. Observation $x \in \mathbb{R}^d$ (the normalized window), action chunk
$y \in \mathbb{R}^m$. Expert conditional $p(y|x)$ with mean $\mu(x)$; on clean
data $p(y|x) = \delta(y - \mu(x))$. Network $f_\theta$; encoder Jacobian
$J_x = \partial f/\partial x$. Data measure $\rho(x)$ over a **finite** set of
windows. All "stiffness" statements are curvatures of the population loss at
its optimum, i.e. the restoring force that resists parameter drift; the
relevance of stiffness comes from the SGD stationary-fluctuation relation
(§1.3), which converts (stiffness, gradient-noise) into (retained, eroded).

---

## 1. MSE: the Jacobian is only supervised through finite differences

$$L_{\mathrm{MSE}} = \mathbb{E}_{x,y}\,\|f(x) - y\|^2, \qquad f^* = \mu .$$

### 1.1 Value-space curvature contains no derivative term  [derived]

Perturb $f \to f^* + \varepsilon g$:

$$\Delta L = \varepsilon^2\, \mathbb{E}_{\rho}\|g(x)\|^2 .$$

The curvature in direction $g$ depends **only on the values of $g$ on the data
points**. A perturbation that vanishes on the $N$ data points but has large
gradient between them costs *exactly zero* population loss. With finite data,
the set of such perturbations is an infinite-dimensional flat manifold of the
loss; nothing in the objective selects among Jacobians agreeing with the
values.

### 1.2 The implicit stiffness on $J_x$  [derived]

The Jacobian is pinned only through *pairs* of nearby samples. Take $x$ and a
neighbor $x' = x + \delta v$ ($\|v\|=1$) with labels $\mu(x), \mu(x')$.
Linearize $f$ around $x$: fitting both points forces the directional
derivative $J_x v \approx (\mu(x') - \mu(x))/\delta$. If the network instead
carries $J_x v = D$, the value error at $x'$ is
$\delta\,\|D - \partial_v \mu\| + O(\delta^2)$, so the loss charge is

$$\kappa_{\mathrm{MSE}}(x, v) \;\propto\; \rho_{\mathrm{loc}}(x)\,\delta^2
\quad\text{with target}\quad J_x v \to \partial_v \mu(x),$$

and, aggregating neighbors, the quadratic form on the Jacobian error
$E = J_x - \partial\mu/\partial x$ is

$$\Delta L \;\approx\; \mathbb{E}_{x'\sim\rho_{\mathrm{near}\,x}}
\big\| E\,(x'-x) \big\|^2 \;=\; \operatorname{tr}\!\big(E\,\Sigma_\rho(x)\,E^\top\big),$$

with $\Sigma_\rho(x)$ the **local input covariance of the data** around $x$.
Two consequences:

* (i) The *target* for $J_x$ is $\partial\mu/\partial x$ — whose row space in a
  label-poor region (a hold: $\mu$ locally constant) is $\approx 0$. MSE then
  *demands* a degenerate Jacobian there.
* (ii) The *stiffness* is $\Sigma_\rho$-weighted: directions along which the
  data does not locally vary, or regions visited rarely, get near-zero
  restoring curvature. With label noise $\sigma_n^2$ on the pair, the
  finite-difference estimate of $\partial_v\mu$ has variance
  $\sigma_n^2/\delta^2$: **the gradient signal on $J_x$ has SNR
  $= \delta\,|\partial_v\mu| / \sigma_n$** — small label variation under noise
  makes the Jacobian channel signal-free.

### 1.3 Erosion corollary (why the flat directions matter)  [derived + measured]

SGD as an Ornstein–Uhlenbeck process along a curvature eigendirection with
stiffness $\kappa$, per-step gradient-noise power $s^2$, learning rate $\eta$:

$$\operatorname{Var}_{\infty}(\theta) \;\approx\; \frac{\eta\, s^2}{2\kappa}.$$

For MSE at convergence on data containing an irreducible residual mass
(aliased switch timing / aleatoric noise), $s^2 = O(1)$ **forever** while
$\kappa$ on fine distinctions is the tiny $\Sigma_\rho$-weighted quantity of
§1.2. The stationary excursion exceeds the scale of the distinction ⇒ erased;
the surviving Jacobian concentrates on the few directions with large
label-variation stiffness. This is the measured collapse: settle-metric
PR $13.4 \to 2.0$ with $s_{\max}/s_{\mathrm{med}} \to 61$ on pointing data;
absent on MP data where $s^2 \to 0$ (every residual reducible, PR arrests at
5.1); reproduced with visual controls in the erosion toy.

---

## 2. Flow matching / MIP / score-based: the Jacobian is an explicit target

### 2.1 The conditional targets and their Jacobians  [derived]

Interpolant $y_t = (1-t)\,\epsilon + t\,y$, $\epsilon \sim \mathcal N(0, I_m)$,
$t \sim U(0,1)$. For the deterministic conditional $y = \mu(x)$, given
$(x, y_t, t)$ one can invert $\epsilon = (y_t - t\mu)/(1-t)$, so every standard
regression target is a *closed-form function of $(x, y_t, t)$*:

| parameterization | target | $\partial(\cdot)/\partial y_t$ | $\partial(\cdot)/\partial x$ |
|---|---|---|---|
| velocity ($v = y - \epsilon$) | $v^*(x,y_t,t)=\dfrac{\mu(x) - y_t}{1-t}$ | $-\dfrac{1}{1-t}\,I_m$ | $\dfrac{1}{1-t}\,\dfrac{\partial\mu}{\partial x}$ |
| $\epsilon$-prediction | $\epsilon^* = \dfrac{y_t - t\mu(x)}{1-t}$ | $\dfrac{1}{1-t}\,I_m$ | $-\dfrac{t}{1-t}\,\dfrac{\partial\mu}{\partial x}$ |
| score $\nabla_{y_t}\!\log p_t$ | $s^* = -\dfrac{y_t - t\mu(x)}{(1-t)^2}$ | $-\dfrac{1}{(1-t)^2}\,I_m$ | $\dfrac{t}{(1-t)^2}\,\dfrac{\partial\mu}{\partial x}$ |

Two structural facts, shared by all three parameterizations:

**(A) A full-rank isotropic Jacobian block is explicitly demanded.** The
target's $y_t$-derivative is $\pm I_m/(1-t)^k$ — rank $m$, isotropic, at
*every* $(x, t)$. Unlike MSE (§1.1), this is not an implicit
finite-difference constraint: the loss input itself sweeps a full-dimensional
Gaussian ball of $y_t$ around $t\mu(x)$ (radius $1-t$), so the value
constraints on that ball pin the derivative *within the ball* by direct
integration — the network's input–output map must hold $m$ independent
unit-scale (after the $(1-t)$ factor) sensitivity directions everywhere on the
tube. Toy verification: $(1-t)\,\partial v/\partial y_t = -1.02 \pm 0.25$
against the analytic $-1$  [measured].

**(B) The $x$-Jacobian is supervised with amplification $\tfrac{t}{1-t}$.**
The target's $x$-dependence is $\partial\mu/\partial x$ scaled by
$1/(1-t)$ (velocity) or $t/(1-t)^k$ ($\epsilon$/score). A network error
$\delta\mu(x)$ in the transmitted state-dependence costs, at time $t$,
$\|\delta\mu\|^2/(1-t)^2$ — so integrated over $t \sim U(0,1)$ the stiffness on
*every state-discriminating direction that $\mu$ uses* is

$$\kappa_{\mathrm{FM}}(x,v) \;\propto\; \Big(\textstyle\int_0^1 \frac{dt}{(1-t)^2}\Big)
\,\rho(x)\,\|\partial_v \mu\|^2
\;\;\gg\;\; \kappa_{\mathrm{MSE}}(x,v),$$

the integral cut off at the smallest retained noise scale (or by time
discretization). Note what is and is not amplified: the *stiffness* per unit
of true label variation is amplified; a direction with $\partial_v\mu = 0$
still has target zero. Flow does **not** invent rank that the task lacks — it
prevents the *pruning* of rank the task has, by making the maintenance
curvature large relative to SGD noise (§1.3 with $\kappa$ boosted).

### 2.2 Supervision over a measure: no memorization fixed point  [derived + measured]

MSE's supervision set is the finite list $\{(x_i, \mu_i)\}$: a network that
memorizes it has *zero* loss and zero gradient — from then on §1.3 noise is
unopposed. FM's supervision set is
$\{(x_i, y_t, t): y_t \in \mathbb{R}^m, t \in [0,1]\}$ with fresh draws each
step: a **continuous measure**. No finite parameter vector achieves zero loss
on unseen draws unless the function is correct *as a function* on the tube;
the maintaining gradient therefore never vanishes. This is the exact content
of the frozen-noise control [measured]: freezing $(\epsilon_i, t_i)$ per
sample — identical objective, finite supervision list — restored the
memorize-then-drift dynamics and destroyed the readout, while fresh draws held
the fine structure indefinitely.

### 2.3 What this predicts, and what stays empirical

* Encoder-rank lift with a *dataset-independent* signature — the stiffness in
  2.1(B) is objective-supplied, not data-supplied  [derived → measured: MIP
  PR $\approx 5$–$7$, k90 $2\times$ L2's, anisotropy halved, on pointing, MP-200,
  MP-2k, and human data].
* The arrest level tracks the objective, not the demo count  [measured: MIP
  4.2/4.8/5.2 across 200/2k/20k].
* **Off-support behavior is NOT derivable from the objective**: the tube
  supervises action-space perturbations at on-support $x$; state-space
  extrapolation inherits only the trunk's smoothness bias. The restoring
  annulus field ($dd < 0$), the excursion capping ($5\%$ vs $28\%$
  escalation), and the off-manifold $\|J_x\|$ suppression (edgejac
  $1.4$–$2.6\times$) are measured properties consistent with, but not implied
  by, §2.1  [measured / open for derivation].

---

## 3. Hetero-Gaussian NLL: the same implicit channel, re-priced

$$L_{\mathrm{HG}} = \mathbb{E}\Big[\frac{\|f(x)-y\|^2}{2\sigma(x)^2}
 + m \log \sigma(x)\Big].$$

### 3.1 σ-stationarity  [derived]

At the optimum in $\sigma$ (holding $f$): $\sigma^2(x) = \tfrac1m\mathbb{E}\big[\|y - f(x)\|^2 \,\big|\, x\big]$ —
the local mean-square residual. Substituting back, the gradient on $f$ is
$(y - f)/\sigma^2(x)$: **per-state equalized** — every state contributes $O(1)$
pressure regardless of its residual scale.

### 3.2 Jacobian stiffness: MSE's formula times $1/\sigma^2$  [derived]

HG is a value-space loss like MSE, so §1.2 applies verbatim with the weight
carried through:

$$\kappa_{\mathrm{HG}}(x,v) \;=\; \frac{\kappa_{\mathrm{MSE}}(x,v)}{\sigma^2(x)} .$$

Now put this into the OU relation. In a *fitted, low-noise* region,
$\sigma^2 \to \sigma_{\min}^2$ (the floor): stiffness amplified by up to
$1/\sigma_{\min}^2$. In a *noisy/unfittable* region, $\sigma^2 \to$ the large
irreducible residual: the gradient-noise power that region exports is damped
by $1/\sigma^4$ in variance. **Both the numerator and the denominator of
$\operatorname{Var}_\infty = \eta s^2/2\kappa$ move favorably**, which is why
the arrest happens and holds:

* scripted pointing data: HG arrests at PR 6.5 and holds 200k+ steps while L2
  slides to 2.0  [measured];
* the constructive proof that the *weights alone* carry it: `snorm`
  ($w = 1/(m + \sigma_{\min}^2)$, plain weighted MSE, no likelihood) reaches
  SR 93 on true scripted data  [measured].

### 3.3 Why HG's chart can exceed everyone's on noisy data  [derived candidate + measured]

On heavy-tailed human data the HG gradient $(y-f)/\sigma^2$ is **unbounded in
$y$**: a tail draw at $10\sigma$ exerts $10\times$ force on $f$ at that state.
To move $f$ at $x_i$ without disturbing nearby $x_j$, the network must hold a
high-gain discriminating direction between them (§1.2 with $|\Delta y|$ large):
outlier-chasing is *rank-building pressure*. The measured signature matches —
HG's human chart is the richest of any arm (PR 11.4, k90 21) while its $\mu$
placement is the most corrupted (worst AS=1 cell, volatile bands): the same
unbounded pull that builds the chart drags the mean  [measured; the
chart–corruption link is a candidate, not derived — note the competing channel:
the NLL can also *absorb* an outlier by inflating $\sigma(x)$ locally at only
logarithmic cost, and which channel dominates depends on the σ-head's
smoothness — **open**].

---

## 4. Student-t NLL: same amplification where fitted, bounded influence where not

$$L_{\mathrm{HT}} = \mathbb{E}\Big[\tfrac{\nu+m}{2}
\log\!\Big(1 + \tfrac{\|y-f(x)\|^2}{\nu\,\sigma(x)^2}\Big) + m\log\sigma(x)\Big],
\qquad
\frac{\partial L}{\partial f} \;=\; -\,\underbrace{\frac{(\nu+m)}{\nu\sigma^2 + \|r\|^2}}_{w(r)}\; r,
\quad r = y - f .$$

### 4.1 Two regimes of the effective weight  [derived]

* **Small residuals** ($\|r\|^2 \ll \nu\sigma^2$):
  $w \to (\nu+m)/(\nu\sigma^2)$ — the *same* $1/\sigma^2$ re-pricing as HG.
  Hence on clean scripted data, where all residuals sit at the floor, HT and
  HG are gradient-identical up to a constant — and indeed their metric ladders
  coincide to two decimals (6.46/6.47) and their boundaries both hold
  [measured].
* **Large residuals** ($\|r\|^2 \gg \nu\sigma^2$): $w \propto 1/\|r\|^2$, so
  the per-sample force $w\,r \propto r/\|r\|^2$ — **bounded influence,
  decaying in the outlier size**. The tail mass exerts vanishing pressure on
  $f$ — and therefore, by §1.2, *no rank-building pressure* either.

### 4.2 The consequences, cleanly split by regime  [derived + measured]

* Scripted side: HT = HG (regime 1 everywhere). The tail is inert; σ-repricing
  carries the arrest.  [measured: identical ladders]
* Human side: HT declines to chase the tail (regime 2 on the heavy draws), so
  its chart stays L2-like (PR 5.2 vs HG 11.4) — while its $\mu$ estimate is the
  robust location of the conditional (the t-MLE), sitting on the central mode
  instead of the tail-dragged mean. Its SR dominance (0.90, flat over AS) with
  a *plain* spectrum is the strongest evidence in the record that **on noisy
  data the operative quantity is $\mu$-placement, not the Jacobian spectrum**
  [measured].
* The duality to note for derivation: HT's $w(r)$ is exactly the weight that
  makes the influence function of the location estimate bounded — loss-tail ↔
  noise-tail matching (Student-t NLL = MLE under the measured t(2) demo noise).

---

## 5. Summary table (stiffness on a state-discriminating direction $v$ at $x$)

| objective | stiffness $\kappa(x,v)$ on $J_x$ | supervision set | noise-mass export $s^2$ | derived §
|---|---|---|---|---|
| MSE | $\rho(x)\,\lambda_{\Sigma_\rho}(v)$, target $\partial_v\mu$ | finite list | $O(1)$, undamped | 1.2 |
| flow/MIP/score | $\times \int \frac{dt}{(1-t)^2}$ amplification; plus explicit full-rank $y_t$-block $I_m/(1-t)^k$ | continuous measure (no zero-loss fixed point) | absorbed by anchor/$\epsilon$-channel | 2.1–2.2 |
| hetero-G | $\kappa_{\mathrm{MSE}}/\sigma^2(x)$, $\sigma^2 \to$ local residual | finite list | damped $\times 1/\sigma^4$ | 3.1–3.2 |
| hetero-t | $= $ HG for $\|r\| \ll \sqrt\nu\sigma$; $w \propto 1/\|r\|^2$ beyond | finite list | *bounded* per sample | 4.1 |

**The unifying statement**: every objective that preserves the Jacobian does it
by keeping the maintenance curvature large relative to the gradient noise in
the SGD stationary balance $\operatorname{Var}_\infty = \eta s^2 / 2\kappa$ —
flow by amplifying $\kappa$ through the $1/(1-t)$ schedule *and* refreshing the
supervision measure so the gradient never dies; the hetero-NLLs by re-pricing
$\kappa \to \kappa/\sigma^2$ *and* damping $s^2$ from the noise mass; MSE does
neither, and its Jacobian survives only where the data itself supplies
curvature.

**Open items for derivation** (ranked): (1) the off-support extension —
formalize why tube-supervision at on-support $x$ yields the measured restoring
annulus field and $\|J_x\|$ suppression (candidate: the $\epsilon$-channel
forces $f$ to be a contraction toward the action manifold in $y_t$; coupling
to $x$-extrapolation through the shared trunk's Lipschitz budget); (2) the
HG chart-vs-σ-absorption competition (§3.3); (3) the human-data $10^6$
Frobenius concentration in L2/HT/MIP but not HG (zero-variance input dims)
[measured, unexplained].
