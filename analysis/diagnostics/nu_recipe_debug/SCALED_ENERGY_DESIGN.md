# Sampling-based proper loss with scale-equivariant log-scale gradients

Design/prototype only. No change to the active nu14 Gaussian-auxiliary run.
This uses an established scaled kernel/energy score, not a claimed new scoring
rule. Reference: Bolin & Wallin, Local scale invariance and robustness of proper
scoring rules, https://arxiv.org/abs/1912.05642 . The factor-two negative-score
convention below is for minimization. Euclidean energy distance supplies strict
propriety in multiple dimensions.

## General score and propriety

Let A(P,y)=E_P ||X-y||, B(P)=E_{P,P} ||X-X'|| with independent draws.
Use S(P,y)=2 A(P,y)/B(P)+log B(P). Domain: nondegenerate distributions on R^d
with finite first moments, so 0<B(P)<infinity. This excludes point masses.

For true Q, let C=B(Q), A=E_{P,Q} ||X-Y||, and D=2A-B-C (energy distance).
Then exactly:

    E_Q S(P,Y) - E_Q S(Q,Y)
      = D/B + C/B - 1 + log(B/C) >= 0.

Energy distance is nonnegative and vanishes only when P=Q; the second term
is u-1-log u >=0 at u=C/B. This proves strict propriety, not merely an empirical
Gaussian calibration property. Conditionally, apply this statement to P(.|o)
and Q(.|o) at each o and average over observations. Neural function sharing and
misspecification still limit how closely the conditional optimum can be attained.

## Spherical fixed-nu Student-t specialization

Sample X=mu(o)+exp(s(o))*U, U~t_nu(0,I_d), fixed nu>1. Here sigma=exp(s)
is the Student-t scale, not its standard deviation. One chi-square draw is
shared across all d coordinates of EACH sampled U.

Let kappa=E||U-U'||. Then B=exp(s)*kappa, and, omitting log(kappa),

    L(mu,s;y) = 2/kappa E_U || (y-mu)*exp(-s) - U || + s.

Neither mu nor s is detached. For fixed nu,d, kappa is a constant and the
expectation can be estimated with independent base samples; its pathwise
gradient is unbiased under the usual differentiation/integration conditions.
Sampling is action-vector noise only: no repeated policy network evaluations.

An exact constant is available (derived here via the Gamma/Beta decomposition):

    kappa = sqrt(nu) Gamma((d+1)/2)/Gamma(d/2)
            * Gamma(nu-1/2)/Gamma(nu-1)
            * [Gamma((nu-1)/2)/Gamma(nu/2)]^2.

Conditioning on V,V'~chi2_nu gives U-U'~N(0,nu*(1/V+1/V')I).
T=V+V'~Gamma(nu,scale=2) and R=V/T~Beta(nu/2,nu/2) are independent.
Since 1/V+1/V'=1/[T R(1-R)], their inverse-half moments give the expression.
For nu=14,d=56, kappa=11.2531740612. The Gaussian limit is
2 Gamma((d+1)/2)/Gamma(d/2).

Do NOT replace B by a tiny Monte Carlo batch and claim its ratio/log score or
gradient is unbiased. If nu or covariance shape is learned, kappa is no longer
a constant: its dependence, the log(kappa) term, and sample-distribution
derivatives must all be included. The prototype intentionally fixes nu and d.

## Scale-gradient property

Let z=(y-mu)*exp(-s). Then

    dL/ds = 1 - 2/kappa E[ z . (z-U) / ||z-U|| ].

It depends on relative residual z, not absolute sigma. Rescaling y,mu,sigma
by the same positive c shifts s by log(c), adds log(c) to the score, and
leaves dL/ds unchanged. With common base draws, this holds for each finite
Monte Carlo estimate too. The gradient w.r.t. sigma itself remains (dL/ds)/sigma;
parameterization, not just the score, is essential to the requested balance.

Synthetic autograd check (nu14,d56,K4096, common draws):

| Residual RMS | sigma | dL/ds |
|---|---|---|
| 2 | 1 | -1.3523741732722536 |
| 10 | 5 | -1.3523741732722536 |

This is a synthetic invariance test, NOT a robot data/performance measurement.
The realized Monte Carlo value is seed-dependent; equality is the property.

Mean gradient norm is bounded by 2/(kappa*sigma) at fixed sigma. Scale gradient
does not have the Student-t NLL's nu/d negative-side saturation: at large ||z||
it grows approximately linearly in ||z||. It is NOT a redescending Student-t
NLL, and the score is not bounded-robust in the terminology of Bolin & Wallin.
Therefore switching to this score changes the robust-fitting behavior even
when using the same Student-t predictive family.

## Practical interpretation

- A simpler control already meets absolute-scale balance: native proper NLL
  with scalar log-sigma parameterization. It retains small-nu scale-gradient
  attenuation, which is distinct from absolute-scale imbalance.
- For an action head, predict a scalar s (e.g. masked mean of raw head outputs)
  then exponentiate, rather than averaging softplus/exp values. Initial bias
  must be recalibrated if switching from the existing softplus parameterization.
- Hard clamps/floors break exact equivariance near their boundaries. Compute
  exp/log and score reductions in FP32 and monitor extremes.
- This balances log-scale output gradients under uniform unit rescaling, not
  all conditional states, individual coordinate units, upstream Jacobians,
  or the relative importance of mu and sigma branches.
- Proper scoring guarantees truthful distributions in expectation when they
  are available, not the best robot control success or finite-sample stability.

Prototype/checks: `scripts/test_scaled_energy_score.py`. All analytic-constant
Monte Carlo checks, exact discrete risk-gap identities, and autograd scaling
checks passed. No new cluster jobs submitted.
