"""Synthetic mathematical checks, NOT robot-policy results.

Minimize S(P,y)=2 E||X-y||/B(P)+log B(P), B(P)=E||X-X'||.
This is twice the negative scaled energy score (scaled CRPS in 1D).
For fixed-shape spherical Student-t and sigma=exp(s), B=sigma*kappa,
so an unbiased stochastic loss/gradient needs samples only in the numerator.
"""
import json
import math

import numpy as np
import torch
from scipy.special import gammaln
from scipy.spatial.distance import cdist


def t_pair_distance_constant(d, nu):
    """Exact E||U-U'|| for independent spherical t_nu(0,I_d), nu>1.

    Derivation: condition on V,V'~chi2_nu; then U-U' is Gaussian with
    variance nu/V+nu/V'. Use independent V+V' and V/(V+V') (Gamma/Beta).
    This is a joint t (one chi-square draw per VECTOR), not independent t axes.
    """
    assert d >= 1 and nu > 1
    log_k = (.5*math.log(nu) + gammaln((d+1)/2)-gammaln(d/2)
             + gammaln(nu-.5)-gammaln(nu-1)
             + 2*gammaln((nu-1)/2)-2*gammaln(nu/2))
    return float(np.exp(log_k))


def scaled_t_energy(mu, log_sigma, target, base_samples, nu):
    """mu,target:[B,d], log_sigma:[B], samples:[K,B,d]. Fixed d and nu.

    Returns one score per example. Omits log(kappa), a parameter-independent
    constant ONLY because nu and d are fixed. No detach on mu or sigma.
    Sampling uses only action-dimensional vectors, not repeated network calls.
    """
    d = mu.shape[-1]
    kappa = t_pair_distance_constant(d, nu)
    z = (target-mu)*torch.exp(-log_sigma[:, None])
    return 2/kappa*torch.linalg.vector_norm(z[None]-base_samples, dim=-1).mean(0)+log_sigma


def main():
    torch.set_num_threads(4)
    rng = np.random.default_rng(20260908)
    checks = {}
    for d, nu in [(1, 3.), (56, 7.), (56, 14.), (56, 224.)]:
        n = 100000
        # Integrate Gaussian directions analytically for a lower-variance MC
        # check of the independently derived pair-distance formula.
        v = rng.chisquare(nu, size=(n, 2))
        radial_mean = math.sqrt(2)*np.exp(gammaln((d+1)/2)-gammaln(d/2))
        draws = radial_mean*np.sqrt(nu/v[:, 0]+nu/v[:, 1])
        exact = t_pair_distance_constant(d, nu)
        mc, se = float(draws.mean()), float(draws.std(ddof=1)/np.sqrt(n))
        assert abs(mc-exact) < 6*se
        checks[f'kappa_d{d}_nu{nu}'] = dict(exact=exact, monte_carlo=mc, standard_error=se)

    d, nu = 56, 14.
    k = 4096
    base = rng.normal(size=(k, 1, d))/np.sqrt(rng.chisquare(nu, size=(k, 1, 1))/nu)
    base = torch.tensor(base, dtype=torch.float64)
    measured = []
    for factor in (1., 5.):
        mu = torch.zeros(1, d, dtype=torch.float64, requires_grad=True)
        target = torch.full((1, d), 2*factor, dtype=torch.float64)
        s = torch.tensor([math.log(factor)], dtype=torch.float64, requires_grad=True)
        loss = scaled_t_energy(mu, s, target, base, nu).sum()
        gm, gs = torch.autograd.grad(loss, (mu, s))
        measured.append(dict(residual_rms=2*factor, sigma=factor,
                             loss=float(loss.detach()), log_sigma_gradient=float(gs),
                             mu_gradient_norm=float(gm.norm())))
    assert abs(measured[0]['log_sigma_gradient']-measured[1]['log_sigma_gradient']) < 1e-12
    assert abs(measured[1]['loss']-measured[0]['loss']-math.log(5)) < 1e-12
    assert abs(measured[0]['mu_gradient_norm']/measured[1]['mu_gradient_norm']-5) < 1e-12
    checks['scale_equivariance_same_samples'] = measured

    # Exact discrete scores: risk gap = energy_distance/B + C/B-1+log(B/C).
    support_q = rng.normal(size=(11, 3))
    weights_q = rng.dirichlet(np.ones(11))
    c = weights_q@cdist(support_q, support_q)@weights_q
    smallest = float('inf')
    for _ in range(100):
        support_p = rng.normal(size=(9, 3))
        weights_p = rng.dirichlet(np.ones(9))
        b = weights_p@cdist(support_p, support_p)@weights_p
        a = weights_p@cdist(support_p, support_q)@weights_q
        gap = 2*a/b+math.log(b)-(2+math.log(c))
        energy_distance = 2*a-b-c
        reconstructed = energy_distance/b+c/b-1+math.log(b/c)
        assert energy_distance >= -1e-12 and gap >= -1e-12
        assert abs(gap-reconstructed) < 1e-12
        smallest = min(smallest, gap)
    checks['exact_discrete_propriety_identity'] = dict(candidates=100, minimum_gap=smallest)
    print(json.dumps(checks, indent=2))
    print('PASS: analytic t constant, scale-equivariant log-scale gradients, proper-score risk-gap identity.')


if __name__ == '__main__':
    main()
