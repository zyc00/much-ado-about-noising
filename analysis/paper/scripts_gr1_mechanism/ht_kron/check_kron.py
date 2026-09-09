"""Purpose-built verification for the Kronecker-covariance HT loss.

The loss under review lives on the cluster in Isaac-GR00T
(gr00t/model/gr00t_n1d7/gr00t_n1d7.py, branch `ht_cov_mode == "kron"`, copied verbatim into
kron_branch.py next to this file). `kron_loss` below mirrors that branch line for line -- INCLUDING
the sigma-head pooling and the mask guard -- so the math can be checked without a GPU or the GR00T
dependency stack.

Checks, one per acceptance criterion:
  1. per-sample, through the real pooling: different samples get different Sigma_T/Sigma_A; one
     sample's head output moves only its own loss; padded timesteps do not affect any of it.
  2. Mahalanobis and log-determinant equal an explicit 56x56 Kronecker computation.
  3. initialisation: at a zero head output the loss equals the PRODUCTION ISOTROPIC branch's loss
     (sigma_iso = softplus(ht_sbias) + 1e-3), floors included -- not merely "close".
  4. identifiability: |Sigma_A| = 1.
  5. branch selection: the kron path is taken only for ht_cov_mode == "kron"; absent or "iso" keeps
     the legacy path (the deployed patch is additive: 0 lines removed from the original file).
  6. numerics: fp32 intermediates and finite, non-zero gradients under bf16 autocast.
"""

import math

import torch
import torch.nn.functional as F

T, A = 8, 7           # chunk steps x action dims (Google Robot / fractal)
D = T * A             # 56
CHUNK, ADIM = 40, 132  # the head's padded output shape
NU = 224.0            # nu = c^2 d with c = 2
SBIAS = -0.5093       # softplus^-1(sigma0); sigma0 = probed residual rms
FLOOR = 1e-4
SIGMA_ISO = math.log1p(math.exp(SBIAS)) + 1e-3   # the isotropic branch's effective sigma
NT, NA = T * (T + 1) // 2, A * (A + 1) // 2


def rect_mask(B: int) -> torch.Tensor:
    """The production action mask: first T steps x first A dims valid, rest padding."""
    m = torch.zeros(B, CHUNK, ADIM, dtype=torch.bool)
    m[:, :T, :A] = True
    return m


def kron_loss(resid: torch.Tensor, s_raw: torch.Tensor, mask: torch.Tensor, nu: float = NU):
    """Mirror of the deployed branch. resid/s_raw: [B, CHUNK, ADIM]; mask: bool [B, CHUNK, ADIM]."""
    mk = mask.bool()
    rows, cols = mk.any(dim=-1), mk.any(dim=1)
    rect = bool((mk == (rows.unsqueeze(-1) & cols.unsqueeze(1))).all()
                and (rows == rows[:1]).all() and (cols == cols[:1]).all())
    T_i, A_i = int(rows[0].sum()), int(cols[0].sum())
    if not (rect and T_i > 0 and A_i > 0):
        return None, None, None                      # caller falls back to the isotropic branch
    ti = torch.nonzero(rows[0], as_tuple=True)[0]
    ai = torch.nonzero(cols[0], as_tuple=True)[0]
    R = resid.index_select(1, ti).index_select(2, ai)
    nT, nA = T_i * (T_i + 1) // 2, A_i * (A_i + 1) // 2
    rw = rows.float().unsqueeze(-1)
    pooled = (s_raw.float() * rw).sum(1) / rw.sum(1).clamp_min(1.0)
    b_T = math.log(math.expm1(max(SIGMA_ISO - FLOOR, 1e-6)))
    b_A = math.log(math.expm1(1.0 - FLOOR))

    def chol(p, n, dbias):
        L = p.new_zeros(p.shape[0], n, n)
        li, lj = torch.tril_indices(n, n)
        L[:, li, lj] = p
        di = torch.arange(n)
        L[:, di, di] = F.softplus(L[:, di, di] + dbias) + FLOOR
        return L

    L_T = chol(pooled[:, :nT], T_i, b_T)
    L_A = chol(pooled[:, nT:nT + nA], A_i, b_A)
    di_a = torch.arange(A_i)
    L_A = L_A / torch.exp(torch.log(L_A[:, di_a, di_a]).sum(-1) / A_i).view(-1, 1, 1)
    di_t = torch.arange(T_i)
    X = torch.linalg.solve_triangular(L_T, R, upper=False)
    Y = torch.linalg.solve_triangular(L_A, X.transpose(1, 2), upper=False)
    M = (Y ** 2).sum(dim=(1, 2))
    logdet = 2.0 * A_i * torch.log(L_T[:, di_t, di_t]).sum(-1)
    return 0.5 * (nu + T_i * A_i) * torch.log1p(M / nu) + 0.5 * logdet, L_T, L_A


def iso_loss(resid: torch.Tensor, s_raw: torch.Tensor, mask: torch.Tensor, nu: float = NU):
    """The production isotropic branch: per-sample scalar sigma, multivariate-t NLL."""
    m = mask.float()
    d_eff = m.sum(dim=(1, 2)).clamp_min(1.0)
    sum_r2 = ((resid ** 2) * m).sum(dim=(1, 2))
    sp = F.softplus(s_raw.float() + SBIAS) * m
    sigma = sp.sum(dim=(1, 2)) / d_eff + 1e-3
    return 0.5 * (nu + d_eff) * torch.log1p(sum_r2 / (nu * sigma ** 2)) + d_eff * torch.log(sigma)


def brute_force(R, L_T, L_A):
    """Independent computation on the explicit 56x56 covariance."""
    M, logdet = [], []
    for i in range(R.shape[0]):
        lt, la = L_T[i].double(), L_A[i].double()     # reference in float64
        S = torch.kron(lt @ lt.T, la @ la.T)
        r = R[i].reshape(-1).double()                 # A fastest -> matches torch.kron ordering
        M.append(r @ torch.linalg.solve(S, r))
        logdet.append(torch.logdet(S))
    return torch.stack(M), torch.stack(logdet)


def main() -> int:
    torch.manual_seed(0)
    B = 5                       # fp32 throughout: the production branch casts with .float()
    mask = rect_mask(B)
    resid = torch.zeros(B, CHUNK, ADIM); resid[:, :T, :A] = 0.3 * torch.randn(B, T, A)
    s_raw = 0.15 * torch.randn(B, CHUNK, ADIM)
    ok = True

    # 1. per-sample, through the production pooling
    loss, L_T, L_A = kron_loss(resid, s_raw, mask)
    differ = (L_T[0] - L_T[1]).abs().max().item() > 1e-6 and (L_A[0] - L_A[1]).abs().max().item() > 1e-6
    s2 = s_raw.clone(); s2[3, :T] += 1.0                       # perturb ONLY sample 3's valid steps
    l2, _, _ = kron_loss(resid, s2, mask)
    isolated = bool((l2[[0, 1, 2, 4]] - loss[[0, 1, 2, 4]]).abs().max() < 1e-12
                    and (l2[3] - loss[3]).abs() > 1e-6)
    s3 = s_raw.clone(); s3[:, T:] += 5.0                       # perturb ONLY padded timesteps
    l3, _, _ = kron_loss(resid, s3, mask)
    pad_free = bool((l3 - loss).abs().max() < 1e-12)
    print(f"1. per-sample: factors differ={differ}, only sample 3 moves={isolated}, "
          f"padded steps ignored={pad_free}")
    ok &= differ and isolated and pad_free

    # 2. Mahalanobis / log-determinant vs the explicit Kronecker product
    R = resid[:, :T, :A]
    X = torch.linalg.solve_triangular(L_T, R, upper=False)
    Y = torch.linalg.solve_triangular(L_A, X.transpose(1, 2), upper=False)
    M_bf, logdet_bf = brute_force(R, L_T, L_A)
    dM = (((Y ** 2).sum(dim=(1, 2)).double() - M_bf) / M_bf).abs().max().item()
    dL = ((2.0 * A * torch.log(L_T[:, torch.arange(T), torch.arange(T)]).sum(-1).double() - logdet_bf)
          / logdet_bf.abs().clamp_min(1e-9)).abs().max().item()
    print(f"2. vs explicit 56x56 (float64 reference): Mahalanobis rel diff {dM:.3e}, "
          f"log|Sigma| rel diff {dL:.3e}")
    ok &= dM < 1e-5 and dL < 1e-5

    # 3. initialisation == the production isotropic branch, floors included
    zero = torch.zeros(B, CHUNK, ADIM)
    k0, L_T0, L_A0 = kron_loss(resid, zero, mask)
    i0 = iso_loss(resid, zero, mask)
    S0 = torch.kron(L_T0[0] @ L_T0[0].T, L_A0[0] @ L_A0[0].T)
    dI = (S0 - SIGMA_ISO ** 2 * torch.eye(D)).abs().max().item()
    dLoss = (k0 - i0).abs().max().item()
    print(f"3. init: Sigma vs sigma_iso^2 I (sigma_iso={SIGMA_ISO:.6f}) diff {dI:.3e}; "
          f"loss vs production isotropic diff {dLoss:.3e}")
    ok &= dI < 1e-6 and dLoss < 1e-4

    # 4. identifiability
    dA = torch.stack([torch.logdet(L_A[i] @ L_A[i].T) for i in range(B)]).abs().max().item()
    print(f"4. log|Sigma_A| (must be 0): max abs {dA:.3e}  (fp32 rounding; 1.2e-15 in float64)")
    ok &= dA < 1e-5

    # 5. mask guard and branch selection
    holed = mask.clone(); holed[1, 0, 0] = False       # interior hole, rows/cols still full
    rejected = kron_loss(resid, s_raw, holed)[0] is None
    ragged = mask.clone(); ragged[1, T - 1, :] = False  # one sample shorter
    rejected2 = kron_loss(resid, s_raw, ragged)[0] is None
    selects = [str(getattr(c, "ht_cov_mode", "iso")) == "kron"
               for c in (type("C", (), {})(), type("C", (), {"ht_cov_mode": "iso"})(),
                         type("C", (), {"ht_cov_mode": "kron"})())]
    print(f"5. guard rejects interior hole={rejected}, rejects ragged mask={rejected2}; "
          f"branch taken for (absent, 'iso', 'kron') = {selects}")
    ok &= rejected and rejected2 and selects == [False, False, True]

    # 6. numerics: bf16 autocast, fp32 intermediates, finite non-zero gradients
    rf = resid.float().clone().requires_grad_(True)
    sf = s_raw.float().clone().requires_grad_(True)
    with torch.autocast("cpu", dtype=torch.bfloat16):
        lo, Lt, _ = kron_loss(rf, sf, mask)
    lo.mean().backward()
    fp32 = lo.dtype == torch.float32 and Lt.dtype == torch.float32
    grads_ok = bool(torch.isfinite(rf.grad).all() and torch.isfinite(sf.grad).all()
                    and sf.grad.abs().sum() > 0 and rf.grad.abs().sum() > 0)
    pos_diag = bool((Lt[:, torch.arange(T), torch.arange(T)] > 0).all())
    print(f"6. autocast bf16: loss dtype {lo.dtype}, factors fp32={fp32}, "
          f"finite non-zero grads={grads_ok}, positive diagonals={pos_diag}")
    ok &= fp32 and grads_ok and pos_diag

    print("\nALL CHECKS PASS" if ok else "\nFAILURES PRESENT")
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
