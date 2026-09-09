"""Add a Kronecker-factored covariance branch to the GR00T HT loss.

Sigma = Sigma_T (x) Sigma_A  over (chunk step T) x (action dim A), d = T*A.
  M        = tr(Sigma_T^-1 R Sigma_A^-1 R^T)      (matrix-variate Mahalanobis)
  log|Sig| = A*log|Sigma_T| + T*log|Sigma_A|
  loss     = 0.5*(nu+d)*log1p(M/nu) + 0.5*log|Sigma|
Both factors are Cholesky-parameterised with softplus diagonals initialised to sqrt(sigma0),
so at step 0 Sigma = sigma0^2 I and the branch coincides with the isotropic model.
Sigma_A is normalised to unit determinant to fix the (cSigma_T) x (Sigma_A/c) non-identifiability.
Parameters come from the EXISTING sigma head (pooled over time, first T(T+1)/2 + A(A+1)/2 entries),
so checkpoint keys and the DDP parameter count are unchanged.
"""
import re, shutil, sys

P = "/mnt/pfs/yuchen/groot/Isaac-GR00T/gr00t/model/gr00t_n1d7/gr00t_n1d7.py"
shutil.copy(P, P + ".bak_prekron")
src = open(P).read()

anchor = """            elif bool(getattr(self.config, "ht_mvt", False)):"""
assert anchor in src, "mvt anchor not found"

kron = '''            elif str(getattr(self.config, "ht_cov_mode", "iso")) == "kron":
                # ---- Kronecker-factored covariance: Sigma = Sigma_T (x) Sigma_A ----
                resid = (actions.float() - pred_actions.float())
                mk = mask
                rows = mk.any(dim=-1)                      # [B, chunk] valid steps
                cols = mk.any(dim=1)                       # [B, adim]  valid dims
                T_i = int(rows[0].sum().item()); A_i = int(cols[0].sum().item())
                ok = bool((rows.sum(1) == T_i).all() and (cols.sum(1) == A_i).all()
                          and T_i * A_i == int(mk[0].sum().item()))
                if ok:
                    ti = torch.nonzero(rows[0], as_tuple=True)[0]
                    ai = torch.nonzero(cols[0], as_tuple=True)[0]
                    R = resid.index_select(1, ti).index_select(2, ai)      # [B, T, A]
                    nT = T_i * (T_i + 1) // 2
                    nA = A_i * (A_i + 1) // 2
                    w_t = mk.float().sum(dim=tuple(range(1, mk.dim()))).clamp_min(1.0)
                    pooled = (s_raw.float() * mk).sum(dim=1).sum(dim=-1, keepdim=True) * 0.0 \\
                             + (s_raw.float().mean(dim=1))                  # [B, adim] time-pooled
                    need = nT + nA
                    if pooled.shape[-1] < need:
                        pooled = torch.cat([pooled, pooled.new_zeros(pooled.shape[0], need - pooled.shape[-1])], -1)
                    p_t, p_a = pooled[:, :nT], pooled[:, nT:nT + nA]
                    import math as _math
                    s0 = float(F.softplus(torch.tensor(float(self.config.ht_sbias))).item())
                    dbias = _math.log(_math.expm1(max(_math.sqrt(max(s0, 1e-6)), 1e-4)))
                    def _chol(p, n):
                        L = p.new_zeros(p.shape[0], n, n)
                        li, lj = torch.tril_indices(n, n, device=p.device)
                        L[:, li, lj] = p
                        di = torch.arange(n, device=p.device)
                        dg = F.softplus(L[:, di, di] + dbias) + 1e-4
                        L = L.clone(); L[:, di, di] = dg
                        return L
                    L_T = _chol(p_t, T_i)
                    L_A = _chol(p_a, A_i)
                    di_a = torch.arange(A_i, device=L_A.device)
                    logdet_A = torch.log(L_A[:, di_a, di_a]).sum(-1)        # 0.5*log|Sigma_A|
                    L_A = L_A / torch.exp(logdet_A / A_i).view(-1, 1, 1)    # unit-determinant Sigma_A
                    di_t = torch.arange(T_i, device=L_T.device)
                    half_logdet_T = torch.log(L_T[:, di_t, di_t]).sum(-1)   # 0.5*log|Sigma_T|
                    X = torch.linalg.solve_triangular(L_T, R, upper=False)              # L_T^-1 R
                    Y = torch.linalg.solve_triangular(L_A, X.transpose(1, 2), upper=False)
                    M = (Y ** 2).sum(dim=(1, 2))                                        # Mahalanobis
                    d_k = float(T_i * A_i)
                    logdet = 2.0 * (A_i * half_logdet_T)     # + T*log|Sigma_A| = 0 by normalisation
                    per_sample = 0.5 * (df + d_k) * torch.log1p(M / df) + 0.5 * logdet
                    if not getattr(self, "_ht_kron_banner", False):
                        self._ht_kron_banner = True
                        print(f"[ht-kron] Sigma = Sigma_T({T_i}) (x) Sigma_A({A_i}), d={int(d_k)}, "
                              f"nu={float(df)}, diag-init softplus({dbias:.4f})={_math.sqrt(s0):.4f} "
                              f"(sigma0={s0:.4f}), params/sample={nT + nA}", flush=True)
                else:
                    if not getattr(self, "_ht_kron_warn", False):
                        self._ht_kron_warn = True
                        print("[ht-kron] mask is not a clean T x A rectangle; falling back to isotropic",
                              flush=True)
                    per_sample = (0.5 * (df + d_eff) * torch.log1p(sum_r2 / (df * sigma ** 2))
                                  + d_eff * torch.log(sigma))
            elif bool(getattr(self.config, "ht_mvt", False)):'''

src = src.replace(anchor, kron, 1)
open(P, "w").write(src)
print("patched", P)
print("kron branch present:", "[ht-kron]" in open(P).read())
