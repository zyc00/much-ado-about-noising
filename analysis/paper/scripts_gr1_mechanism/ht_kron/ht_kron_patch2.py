"""Kronecker-factored covariance for the GR00T HT loss (corrected).

Sigma = Sigma_T (x) Sigma_A over (chunk step T) x (action dim A), d = T*A, learned PER SAMPLE.
  M        = tr(Sigma_T^-1 R Sigma_A^-1 R^T)
  log|Sig| = A*log|Sigma_T| + T*log|Sigma_A|   (second term is 0: Sigma_A is unit-determinant)
  loss     = 0.5*(nu+d)*log1p(M/nu) + 0.5*log|Sigma|

Scale lives entirely in Sigma_T because Sigma_A is renormalised to |Sigma_A| = 1 (which fixes the
(c*Sigma_T) (x) (Sigma_A/c) degeneracy). Hence the initialisation puts sigma0 on diag(L_T) and 1 on
diag(L_A), giving Sigma = sigma0^2 I at step 0 -- exactly the isotropic model.
Both factors come from the existing sigma head, pooled over VALID timesteps only, so every sample
gets its own Sigma_T, Sigma_A (no new parameters, checkpoint keys unchanged).
"""
import shutil

P = "/mnt/pfs/yuchen/groot/Isaac-GR00T/gr00t/model/gr00t_n1d7/gr00t_n1d7.py"
shutil.copy(P + ".bak_prekron", P)          # start from the unpatched file
src = open(P).read()
anchor = """            elif bool(getattr(self.config, "ht_mvt", False)):"""
assert anchor in src, "mvt anchor not found"

kron = '''            elif str(getattr(self.config, "ht_cov_mode", "iso")) == "kron":
                # ---- per-sample Kronecker covariance: Sigma = Sigma_T (x) Sigma_A ----
                rows = mask.any(dim=-1)                    # [B, chunk] valid timesteps
                cols = mask.any(dim=1)                     # [B, adim]  valid action dims
                same = bool((rows == rows[:1]).all() and (cols == cols[:1]).all())
                T_i = int(rows[0].sum().item())
                A_i = int(cols[0].sum().item())
                if same and T_i > 0 and A_i > 0 and T_i * A_i == int(mask[0].sum().item()):
                    ti = torch.nonzero(rows[0], as_tuple=True)[0]
                    ai = torch.nonzero(cols[0], as_tuple=True)[0]
                    R = (actions.float() - pred_actions.float()).index_select(1, ti).index_select(2, ai)
                    nT, nA = T_i * (T_i + 1) // 2, A_i * (A_i + 1) // 2
                    # pool the sigma head over VALID timesteps -> one parameter vector per sample
                    rw = rows.float().unsqueeze(-1)
                    pooled = (s_raw.float() * rw).sum(1) / rw.sum(1).clamp_min(1.0)   # [B, adim]
                    if pooled.shape[-1] < nT + nA:
                        raise RuntimeError(f"sigma head width {pooled.shape[-1]} < {nT + nA} needed for kron")
                    import math as _math
                    s0 = float(F.softplus(torch.tensor(float(self.config.ht_sbias))).item())
                    # Sigma_A is unit-determinant, so the whole scale sits in Sigma_T:
                    #   diag(L_T) init = sigma0  -> Sigma_T = sigma0^2 I
                    #   diag(L_A) init = 1       -> Sigma_A = I (normalisation is a no-op at init)
                    b_T = float(self.config.ht_sbias)
                    b_A = _math.log(_math.expm1(1.0))
                    def _chol(p, n, dbias):
                        L = p.new_zeros(p.shape[0], n, n)
                        li, lj = torch.tril_indices(n, n, device=p.device)
                        L[:, li, lj] = p
                        di = torch.arange(n, device=p.device)
                        L[:, di, di] = F.softplus(L[:, di, di] + dbias) + 1e-4
                        return L
                    L_T = _chol(pooled[:, :nT], T_i, b_T)
                    L_A = _chol(pooled[:, nT:nT + nA], A_i, b_A)
                    di_a = torch.arange(A_i, device=L_A.device)
                    L_A = L_A / torch.exp(torch.log(L_A[:, di_a, di_a]).sum(-1) / A_i).view(-1, 1, 1)
                    di_t = torch.arange(T_i, device=L_T.device)
                    X = torch.linalg.solve_triangular(L_T, R, upper=False)                  # L_T^-1 R
                    Y = torch.linalg.solve_triangular(L_A, X.transpose(1, 2), upper=False)  # L_A^-1 (L_T^-1 R)^T
                    M = (Y ** 2).sum(dim=(1, 2))                    # tr(Sig_T^-1 R Sig_A^-1 R^T)
                    d_k = float(T_i * A_i)
                    logdet = 2.0 * A_i * torch.log(L_T[:, di_t, di_t]).sum(-1)   # A*log|Sig_T|
                    per_sample = 0.5 * (df + d_k) * torch.log1p(M / df) + 0.5 * logdet
                    if not getattr(self, "_ht_kron_banner", False):
                        self._ht_kron_banner = True
                        print(f"[ht-kron] per-sample Sigma_T({T_i}) (x) Sigma_A({A_i}), d={int(d_k)}, "
                              f"nu={float(df)}, init diag(L_T)={_math.log1p(_math.exp(b_T)):.4f}=sigma0, "
                              f"diag(L_A)=1.0 -> Sigma_init={s0 ** 2:.4f}*I, params/sample={nT + nA}",
                              flush=True)
                else:
                    if not getattr(self, "_ht_kron_warn", False):
                        self._ht_kron_warn = True
                        print("[ht-kron] action mask is not one shared T x A rectangle; using isotropic",
                              flush=True)
                    per_sample = (0.5 * (df + d_eff) * torch.log1p(sum_r2 / (df * sigma ** 2))
                                  + d_eff * torch.log(sigma))
            elif bool(getattr(self.config, "ht_mvt", False)):'''

open(P, "w").write(src.replace(anchor, kron, 1))
print("re-patched from clean backup; kron branch present:", "[ht-kron]" in open(P).read())
