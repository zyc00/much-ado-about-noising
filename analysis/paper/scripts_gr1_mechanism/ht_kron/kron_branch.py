            elif str(getattr(self.config, "ht_cov_mode", "iso")) == "kron":
                # ---- per-sample Kronecker covariance: Sigma = Sigma_T (x) Sigma_A ----
                mk = mask.bool()
                rows = mk.any(dim=-1)                      # [B, chunk] valid timesteps
                cols = mk.any(dim=1)                       # [B, adim]  valid action dims
                # every sample must share ONE rectangular mask: rows (x) cols with no interior holes
                rect = bool((mk == (rows.unsqueeze(-1) & cols.unsqueeze(1))).all()
                            and (rows == rows[:1]).all() and (cols == cols[:1]).all())
                T_i, A_i = int(rows[0].sum().item()), int(cols[0].sum().item())
                if rect and T_i > 0 and A_i > 0:
                    ti = torch.nonzero(rows[0], as_tuple=True)[0]
                    ai = torch.nonzero(cols[0], as_tuple=True)[0]
                    R = (actions.float() - pred_actions.float()).index_select(1, ti).index_select(2, ai)
                    nT, nA = T_i * (T_i + 1) // 2, A_i * (A_i + 1) // 2
                    # one parameter vector per sample: sigma head pooled over that sample's valid steps
                    rw = rows.float().unsqueeze(-1)
                    pooled = (s_raw.float() * rw).sum(1) / rw.sum(1).clamp_min(1.0)   # [B, adim]
                    if pooled.shape[-1] < nT + nA:
                        raise RuntimeError(f"sigma head width {pooled.shape[-1]} < {nT + nA} for kron")
                    import math as _math
                    FLOOR = 1e-4
                    # match the isotropic branch's effective sigma at a zero head output, floor included
                    s_iso = float(F.softplus(torch.tensor(float(self.config.ht_sbias))).item()) + 1e-3
                    b_T = _math.log(_math.expm1(max(s_iso - FLOOR, 1e-6)))   # diag(L_T) init = s_iso
                    b_A = _math.log(_math.expm1(1.0 - FLOOR))                # diag(L_A) init = 1
                    def _chol(p, n, dbias):
                        L = p.new_zeros(p.shape[0], n, n)
                        li, lj = torch.tril_indices(n, n, device=p.device)
                        L[:, li, lj] = p
                        di = torch.arange(n, device=p.device)
                        L[:, di, di] = F.softplus(L[:, di, di] + dbias) + FLOOR
                        return L
                    L_T = _chol(pooled[:, :nT], T_i, b_T)
                    L_A = _chol(pooled[:, nT:nT + nA], A_i, b_A)
                    di_a = torch.arange(A_i, device=L_A.device)
                    L_A = L_A / torch.exp(torch.log(L_A[:, di_a, di_a]).sum(-1) / A_i).view(-1, 1, 1)
                    di_t = torch.arange(T_i, device=L_T.device)
                    X = torch.linalg.solve_triangular(L_T, R, upper=False)                  # L_T^-1 R
                    Y = torch.linalg.solve_triangular(L_A, X.transpose(1, 2), upper=False)  # L_A^-1 X^T
                    M = (Y ** 2).sum(dim=(1, 2))                    # tr(Sig_T^-1 R Sig_A^-1 R^T)
                    d_k = float(T_i * A_i)
                    logdet = 2.0 * A_i * torch.log(L_T[:, di_t, di_t]).sum(-1)   # A*log|Sig_T|
                    per_sample = 0.5 * (df + d_k) * torch.log1p(M / df) + 0.5 * logdet
                    if not getattr(self, "_ht_kron_banner", False):
                        self._ht_kron_banner = True
                        print(f"[ht-kron] per-sample Sigma_T({T_i}) (x) Sigma_A({A_i}), d={int(d_k)}, "
                              f"nu={float(df)}, init diag(L_T)={s_iso:.6f}=sigma_iso, diag(L_A)=1 -> "
                              f"Sigma_init={s_iso ** 2:.6f}*I, params/sample={nT + nA}", flush=True)
                else:
                    if not getattr(self, "_ht_kron_warn", False):
                        self._ht_kron_warn = True
                        print("[ht-kron] action mask is not one shared hole-free T x A rectangle; "
                              "using isotropic", flush=True)
                    per_sample = (0.5 * (df + d_eff) * torch.log1p(sum_r2 / (df * sigma ** 2))
                                  + d_eff * torch.log(sigma))
            elif bool(getattr(self.config, "ht_mvt", False)):
