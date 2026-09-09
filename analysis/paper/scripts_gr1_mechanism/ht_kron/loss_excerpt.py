            model_output, _ = self.model(
                hidden_states=sa_embs,
                encoder_hidden_states=vl_embeds,
                encoder_attention_mask=vl_attn_mask,
                timestep=t_discretized,
                return_all_hidden_states=True,
            )

        pred = self.action_decoder(model_output, embodiment_id)
        pred_actions = pred[:, -actions.shape[1] :]

        # Slice out only the action portion of pred and target.
        action_mask = action_input.action_mask
        if self.config.loss_type == "hetero_t":
            # Heteroscedastic Student-t NLL, matching the MIP-harness
            # regression_hetero_t estimator exactly: a PER-SAMPLE scalar sigma
            # (masked mean of softplus over the chunk) and a multivariate-t NLL
            # over the whole-chunk residual, normalized per masked dim.
            # Computed in fp32: the NLL's log/divide chain is less bf16-tolerant
            # than the flow branch's plain MSE.
            s_raw = self.sigma_decoder(model_output, embodiment_id)[:, -actions.shape[1] :]
            mask = action_mask.float()
            d_eff = mask.sum(dim=tuple(range(1, mask.dim()))).clamp_min(1.0)  # [B]
            r2 = ((actions.float() - pred_actions.float()) ** 2) * mask
            sum_r2 = r2.sum(dim=tuple(range(1, r2.dim())))  # [B]
            m = sum_r2 / d_eff  # [B] gate statistic
            if int(getattr(self.config, "ht_mse_steps", 0) or 0) > 0:
                if bool(getattr(self.config, "mse_mode_now", False)):
                    # phase A: plain masked MSE; sigma receives no gradient here.
                    with torch.no_grad():
                        buf = getattr(self, "_mse_mbuf", None)
                        if buf is None:
                            buf = self._mse_mbuf = []
                        buf.append(float(m.median()))
                        if len(buf) > 100:
                            del buf[0]
                    if not getattr(self, "_mse_banner", False):
                        self._mse_banner = True
                        print(f"[gr00t-mse] MSE phase active for "
                              f"{self.config.ht_mse_steps} steps", flush=True)
                    action_loss = m
                    return {"loss": (sum_r2.sum() / mask.sum().clamp_min(1.0)),
                            "action_loss": action_loss}
                if not getattr(self, "_mse_recal", False):
                    # switch point: recalibrate sbias to the ACTUAL residual scale
                    self._mse_recal = True
                    buf = sorted(getattr(self, "_mse_mbuf", []) or [float(m.median())])
                    med_m = buf[len(buf) // 2]
                    rms = max(med_m, 1e-8) ** 0.5
                    old_sb = float(self.config.ht_sbias)
                    import math as _math
                    self.config.ht_sbias = float(_math.log(_math.expm1(max(rms, 1e-4))))
                    print(f"[gr00t-mse] RECALIB sbias {old_sb:.4f} -> "
                          f"{self.config.ht_sbias:.4f}  (residual rms {rms:.4f}, "
                          f"sigma_new {_math.log1p(_math.exp(self.config.ht_sbias)):.4f}, "
                          f"n_batches {len(buf)})", flush=True)
            sp = F.softplus(s_raw.float() + self.config.ht_sbias) * mask
            sigma = sp.sum(dim=tuple(range(1, sp.dim()))) / d_eff + 1e-3  # [B]
            df = self.config.ht_df
            if getattr(self.config, "ht_shrink", False):
                # Online split-half reliability of m, then shrink toward the batch
                    - torch.lgamma(half) + torch.lgamma(0.5 * nu)
                    + 0.5 * d_eff * torch.log(nu)
                )
                self._lnu_calls = getattr(self, "_lnu_calls", 0) + 1
                if self._lnu_calls % 200 == 1:
                    _q = torch.quantile(nu.detach().float(),
                                        torch.tensor([0.1, 0.5, 0.9], device=nu.device))
                    print(f"[groot-learnnu] call~{self._lnu_calls} nu q10/50/90 = "
                          f"{float(_q[0]):.1f}/{float(_q[1]):.1f}/{float(_q[2]):.1f}", flush=True)
            elif str(getattr(self.config, "ht_cov_mode", "iso")) == "kron":
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
                    pooled = (s_raw.float() * mk).sum(dim=1).sum(dim=-1, keepdim=True) * 0.0 \
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
            elif bool(getattr(self.config, "ht_mvt", False)):
                # true multivariate Student-t (isotropic scale, dimension d_eff)
                if not getattr(self, "_mvt_banner", False):
                    self._mvt_banner = True
                    print(f"[ht-mvt] multivariate Student-t: 0.5*(nu+d)*log1p(S/(nu*sigma^2)), "
                          f"nu={df}", flush=True)
                per_sample = (
                    0.5 * (df + d_eff) * torch.log1p(sum_r2 / (df * sigma**2))
                    + d_eff * torch.log(sigma)
                )
            elif (getattr(self.config, "ht_sum_mode", "inside") != "inside"
                  or getattr(self.config, "ht_sigma_mode", "homog") != "homog"):
                # ---- 2x2 likelihood-shape ablation --------------------------
                sum_mode = getattr(self.config, "ht_sum_mode", "inside")
                sig_mode = getattr(self.config, "ht_sigma_mode", "homog")
                if sig_mode == "perel":
                    # fully per-element sigma: the head's native output, no averaging.
                    # Captures (timestep x action-dim) interaction that neither
                    # marginal shows -- e.g. gripper error concentrated at switches.
                    vec = [float(x) for x in (getattr(self.config, "ht_sbias_vec", ()) or ())]
                    A = s_raw.shape[-1]
                    if len(vec) < A:
                        vec = vec + [float(self.config.ht_sbias)] * (A - len(vec))
                    b = torch.tensor(vec[:A], device=s_raw.device, dtype=torch.float32)
                    b = b.view(*([1] * (s_raw.dim() - 1)), A)
                    sig_el = F.softplus(s_raw.float() + b) + 1e-3
                elif sig_mode == "perdim":
                    # ONE sigma per action dim, shared across time: average the
                    # per-element softplus over the time axis, then broadcast back.
                    # (Without this collapse perdim is identical to perel.)
                    vec = [float(x) for x in (getattr(self.config, "ht_sbias_vec", ()) or ())]
                    A = s_raw.shape[-1]
                    if len(vec) < A:            # pad with the scalar sbias
                        vec = vec + [float(self.config.ht_sbias)] * (A - len(vec))
                    b = torch.tensor(vec[:A], device=s_raw.device, dtype=torch.float32)
                    b = b.view(*([1] * (s_raw.dim() - 1)), A)
                    sp_el = F.softplus(s_raw.float() + b)
                    tdims = tuple(range(1, s_raw.dim() - 1))           # time axes only
                    w = mask.sum(dim=tdims, keepdim=True).clamp_min(1.0)
