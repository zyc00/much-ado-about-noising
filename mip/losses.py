"""Losses for iterative policy training."""

from collections.abc import Callable

import numpy as np
import torch
import torch.nn.functional as F

from mip.config import OptimizationConfig
from mip.encoders import BaseEncoder
from mip.flow_map import FlowMap
from mip.interpolant import Interpolant


def get_norm(x: torch.Tensor, norm_type: str, cauchy_c: float = 0.2) -> torch.Tensor:
    if norm_type == "l2":
        # squared L2 (no sqrt)
        return torch.sum(x * x, dim=-1)
    elif norm_type == "l1":
        return torch.sum(torch.abs(x), dim=-1)
    elif norm_type == "smooth_l1":
        # per-element smooth L1, then sum over last dim
        return torch.sum(
            F.smooth_l1_loss(x, torch.zeros_like(x), reduction="none"), dim=-1
        )
    elif norm_type == "cauchy":
        return torch.sum(torch.log1p((x / cauchy_c) ** 2), dim=-1)
    else:
        raise NotImplementedError(f"Norm type {norm_type} not implemented.")


def get_loss_fn(loss_type: str) -> Callable:
    if loss_type == "flow":
        return flow_loss
    elif loss_type == "regression_cauchy":
        return regression_cauchy_loss
    elif loss_type == "regression_relerr":
        return regression_relerr_loss
    elif loss_type == "regression_residual":
        return regression_residual_loss
    elif loss_type == "mip_siganneal":
        return mip_siganneal_loss
    elif loss_type == "regression_fadehint":
        return regression_fadehint_loss
    elif loss_type == "mip_quant":
        return mip_quant_loss
    elif loss_type == "regression_normed":
        return regression_normed_loss
    elif loss_type == "regression_stdt":
        return regression_stdt_loss
    elif loss_type == "regression_welsch":
        return regression_welsch_loss
    elif loss_type == "regression_welsch_hetero":
        return regression_welsch_hetero_loss
    elif loss_type == "regression_welsch_l2mix":
        return regression_welsch_l2mix_loss
    elif loss_type == "regression_barron":
        return regression_barron_loss
    elif loss_type == "regression_barron_hetero":
        return regression_barron_hetero_loss
    elif loss_type == "regression_barron_hetero_diag":
        return regression_barron_hetero_diag_loss
    elif loss_type == "regression_hetero_t":
        return regression_hetero_t_loss
    elif loss_type == "regression_hetero_t_learnnu":
        return regression_hetero_t_learnnu_loss
    elif loss_type == "regression_hetero_t_learnnu_cond":
        return regression_hetero_t_learnnu_cond_loss
    elif loss_type == "regression_hetero_t_learnnu_cond2":
        return regression_hetero_t_learnnu_cond2_loss
    elif loss_type == "xm":
        return xm_loss

    elif loss_type == "regression_globalt":
        return regression_globalt_loss
    elif loss_type == "regression_gmm":
        return regression_gmm_loss
    elif loss_type == "regression_sigmaw":
        return regression_sigmaw_loss
    elif loss_type == "regression_selfsw":
        return regression_selfsw_loss
    elif loss_type == "regression_invw":
        return regression_invw_loss
    elif loss_type == "regression_gfloor":
        return regression_gfloor_loss
    elif loss_type == "regression_selfnorm":
        return regression_selfnorm_loss
    elif loss_type == "regression_hetero_t_mixnn":
        return regression_hetero_t_mixnn_loss
    elif loss_type == "regression_hetero_t_midlin":
        return regression_hetero_t_midlin_loss
    elif loss_type == "regression_hetero_t_jitcons":
        return regression_hetero_t_jitcons_loss
    elif loss_type == "regression_hetero_gauss_midlin":
        return regression_hetero_gauss_midlin_loss
    elif loss_type == "regression_jspec":
        return regression_jspec_loss
    elif loss_type == "regression_hetero_t_jspec":
        return regression_hetero_t_jspec_loss
    elif loss_type == "regression_hetero_gauss_jspec":
        return regression_hetero_gauss_jspec_loss
    elif loss_type == "regression_hetero_t_rankmid":
        return regression_hetero_t_rankmid_loss
    elif loss_type == "regression_hetero_gauss_rankmid":
        return regression_hetero_gauss_rankmid_loss
    elif loss_type == "regression_focal":
        return regression_focal_loss
    elif loss_type == "regression_fdistill":
        return regression_fdistill_loss
    elif loss_type == "regression_pathdamp":
        return regression_pathdamp_loss
    elif loss_type == "regression_condnum":
        return regression_condnum_loss
    elif loss_type == "regression_condann":
        return regression_condann_loss
    elif loss_type == "regression_condreg":
        return regression_condreg_loss
    elif loss_type == "regression_condreg_axis":
        return regression_condreg_axis_loss
    elif loss_type == "regression_dcr":
        return regression_dcr_loss
    elif loss_type == "regression_hetero_gauss_cnd":
        return regression_hetero_gauss_cnd_loss
    elif loss_type == "regression_hetero_diag":
        return regression_hetero_diag_loss
    elif loss_type == "regression_hetero_t_diag":
        return regression_hetero_t_diag_loss
    elif loss_type == "regression_hetero_t_cnd":
        return regression_hetero_t_cnd_loss
    elif loss_type == "regression_dcw":
        return regression_dcw_loss
    elif loss_type == "regression_labelnoise":
        return regression_labelnoise_loss
    elif loss_type == "regression_featdrop":
        return regression_featdrop_loss
    elif loss_type == "regression_snteach":
        return regression_snteach_loss
    elif loss_type == "regression_emaret":
        return regression_emaret_loss
    elif loss_type == "regression_hgclip":
        return regression_hgclip_loss
    elif loss_type == "regression_hetero_gauss":
        return regression_hetero_gauss_loss
    elif loss_type == "regression_sigaux":
        return regression_sigaux_loss
    elif loss_type == "regression_distinc":
        return regression_distinc_loss
    elif loss_type == "regression_hetero_t_fdlip":
        return regression_hetero_t_fdlip_loss
    elif loss_type == "regression_hetero_t_tublip":
        return regression_hetero_t_tublip_loss
    elif loss_type == "regression_pace":
        return regression_pace_loss
    elif loss_type == "regression_paceabs":
        return regression_paceabs_loss
    elif loss_type == "regression_trim":
        return regression_trim_loss
    elif loss_type == "regression_student_t":
        return regression_student_t_loss
    elif loss_type == "regression":
        return regression_loss
    elif loss_type == "regression_headonly":
        return regression_headonly_loss
    elif loss_type == "regression_binw":
        return regression_binw_loss
    elif loss_type == "regression_dup100":
        return regression_dup100_loss
    elif loss_type == "regression_rw":
        return regression_rw_loss
    elif loss_type == "mip_rw":
        return mip_rw_loss
    elif loss_type == "mip_shufx":
        return mip_shufx_loss
    elif loss_type == "mip_nonoise_atk":
        return mip_nonoise_atk_loss
    elif loss_type == "mip_nonoise":
        return mip_nonoise_loss
    elif loss_type == "mip_distill":
        return mip_distill_loss
    elif loss_type == "mip_lambda":
        return mip_lambda_loss
    elif loss_type == "mip_distill2":
        return mip_distill2_loss
    elif loss_type == "mip_auxtan":
        return mip_auxtan_loss
    elif loss_type == "mip_auxflip":
        return mip_auxflip_loss
    elif loss_type == "mip_auxdetach":
        return mip_auxdetach_loss
    elif loss_type == "mip_step1":
        return mip_loss  # eval-only alias: step1 sampler, mip loss for config compose
    elif loss_type == "regression_frozentrunk":
        return regression_frozentrunk_loss
    elif loss_type == "regression_tauteacher":
        return regression_tauteacher_loss
    elif loss_type == "regression_taufeat":
        return regression_taufeat_loss
    elif loss_type == "regression_badpen":
        return regression_badpen_loss
    elif loss_type == "regression_opanchor":
        return regression_opanchor_loss
    elif loss_type == "mip_tubeaux":
        return mip_tubeaux_loss
    elif loss_type == "mip_scramaux":
        return mip_scramaux_loss
    elif loss_type == "mip_randaux":
        return mip_randaux_loss
    elif loss_type == "regression_cvu":
        return regression_cvu_loss
    elif loss_type == "regression_recgeo":
        return regression_recgeo_loss
    elif loss_type == "regression_pdprior":
        return regression_pdprior_loss
    elif loss_type == "regression_manifold":
        return regression_manifold_loss
    elif loss_type == "regression_obsnoise":
        return regression_obsnoise_loss
    elif loss_type == "regression_offmanjac":
        return regression_offmanjac_loss
    elif loss_type == "regression_geomreg":
        return regression_geomreg_loss
    elif loss_type == "regression_stressreg":
        return regression_stressreg_loss
    elif loss_type == "regression_randaug":
        return regression_randaug_loss
    elif loss_type == "regression_sigreg":
        return regression_sigreg_loss
    elif loss_type == "regression_tjitter":
        return regression_tjitter_loss
    elif loss_type == "regression_jacreg":
        return regression_jacreg_loss
    elif loss_type == "mip_eqw":
        return mip_eqw_loss
    elif loss_type == "denoise_only":
        return denoise_only_loss
    elif loss_type == "denoise_zeroin":
        return denoise_zeroin_loss
    elif loss_type == "denoise_zeroin_flat":
        return denoise_zeroin_flat_loss
    elif loss_type == "denoise_randin":
        return denoise_randin_loss
    elif loss_type == "denoise_scramble":
        return denoise_scramble_loss
    elif loss_type == "straight_flow":
        return straight_flow_loss
    elif loss_type == "tsd":
        return tsd_loss
    elif loss_type == "mip_heterov1":
        return mip_heterov1_loss
    elif loss_type == "mip_cauchyv1":
        return mip_cauchyv1_loss
    elif loss_type == "mip":
        return mip_loss
    elif loss_type == "mip_auxdet":
        return mip_auxdet_loss
    elif loss_type == "mip_zeroaux":
        return mip_zeroaux_loss
    elif loss_type == "lmd":
        return lmd_loss
    elif loss_type == "ctm":
        return ctm_loss
    elif loss_type == "psd":
        return psd_loss
    elif loss_type == "lsd":
        return lsd_loss
    elif loss_type == "esd":
        return esd_loss
    elif loss_type == "mf":
        return mf_loss
    else:
        raise NotImplementedError(f"Loss type {loss_type} not implemented.")


def flow_loss(
    config: OptimizationConfig,
    flow_map: FlowMap,
    encoder: BaseEncoder,
    interp: Interpolant,
    act: torch.Tensor,
    obs: torch.Tensor,
    delta_t: torch.Tensor,
) -> float:
    """Flow model loss, matching the velocity field.

    Args:
        flow_map (FlowMap): the flow map
        interp (Interpolant): the interpolant
        obs (torch.Tensor): the target state
        obs (torch.Tensor): the label
        delta_t (torch.Tensor): the time step difference, used for flow map / shortcut model / consistency training only.

    Returns:
        float: the loss
    """
    # sample - use empty+uniform_/normal_ for CUDA graph compatibility
    t = torch.empty_like(delta_t).uniform_(0, 1)
    act_0 = torch.empty_like(act).normal_(0, 1)
    act_1 = act

    # get condition
    obs_emb = encoder(obs, None)

    # predict
    act_t = interp.calc_It(t, act_0, act_1)
    act_t_dot = interp.calc_It_dot(t, act_0, act_1)
    b_t = flow_map.get_velocity(t, act_t, obs_emb)

    # compute loss
    loss = get_norm(b_t - act_t_dot, config.norm_type, config.cauchy_c)
    loss = config.loss_scale * torch.mean(loss)
    return loss, {}


def regression_cauchy_loss(
    config: OptimizationConfig,
    flow_map: FlowMap,
    encoder: BaseEncoder,
    interp: Interpolant,
    act: torch.Tensor,
    obs: torch.Tensor,
    delta_t: torch.Tensor,
) -> float:
    """Regression with Cauchy/Lorentzian loss — log(1 + (e/c)^2)."""
    t = torch.zeros_like(delta_t, device=delta_t.device)
    act_0 = torch.zeros_like(act, device=act.device)
    obs_emb = encoder(obs, None)
    act_pred = flow_map.get_velocity(t, act_0, obs_emb)

    c = config.cauchy_c
    loss = torch.log1p(((act_pred - act) / c) ** 2)
    loss = config.loss_scale * torch.mean(loss)
    return loss, {}


def regression_normed_loss(
    config: OptimizationConfig,
    flow_map: FlowMap,
    encoder: BaseEncoder,
    interp: Interpolant,
    act: torch.Tensor,
    obs: torch.Tensor,
    delta_t: torch.Tensor,
) -> float:
    """PURE GRADIENT REBALANCING (maximal per-sample form): loss = mean ||r||_2
    (unsquared). Every sample contributes a UNIT-NORM gradient direction — the extreme
    equalization of the per-sample gradient budget; no hyperparameters, no aux, no
    architecture change."""
    t = torch.zeros_like(delta_t, device=delta_t.device)
    act_0 = torch.zeros_like(act, device=act.device)
    obs_emb = encoder(obs, None)
    act_pred = flow_map.get_velocity(t, act_0, obs_emb)
    r = act_pred - act
    per = r.reshape(len(act), -1).norm(dim=1)
    loss = config.loss_scale * torch.mean(per)
    return loss, {}


_STDT_STATE = {}


def regression_stdt_loss(
    config: OptimizationConfig,
    flow_map: FlowMap,
    encoder: BaseEncoder,
    interp: Interpolant,
    act: torch.Tensor,
    obs: torch.Tensor,
    delta_t: torch.Tensor,
) -> float:
    """PURE GRADIENT REBALANCING (per-channel + per-sample + tail form): residuals
    standardized per action-dim by an EMA scale (channel rebalancing), per-sample learned
    scale from the scalar head (state-conditional rebalancing), Student-t nu=2 tails
    (sample rebalancing). The maximal principled rebalancer; no aux, no arch change."""
    t = torch.zeros_like(delta_t, device=delta_t.device)
    act_0 = torch.zeros_like(act, device=act.device)
    obs_emb = encoder(obs, None)
    act_pred, s_raw = flow_map.net(act_0, t, t, obs_emb)
    r = act_pred - act
    r2d = (r.detach() ** 2).mean(dim=(0, 1))
    key = "ema"
    if key not in _STDT_STATE:
        _STDT_STATE[key] = r2d.clone()
    else:
        _STDT_STATE[key] = 0.99 * _STDT_STATE[key] + 0.01 * r2d
    sd = (_STDT_STATE[key] + 1e-8).sqrt()
    sb = torch.nn.functional.softplus(s_raw).reshape(len(act), -1).mean(dim=1) + 1e-3
    z2 = (r / sd) ** 2 / sb.view(-1, 1, 1) ** 2
    nu = config.student_t_df
    D = r[0].numel()
    per = 0.5 * (nu + 1.0) * torch.log1p(z2 / nu).sum(dim=(1, 2)) + D * torch.log(sb)
    loss = config.loss_scale * torch.mean(per) / D
    return loss, {}


def regression_hetero_t_learnnu_loss(
    config: OptimizationConfig,
    flow_map: FlowMap,
    encoder: BaseEncoder,
    interp: Interpolant,
    act: torch.Tensor,
    obs: torch.Tensor,
    delta_t: torch.Tensor,
) -> float:
    """hetero-t with LEARNABLE degrees of freedom: nu = 0.5 + softplus(raw),
    raw registered on flow_map (TrainingAgent) so it joins the optimizer.
    Unlike the fixed-nu loss, the nu-dependent normalizer (lgamma terms) is
    kept - it is what trades tail-heaviness against density mass, making
    the nu-gradient a proper MLE signal."""
    t = torch.zeros_like(delta_t, device=delta_t.device)
    act_0 = torch.zeros_like(act, device=act.device)
    obs_emb = encoder(obs, None)
    act_pred, s_raw = flow_map.net(act_0, t, t, obs_emb)
    sigma = (
        torch.nn.functional.softplus(s_raw).reshape(len(act), -1).mean(dim=1)
        + 1e-3
    )
    nu = 0.5 + torch.nn.functional.softplus(flow_map.learn_nu_raw)
    r2 = (act_pred - act) ** 2
    n_dim = r2[0].numel()
    sum_r2 = r2.sum(dim=tuple(range(1, r2.dim())))
    per = (
        0.5 * (nu + 1.0) * torch.log1p(sum_r2 / (nu * sigma**2 * n_dim)) * n_dim
        + n_dim * torch.log(sigma)
        + n_dim
        * (
            torch.lgamma(0.5 * nu)
            - torch.lgamma(0.5 * (nu + 1.0))
            + 0.5 * torch.log(nu)
        )
    )
    loss = config.loss_scale * torch.mean(per) / n_dim
    return loss, {}


def regression_hetero_t_learnnu_cond2_loss(
    config: OptimizationConfig,
    flow_map: FlowMap,
    encoder: BaseEncoder,
    interp: Interpolant,
    act: torch.Tensor,
    obs: torch.Tensor,
    delta_t: torch.Tensor,
) -> float:
    """State-conditional learnable nu, v2 -- NO SIGMOID (see PART CDXLII).

    v1 used nu = 1 + 29*sigmoid(base + gate*head): the base drifted to the
    sigmoid ceiling, the gradient vanished there, and the conditional spread
    collapsed to exactly zero (p10 == p90 == 27.6) -- the bound became an
    attractor.  v2 removes every saturating nonlinearity:

        nu(s) = nu_floor + softplus(nu_base_raw + gate * g(s))

    softplus is monotone with a gradient that never vanishes upward, so
    there is no ceiling to stick to; the multiplicative reading is
    exp-shaped near the base but cannot blow up because g is shrunk toward
    0 by nu_cond_reg and (optionally) hard-capped by NU_MAX for stability
    only -- the cap sits far above the operating range rather than at the
    edge of it.  Base init puts nu at NU_INIT (default 2, the measured
    demo df) instead of v1's 4, so deviations are learned around the value
    the fixed-nu recipe uses.
    """
    import os as _os

    t = torch.zeros_like(delta_t, device=delta_t.device)
    act_0 = torch.zeros_like(act, device=act.device)
    obs_emb = encoder(obs, None)
    act_pred, s_raw = flow_map.net(act_0, t, t, obs_emb)
    sigma = (
        torch.nn.functional.softplus(s_raw).reshape(len(act), -1).mean(dim=1)
        + 1e-3
    )

    nu_floor = float(_os.environ.get("NU_FLOOR", "1.0"))
    nu_max = float(_os.environ.get("NU_MAX", "40.0"))
    nu_mode = _os.environ.get("NU_MODE", "softplus")  # softplus | exp
    dnu_raw = flow_map.nu_cond_head(obs_emb.reshape(len(act), -1)).reshape(-1)
    gate = torch.clamp(
        flow_map.nu_cond_step / max(float(config.nu_cond_warmup), 1.0), 0.0, 1.0
    )
    if nu_mode == "exp":
        # multiplicative: nu(s) = nu_base * exp(gate * g(s)), nu_base>floor.
        # g is a log-ratio, so deviations are symmetric in relative terms
        # (g=+0.7 doubles nu, g=-0.7 halves it) and the shrinkage penalty
        # g^2 is a proper log-space prior around the global base.
        nu_dev = torch.clamp(gate * dnu_raw, min=-3.0, max=3.0)
        nu_base = nu_floor + torch.nn.functional.softplus(flow_map.nu_base_raw)
        nu = nu_base * torch.exp(nu_dev)
    else:
        # additive-in-raw: nu(s) = floor + softplus(base + gate * g(s))
        nu = nu_floor + torch.nn.functional.softplus(
            flow_map.nu_base_raw + gate * dnu_raw
        )
    nu = torch.clamp(nu, min=nu_floor, max=nu_max)

    r2 = (act_pred - act) ** 2
    n_dim = r2[0].numel()
    sum_r2 = r2.sum(dim=tuple(range(1, r2.dim())))
    per = (
        0.5 * (nu + 1.0) * torch.log1p(sum_r2 / (nu * sigma**2 * n_dim)) * n_dim
        + n_dim * torch.log(sigma)
        + n_dim
        * (
            torch.lgamma(0.5 * nu)
            - torch.lgamma(0.5 * (nu + 1.0))
            + 0.5 * torch.log(nu)
        )
    )
    loss = (
        config.loss_scale * torch.mean(per) / n_dim
        + config.nu_cond_reg * torch.mean(dnu_raw**2)
    )
    flow_map._nu_stats = (
        nu.detach().mean(),
        torch.quantile(nu.detach(), 0.1),
        torch.quantile(nu.detach(), 0.9),
    )
    return loss, {}


def regression_hetero_t_learnnu_cond_loss(
    config: OptimizationConfig,
    flow_map: FlowMap,
    encoder: BaseEncoder,
    interp: Interpolant,
    act: torch.Tensor,
    obs: torch.Tensor,
    delta_t: torch.Tensor,
) -> float:
    """State-CONDITIONAL learnable nu with anti-collapse constraints:
    nu(s) = 1 + 29*sigmoid(nu_base + gate * head(obs_emb)) in [1, 30];
    gate ramps 0->1 after nu_cond_warmup steps (mean/sigma fit first, so
    early model error cannot be absorbed as heavy tails); deviation
    regularizer nu_cond_reg * mean(head^2) shrinks state-variation toward
    the pooled-MLE global base (nu-sigma identifiability). Full t NLL
    normalizer kept, as in learnnu."""
    t = torch.zeros_like(delta_t, device=delta_t.device)
    act_0 = torch.zeros_like(act, device=act.device)
    obs_emb = encoder(obs, None)
    act_pred, s_raw = flow_map.net(act_0, t, t, obs_emb)
    sigma = (
        torch.nn.functional.softplus(s_raw).reshape(len(act), -1).mean(dim=1)
        + 1e-3
    )
    dnu_raw = flow_map.nu_cond_head(obs_emb.reshape(len(act), -1)).reshape(-1)
    gate = torch.clamp(
        flow_map.nu_cond_step / max(float(config.nu_cond_warmup), 1.0), 0.0, 1.0
    )
    nu = 1.0 + 29.0 * torch.sigmoid(flow_map.nu_base_raw + gate * dnu_raw)
    r2 = (act_pred - act) ** 2
    n_dim = r2[0].numel()
    sum_r2 = r2.sum(dim=tuple(range(1, r2.dim())))
    per = (
        0.5 * (nu + 1.0) * torch.log1p(sum_r2 / (nu * sigma**2 * n_dim)) * n_dim
        + n_dim * torch.log(sigma)
        + n_dim
        * (
            torch.lgamma(0.5 * nu)
            - torch.lgamma(0.5 * (nu + 1.0))
            + 0.5 * torch.log(nu)
        )
    )
    loss = (
        config.loss_scale * torch.mean(per) / n_dim
        + config.nu_cond_reg * torch.mean(dnu_raw**2)
    )
    flow_map._nu_stats = (
        nu.detach().mean(),
        torch.quantile(nu.detach(), 0.1),
        torch.quantile(nu.detach(), 0.9),
    )
    return loss, {}


def _gnc_anneal(fn, base, start_env, frac_env, config):
    """Graduated non-convexity: anneal a kernel scale from a loose start
    to its target over the first frac of training. Counts calls on the
    loss function itself (one call per gradient step)."""
    import os as _os
    start = float(_os.environ.get(start_env, "0"))
    if start <= 0:
        return base
    frac = float(_os.environ.get(frac_env, "0.4"))
    step = getattr(fn, "_gnc_step", 0)
    fn._gnc_step = step + 1
    total = max(int(config.gradient_steps * frac), 1)
    w = max(0.0, 1.0 - step / total)
    return base + (start - base) * w


def regression_welsch_loss(
    config: OptimizationConfig,
    flow_map: FlowMap,
    encoder: BaseEncoder,
    interp: Interpolant,
    act: torch.Tensor,
    obs: torch.Tensor,
    delta_t: torch.Tensor,
) -> float:
    """Direct reverse-KL surrogate (correntropy/Welsch): minimizing
    1 - exp(-||r||^2 / (2 h^2 D)) maximizes the h-smoothed conditional
    label density at the prediction (sample-based mode-seeking).
    h -> inf recovers MSE; kernel width from WELSCH_H (normalized action
    units, default 0.1)."""
    t = torch.zeros_like(delta_t, device=delta_t.device)
    act_0 = torch.zeros_like(act, device=act.device)
    obs_emb = encoder(obs, None)
    act_pred, _ = flow_map.net(act_0, t, t, obs_emb)
    import os as _os
    h = float(_os.environ.get("WELSCH_H", "0.1"))
    h = _gnc_anneal(regression_welsch_loss, h, "WELSCH_H_START",
                    "WELSCH_WARM_FRAC", config)
    r2 = (act_pred - act) ** 2
    d = r2[0].numel()
    sum_r2 = r2.sum(dim=tuple(range(1, r2.dim())))
    loss = config.loss_scale * torch.mean(
        1.0 - torch.exp(-sum_r2 / (2.0 * h * h * d)))
    return loss, {}


def regression_welsch_l2mix_loss(
    config: OptimizationConfig,
    flow_map: FlowMap,
    encoder: BaseEncoder,
    interp: Interpolant,
    act: torch.Tensor,
    obs: torch.Tensor,
    delta_t: torch.Tensor,
) -> float:
    """Linear homotopy L2 -> Welsch: (1-lam)*MSE + lam*Welsch(h), with
    lam ramping 0 -> 1 over the first WELSCH_MIX_FRAC of training and h
    fixed at the target width. The L2 term keeps gradients alive at any
    residual until the endpoint is pure Welsch. Env: WELSCH_H,
    WELSCH_MIX_FRAC."""
    t = torch.zeros_like(delta_t, device=delta_t.device)
    act_0 = torch.zeros_like(act, device=act.device)
    obs_emb = encoder(obs, None)
    act_pred, _ = flow_map.net(act_0, t, t, obs_emb)
    import os as _os
    h = float(_os.environ.get("WELSCH_H", "0.05"))
    frac = float(_os.environ.get("WELSCH_MIX_FRAC", "0.5"))
    fn = regression_welsch_l2mix_loss
    step = getattr(fn, "_step", 0)
    fn._step = step + 1
    lam = min(1.0, step / max(int(config.gradient_steps * frac), 1))
    r2 = (act_pred - act) ** 2
    d = r2[0].numel()
    sum_r2 = r2.sum(dim=tuple(range(1, r2.dim())))
    wel = 1.0 - torch.exp(-sum_r2 / (2.0 * h * h * d))
    mse = 0.5 * sum_r2 / d
    per = (1.0 - lam) * mse + lam * wel
    return config.loss_scale * torch.mean(per), {}


def regression_welsch_hetero_loss(
    config: OptimizationConfig,
    flow_map: FlowMap,
    encoder: BaseEncoder,
    interp: Interpolant,
    act: torch.Tensor,
    obs: torch.Tensor,
    delta_t: torch.Tensor,
) -> float:
    """Welsch with LEARNED per-state kernel width (cold-start fix: the
    width stays wide where residuals are large, so gradients never
    vanish). Penalty WELSCH_BETA * log(sigma) prevents the width from
    diverging; floor WELSCH_SMIN prevents collapse."""
    t = torch.zeros_like(delta_t, device=delta_t.device)
    act_0 = torch.zeros_like(act, device=act.device)
    obs_emb = encoder(obs, None)
    act_pred, s_raw = flow_map.net(act_0, t, t, obs_emb)
    import os as _os
    beta = float(_os.environ.get("WELSCH_BETA", "1.0"))
    smin = float(_os.environ.get("WELSCH_SMIN", "0.02"))
    sigma = torch.nn.functional.softplus(s_raw).reshape(
        len(act), -1).mean(dim=1) + smin
    r2 = (act_pred - act) ** 2
    d = r2[0].numel()
    sum_r2 = r2.sum(dim=tuple(range(1, r2.dim())))
    per = 1.0 - torch.exp(-sum_r2 / (2.0 * sigma ** 2 * d))         + beta * torch.log(sigma)
    return config.loss_scale * torch.mean(per), {}


def regression_barron_loss(
    config: OptimizationConfig,
    flow_map: FlowMap,
    encoder: BaseEncoder,
    interp: Interpolant,
    act: torch.Tensor,
    obs: torch.Tensor,
    delta_t: torch.Tensor,
) -> float:
    """Barron general robust loss (fixed alpha, scale c): rho(r) =
    (|2-a|/a) * (((r/c)^2/|2-a| + 1)^(a/2) - 1). alpha=2 -> L2,
    0 -> Cauchy, -2 -> Geman-McClure, -inf -> Welsch. Polynomial
    redescending tails keep cold-start gradients alive. Env: BARRON_A,
    BARRON_C. Chunk-norm convention matching the HT loss."""
    t = torch.zeros_like(delta_t, device=delta_t.device)
    act_0 = torch.zeros_like(act, device=act.device)
    obs_emb = encoder(obs, None)
    act_pred, _ = flow_map.net(act_0, t, t, obs_emb)
    import os as _os
    a = float(_os.environ.get("BARRON_A", "-2.0"))
    c = float(_os.environ.get("BARRON_C", "0.1"))
    c = _gnc_anneal(regression_barron_loss, c, "BARRON_C_START",
                    "BARRON_WARM_FRAC", config)
    r2 = (act_pred - act) ** 2
    d = r2[0].numel()
    x = r2.sum(dim=tuple(range(1, r2.dim()))) / (c * c * d)
    if abs(a) < 1e-6:
        per = torch.log1p(0.5 * x)
    else:
        b = abs(2.0 - a)
        per = (b / a) * ((x / b + 1.0) ** (a / 2.0) - 1.0)
    return config.loss_scale * torch.mean(per), {}


def regression_barron_hetero_loss(
    config: OptimizationConfig,
    flow_map: FlowMap,
    encoder: BaseEncoder,
    interp: Interpolant,
    act: torch.Tensor,
    obs: torch.Tensor,
    delta_t: torch.Tensor,
) -> float:
    """Barron rho with LEARNED per-state scale c(s) plus the D*log(c)
    normalizer (MLE-style): adaptive-width cold start, self-annealing
    sharpness. Env: BARRON_A (fixed alpha), BARRON_BETA, BARRON_CMIN.
    Optional BARRON_MIX_FRAC: linear L2->Barron homotopy, lam ramping
    0 -> 1 over the first BARRON_MIX_FRAC of training (GNC-style convex
    warmup; the log(c) normalizer stays on so c(s) trains throughout)."""
    t = torch.zeros_like(delta_t, device=delta_t.device)
    act_0 = torch.zeros_like(act, device=act.device)
    obs_emb = encoder(obs, None)
    act_pred, s_raw = flow_map.net(act_0, t, t, obs_emb)
    import os as _os
    a = float(_os.environ.get("BARRON_A", "-2.0"))
    beta = float(_os.environ.get("BARRON_BETA", "1.0"))
    cmin = float(_os.environ.get("BARRON_CMIN", "0.02"))
    c = torch.nn.functional.softplus(s_raw).reshape(
        len(act), -1).mean(dim=1) + cmin
    r2 = (act_pred - act) ** 2
    d = r2[0].numel()
    sum_r2 = r2.sum(dim=tuple(range(1, r2.dim())))
    x = sum_r2 / (c * c * d)
    if abs(a) < 1e-6:
        rho = torch.log1p(0.5 * x)
    else:
        b = abs(2.0 - a)
        rho = (b / a) * ((x / b + 1.0) ** (a / 2.0) - 1.0)
    per = rho + beta * torch.log(c)
    mix = _os.environ.get("BARRON_MIX_FRAC")
    if mix is not None:
        fn = regression_barron_hetero_loss
        step = getattr(fn, "_step", 0)
        fn._step = step + 1
        lam = min(1.0, step / max(
            int(config.gradient_steps * float(mix)), 1))
        per = (1.0 - lam) * 0.5 * sum_r2 / d + lam * per
    return config.loss_scale * torch.mean(per), {}


def regression_barron_hetero_diag_loss(
    config: OptimizationConfig,
    flow_map: FlowMap,
    encoder: BaseEncoder,
    interp: Interpolant,
    act: torch.Tensor,
    obs: torch.Tensor,
    delta_t: torch.Tensor,
) -> float:
    """Anisotropic hetero-Barron: learned per-sample scale c(s) times an
    EMA per-dimension residual profile (mode boundaries differ per sample
    AND per action dimension). Env: BARRON_A, BARRON_BETA, BARRON_CMIN,
    HGD_EMA."""
    import os as _os
    t = torch.zeros_like(delta_t, device=delta_t.device)
    act_0 = torch.zeros_like(act, device=act.device)
    obs_emb = encoder(obs, None)
    act_pred, s_raw = flow_map.net(act_0, t, t, obs_emb)
    a = float(_os.environ.get("BARRON_A", "-2.0"))
    beta = float(_os.environ.get("BARRON_BETA", "1.0"))
    cmin = float(_os.environ.get("BARRON_CMIN", "0.02"))
    ema = float(_os.environ.get("HGD_EMA", "0.99"))
    c = torch.nn.functional.softplus(s_raw).reshape(
        len(act), -1).mean(dim=1) + cmin
    r = (act_pred - act).reshape(len(act), -1)
    d = r.shape[1]
    with torch.no_grad():
        rms = r.detach().pow(2).mean(dim=0).sqrt() + 1e-4
        prof = getattr(regression_barron_hetero_diag_loss, "_prof", None)
        if prof is None or prof.shape != rms.shape:
            prof = rms.clone()
        else:
            prof = ema * prof + (1 - ema) * rms
        regression_barron_hetero_diag_loss._prof = prof
        proen = prof / prof.mean()
    x = (r ** 2 / (proen[None, :] ** 2)).sum(dim=1) / (c ** 2 * d)
    if abs(a) < 1e-6:
        rho = torch.log1p(0.5 * x)
    else:
        b = abs(2.0 - a)
        rho = (b / a) * ((x / b + 1.0) ** (a / 2.0) - 1.0)
    per = rho + beta * torch.log(c)
    return config.loss_scale * torch.mean(per), {}


def regression_hetero_t_loss(
    config: OptimizationConfig,
    flow_map: FlowMap,
    encoder: BaseEncoder,
    interp: Interpolant,
    act: torch.Tensor,
    obs: torch.Tensor,
    delta_t: torch.Tensor,
) -> float:
    """DESIGNED FROM THE MECHANISM (PART CLXVII): Student-t NLL with LEARNED per-sample
    scale, read from the network's scalar head. Adaptive, state-conditional budget
    equalization — generalizes Cauchy (fixed c) per the crowding-out account; nu matches
    the measured df~2 tails. Single-pass; pre-registered ceiling = MIP step-1 band."""
    t = torch.zeros_like(delta_t, device=delta_t.device)
    act_0 = torch.zeros_like(act, device=act.device)
    obs_emb = encoder(obs, None)
    act_pred, s_raw = flow_map.net(act_0, t, t, obs_emb)
    import os as _os
    _sb = float(_os.environ.get("HT_SBIAS", "0"))
    _smin = float(_os.environ.get("HT_SMIN", "0"))
    sigma = torch.nn.functional.softplus(s_raw + _sb).reshape(len(act), -1).mean(dim=1) + 1e-3 + _smin
    nu = config.student_t_df
    r2 = (act_pred - act) ** 2
    _dead = float(_os.environ.get("HT_DEAD", "0"))
    _sum_r2 = r2.sum(dim=tuple(range(1, r2.dim())))
    if _dead > 0:
        # dead zone: residuals below _dead (per-dim RMS) contribute no loss —
        # removes the small-error attractor (anti-memorization floor)
        _sum_r2 = torch.relu(_sum_r2 - r2[0].numel() * _dead ** 2)
    per = 0.5 * (nu + 1.0) * torch.log1p(
        _sum_r2 / (nu * sigma ** 2 * r2[0].numel())
    ) * r2[0].numel() + r2[0].numel() * torch.log(sigma)
    loss = config.loss_scale * torch.mean(per) / r2[0].numel()
    _mix = float(_os.environ.get("HT_MIX", "0"))
    if _mix > 0:
        # additive MSE term: restores the unbounded pocket gradient sink
        # alongside the t term's normalization (anti-spiral hybrid)
        loss = loss + config.loss_scale * _mix * r2.mean()
    return loss, {}


_SIGMAW_STATE = {}


def regression_invw_loss(
    config: OptimizationConfig,
    flow_map: FlowMap,
    encoder: BaseEncoder,
    interp: Interpolant,
    act: torch.Tensor,
    obs: torch.Tensor,
    delta_t: torch.Tensor,
) -> float:
    """ANTI-RETIREMENT CONTROL (PART CCCXXX pre-reg): plain MSE with per-sample
    weights w ~ m^(-INVW_POW) (m = detached per-sample MSE; POW=1 equalizes
    loss, POW=0.5 equalizes gradient norm), clipped [0.02, 50], mean-1.
    Directly keeps near-converged distinctions funded. Predictions: on
    SCRIPTED data reproduces the hetero-G link-4 rescue (retirement is the
    mechanism); on HUMAN data catastrophic (inverse weighting amplifies
    noise) — the two-sided symmetry that motivates a learned sigma map."""
    import os as _os

    t = torch.zeros_like(delta_t, device=delta_t.device)
    act_0 = torch.zeros_like(act, device=act.device)
    obs_emb = encoder(obs, None)
    act_pred, _ = flow_map.net(act_0, t, t, obs_emb)
    per = ((act_pred - act) ** 2).reshape(len(act), -1).mean(dim=1)
    p = float(_os.environ.get("INVW_POW", "1.0"))
    with torch.no_grad():
        w = (per.detach() + 1e-8) ** (-p)
        w = torch.clamp(w / w.mean(), 0.02, 50.0)
    loss = config.loss_scale * torch.mean(per * w)
    return loss, {}


_GF_STATE = {}
_EMARET_STATE = {}


def regression_gfloor_loss(
    config: OptimizationConfig,
    flow_map: FlowMap,
    encoder: BaseEncoder,
    interp: Interpolant,
    act: torch.Tensor,
    obs: torch.Tensor,
    delta_t: torch.Tensor,
) -> float:
    """ANTI-RETIREMENT CONTROL 1 (PART CCCXXXVI pre-reg): MSE with a gradient
    FLOOR — samples whose residual falls below tau (EMA of batch median) get
    boosted w = clamp(tau/m, 1, WMAX); nobody is ever down-weighted. The
    surgical fix of invw's inversion: bounded re-funding of converged
    content, zero starvation of still-learning content."""
    import os as _os

    t = torch.zeros_like(delta_t, device=delta_t.device)
    act_0 = torch.zeros_like(act, device=act.device)
    obs_emb = encoder(obs, None)
    act_pred, _ = flow_map.net(act_0, t, t, obs_emb)
    per = ((act_pred - act) ** 2).reshape(len(act), -1).mean(dim=1)
    c = float(_os.environ.get("GF_C", "0.5"))
    wmax = float(_os.environ.get("GF_WMAX", "20.0"))
    q = float(_os.environ.get("GF_Q", "0.5"))
    with torch.no_grad():
        med = (per.detach().median() if q == 0.5
               else torch.quantile(per.detach(), q))
        _GF_STATE["tau"] = (med if "tau" not in _GF_STATE
                            else 0.99 * _GF_STATE["tau"] + 0.01 * med)
        w = torch.clamp(c * _GF_STATE["tau"] / (per.detach() + 1e-12), 1.0, wmax)
    loss = config.loss_scale * torch.mean(per * w)
    return loss, {}


def regression_featdrop_loss(
    config: OptimizationConfig,
    flow_map: FlowMap,
    encoder: BaseEncoder,
    interp: Interpolant,
    act: torch.Tensor,
    obs: torch.Tensor,
    delta_t: torch.Tensor,
) -> float:
    """PART CCCLXII: plain MSE + per-FEATURE dropout on the observation
    (a whole feature is masked in BOTH frames, so it cannot be recovered
    from the other frame). This FORCES the policy to spread its reliance
    across features — the user's "committed to more features" mechanism —
    without touching the loss geometry, the targets, or the gain.
    If feature breadth is the lever, this should raise PRfeat AND SR; if it
    is only a marker, PRfeat rises and SR does not. Inference is unmasked.
    Env: FEATDROP_P (0.1)."""
    import os as _os

    p = float(_os.environ.get("FEATDROP_P", "0.1"))
    t = torch.zeros_like(delta_t, device=delta_t.device)
    act_0 = torch.zeros_like(act, device=act.device)
    obs_t = obs["state"] if isinstance(obs, dict) else obs
    if flow_map.training and p > 0:
        keep = (torch.rand((obs_t.shape[0], 1, obs_t.shape[-1]),
                           device=obs_t.device) > p).to(obs_t.dtype)
        obs_t = obs_t * keep / (1.0 - p)
    emb = encoder({"state": obs_t} if isinstance(obs, dict) else obs_t, None)
    pred, _ = flow_map.net(act_0, t, t, emb)
    loss = config.loss_scale * ((pred - act) ** 2).mean()
    return loss, {}


def _condreg_penalty(flow_map, encoder, act, obs, t, act_0, pred):
    """Shared bounded spectral-flatness penalty (see regression_condreg)."""
    import os as _os

    tau = float(_os.environ.get("CONDREG_TAU", "1.0"))
    eps = float(_os.environ.get("CONDREG_EPS", "0.05"))
    K = int(_os.environ.get("CONDREG_K", "6"))
    obs_t = obs["state"] if isinstance(obs, dict) else obs
    d2s = []
    for _k in range(K):
        v = torch.randn_like(obs_t)
        v = v / (v.reshape(len(act), -1).norm(dim=1)
                 .reshape(-1, *([1] * (v.dim() - 1))) + 1e-9)
        emb_p = encoder({"state": obs_t + eps * v}, None)
        pred_p, _ = flow_map.net(act_0, t, t, emb_p)
        d2s.append(((pred_p - pred) ** 2).reshape(len(act), -1).mean(dim=1))
    D = torch.stack(d2s, 1)
    cv2 = D.var(dim=1) / (D.mean(dim=1) ** 2 + 1e-12)
    return torch.relu(cv2 - tau).mean()


_HGD_STATE = {}


def regression_hetero_diag_loss(
    config: OptimizationConfig,
    flow_map: FlowMap,
    encoder: BaseEncoder,
    interp: Interpolant,
    act: torch.Tensor,
    obs: torch.Tensor,
    delta_t: torch.Tensor,
) -> float:
    """ANISOTROPIC (diagonal-covariance) hetero-Gauss: sigma_d(x) =
    softplus(s_raw(x)) * c_d, c_d = EMA per-action-dim residual RMS
    (detached, normalized to geometric mean 1). Tests whether the hetero
    family's cap (HG 87 < MIP 94) is the ISOTROPY of its noise model or
    the unbiased-mean assumption. Env: HGD_EMA (0.99)."""
    import os as _os

    ema = float(_os.environ.get("HGD_EMA", "0.99"))
    mode = _os.environ.get("HGD_MODE", "dim")   # dim (D,) | stepdim (H,D)
    t = torch.zeros_like(delta_t, device=delta_t.device)
    act_0 = torch.zeros_like(act, device=act.device)
    obs_emb = encoder(obs, None)
    act_pred, s_raw = flow_map.net(act_0, t, t, obs_emb)
    sigma = torch.nn.functional.softplus(s_raw).reshape(len(act), -1).mean(dim=1) + 1e-3
    r = act_pred - act                                   # (B, H, D)
    if mode == "stepdim":
        rms_d = (r.detach() ** 2).mean(dim=0).sqrt() + 1e-6    # (H, D)
    else:
        rms_d = (r.detach() ** 2).mean(dim=(0, 1)).sqrt() + 1e-6   # (D,)
    if "c" not in _HGD_STATE or _HGD_STATE["c"].shape != rms_d.shape:
        _HGD_STATE["c"] = rms_d
    else:
        _HGD_STATE["c"] = ema * _HGD_STATE["c"] + (1 - ema) * rms_d
    c = _HGD_STATE["c"]
    c = c / torch.exp(torch.log(c).mean())               # geo-mean 1
    n = r[0].numel()
    cw = (c ** 2).reshape(1, 1, -1) if mode == "dim" else (c ** 2)[None]
    r2w = (r ** 2 / cw).sum(dim=(1, 2))
    per = 0.5 * r2w / (sigma ** 2) + n * torch.log(sigma)
    return config.loss_scale * (torch.mean(per) / n), {}


def regression_hetero_t_diag_loss(
    config: OptimizationConfig,
    flow_map: FlowMap,
    encoder: BaseEncoder,
    interp: Interpolant,
    act: torch.Tensor,
    obs: torch.Tensor,
    delta_t: torch.Tensor,
) -> float:
    """Anisotropic hetero-t: Student-t NLL with the EMA per-dim (or
    per-step-dim) noise profile of regression_hetero_diag. Env: HGD_EMA,
    HGD_MODE; df via optimization.student_t_df."""
    import os as _os

    ema = float(_os.environ.get("HGD_EMA", "0.99"))
    mode = _os.environ.get("HGD_MODE", "dim")
    t = torch.zeros_like(delta_t, device=delta_t.device)
    act_0 = torch.zeros_like(act, device=act.device)
    obs_emb = encoder(obs, None)
    act_pred, s_raw = flow_map.net(act_0, t, t, obs_emb)
    sigma = torch.nn.functional.softplus(s_raw).reshape(len(act), -1).mean(dim=1) + 1e-3
    r = act_pred - act
    if mode == "stepdim":
        rms_d = (r.detach() ** 2).mean(dim=0).sqrt() + 1e-6
    else:
        rms_d = (r.detach() ** 2).mean(dim=(0, 1)).sqrt() + 1e-6
    key = "ct"
    if key not in _HGD_STATE or _HGD_STATE[key].shape != rms_d.shape:
        _HGD_STATE[key] = rms_d
    else:
        _HGD_STATE[key] = ema * _HGD_STATE[key] + (1 - ema) * rms_d
    c = _HGD_STATE[key]
    c = c / torch.exp(torch.log(c).mean())
    n = r[0].numel()
    cw = (c ** 2).reshape(1, 1, -1) if mode == "dim" else (c ** 2)[None]
    r2w = (r ** 2 / cw).sum(dim=(1, 2))
    nu = config.student_t_df
    per = 0.5 * (nu + 1.0) * torch.log1p(r2w / (nu * sigma ** 2 * n)) * n \
        + n * torch.log(sigma)
    return config.loss_scale * (torch.mean(per) / n), {}


def regression_hetero_gauss_cnd_loss(
    config: OptimizationConfig,
    flow_map: FlowMap,
    encoder: BaseEncoder,
    interp: Interpolant,
    act: torch.Tensor,
    obs: torch.Tensor,
    delta_t: torch.Tensor,
) -> float:
    """COMPOSITION ARM: hetero-Gauss NLL (residual-adaptive discounting)
    + condreg penalty (directional-gain conditioning) — tests channel
    additivity (HG +24 and condreg +14/19 from disjoint mechanisms)."""
    import os as _os

    lam = float(_os.environ.get("CONDREG_LAM", "3e-4"))
    t = torch.zeros_like(delta_t, device=delta_t.device)
    act_0 = torch.zeros_like(act, device=act.device)
    obs_emb = encoder(obs, None)
    act_pred, s_raw = flow_map.net(act_0, t, t, obs_emb)
    sigma = torch.nn.functional.softplus(s_raw).reshape(len(act), -1).mean(dim=1) + 1e-3
    r2 = (act_pred - act) ** 2
    n = r2[0].numel()
    per = r2.sum(dim=tuple(range(1, r2.dim()))) / (2 * sigma ** 2)         + n * torch.log(sigma)
    nll = torch.mean(per) / n
    pen = _condreg_penalty(flow_map, encoder, act, obs, t, act_0, act_pred)
    return config.loss_scale * (nll + lam * pen), {}


def regression_hetero_t_cnd_loss(
    config: OptimizationConfig,
    flow_map: FlowMap,
    encoder: BaseEncoder,
    interp: Interpolant,
    act: torch.Tensor,
    obs: torch.Tensor,
    delta_t: torch.Tensor,
) -> float:
    """COMPOSITION ARM: hetero-t NLL + condreg penalty."""
    import os as _os

    lam = float(_os.environ.get("CONDREG_LAM", "3e-4"))
    t = torch.zeros_like(delta_t, device=delta_t.device)
    act_0 = torch.zeros_like(act, device=act.device)
    obs_emb = encoder(obs, None)
    act_pred, s_raw = flow_map.net(act_0, t, t, obs_emb)
    sigma = torch.nn.functional.softplus(s_raw).reshape(len(act), -1).mean(dim=1) + 1e-3
    nu = config.student_t_df
    r2 = (act_pred - act) ** 2
    n = r2[0].numel()
    per = 0.5 * (nu + 1.0) * torch.log1p(
        r2.sum(dim=tuple(range(1, r2.dim()))) / (nu * sigma ** 2 * n)
    ) * n + n * torch.log(sigma)
    nll = torch.mean(per) / n
    pen = _condreg_penalty(flow_map, encoder, act, obs, t, act_0, act_pred)
    return config.loss_scale * (nll + lam * pen), {}


def regression_dcr_loss(
    config: OptimizationConfig,
    flow_map: FlowMap,
    encoder: BaseEncoder,
    interp: Interpolant,
    act: torch.Tensor,
    obs: torch.Tensor,
    delta_t: torch.Tensor,
) -> float:
    """COHERENT-DRIFT-RESPONSE penalty (training-side causal test of the
    transport channel): MSE + lam * E_u || mean_t [f(x + delta*u) - f(x)]_t ||^2.
    Penalizes only the CHUNK-COHERENT (DC) component of the response to an
    off-support displacement — the causally lethal error mode per the
    ACT_SBIAS/ACT_NOISE injections — leaving incoherent response and overall
    gain free. Env: DCR_LAM (1e-3), DCR_DELTA (0.3), DCR_K (4)."""
    import os as _os

    lam = float(_os.environ.get("DCR_LAM", "1e-3"))
    delta = float(_os.environ.get("DCR_DELTA", "0.3"))
    K = int(_os.environ.get("DCR_K", "4"))
    t = torch.zeros_like(delta_t, device=delta_t.device)
    act_0 = torch.zeros_like(act, device=act.device)
    obs_t = obs["state"] if isinstance(obs, dict) else obs
    emb = encoder({"state": obs_t} if isinstance(obs, dict) else obs_t, None)
    pred, _ = flow_map.net(act_0, t, t, emb)
    mse = ((pred - act) ** 2).mean()
    pens = []
    for _k in range(K):
        u = torch.randn_like(obs_t)
        u = u / (u.reshape(len(act), -1).norm(dim=1)
                 .reshape(-1, *([1] * (u.dim() - 1))) + 1e-9)
        emb_p = encoder({"state": obs_t + delta * u}, None)
        pred_p, _ = flow_map.net(act_0, t, t, emb_p)
        resp = pred_p - pred                     # (B, H, D)
        dc = resp.mean(dim=1)                    # (B, D) chunk-coherent part
        pens.append((dc ** 2).sum(dim=1))
    pen = torch.stack(pens, 1).mean()
    return config.loss_scale * (mse + lam * pen), {}


def regression_condnum_loss(
    config: OptimizationConfig,
    flow_map: FlowMap,
    encoder: BaseEncoder,
    interp: Interpolant,
    act: torch.Tensor,
    obs: torch.Tensor,
    delta_t: torch.Tensor,
) -> float:
    """CONDITION-NUMBER face: plain MSE + hinge on the log top/median ratio
    of the K-probe FD gain spectrum, log(d_max/d_med) — bounds the local
    condition-number proxy without touching overall gain or spread shape.
    Env: CN_LAM (3e-3), CN_TAU (log-ratio target, default log 4), CN_EPS
    (0.05), CN_K (6)."""
    import math as _math
    import os as _os

    lam = float(_os.environ.get("CN_LAM", "3e-3"))
    tau = float(_os.environ.get("CN_TAU", str(_math.log(4.0))))
    eps = float(_os.environ.get("CN_EPS", "0.05"))
    K = int(_os.environ.get("CN_K", "6"))
    t = torch.zeros_like(delta_t, device=delta_t.device)
    act_0 = torch.zeros_like(act, device=act.device)
    obs_t = obs["state"] if isinstance(obs, dict) else obs
    emb = encoder({"state": obs_t} if isinstance(obs, dict) else obs_t, None)
    pred, _ = flow_map.net(act_0, t, t, emb)
    mse = ((pred - act) ** 2).mean()
    d2s = []
    for _k in range(K):
        v = torch.randn_like(obs_t)
        v = v / (v.reshape(len(act), -1).norm(dim=1)
                 .reshape(-1, *([1] * (v.dim() - 1))) + 1e-9)
        emb_p = encoder({"state": obs_t + eps * v}, None)
        pred_p, _ = flow_map.net(act_0, t, t, emb_p)
        d2s.append(((pred_p - pred) ** 2).reshape(len(act), -1).mean(dim=1))
    D = torch.stack(d2s, 1)
    logratio = torch.log(D.max(dim=1).values + 1e-12) - torch.log(
        D.median(dim=1).values + 1e-12)
    pen = torch.relu(logratio - tau).mean()
    loss = config.loss_scale * (mse + lam * pen)
    return loss, {"condnum/mse": float(mse.detach()),
                  "condnum/logratio": float(logratio.mean().detach())}


def regression_condann_loss(
    config: OptimizationConfig,
    flow_map: FlowMap,
    encoder: BaseEncoder,
    interp: Interpolant,
    act: torch.Tensor,
    obs: torch.Tensor,
    delta_t: torch.Tensor,
) -> float:
    """ANNULUS-LOCATED spectral shaping: plain MSE + the condreg CV^2 hinge
    evaluated at an annulus-displaced base point obs + delta*w (w random
    unit, delta ~ U(CA_DLO, CA_DHI) in normalized obs units) — shapes the
    Jacobian spectrum where rollout excursions actually live instead of on
    the support. Env: CA_LAM (3e-3), CA_TAU (1.0), CA_EPS (0.05), CA_K (6),
    CA_DLO (0.1), CA_DHI (0.6)."""
    import os as _os

    lam = float(_os.environ.get("CA_LAM", "3e-3"))
    tau = float(_os.environ.get("CA_TAU", "1.0"))
    eps = float(_os.environ.get("CA_EPS", "0.05"))
    K = int(_os.environ.get("CA_K", "6"))
    dlo = float(_os.environ.get("CA_DLO", "0.1"))
    dhi = float(_os.environ.get("CA_DHI", "0.6"))
    t = torch.zeros_like(delta_t, device=delta_t.device)
    act_0 = torch.zeros_like(act, device=act.device)
    obs_t = obs["state"] if isinstance(obs, dict) else obs
    emb = encoder({"state": obs_t} if isinstance(obs, dict) else obs_t, None)
    pred, _ = flow_map.net(act_0, t, t, emb)
    mse = ((pred - act) ** 2).mean()
    w = torch.randn_like(obs_t)
    w = w / (w.reshape(len(act), -1).norm(dim=1)
             .reshape(-1, *([1] * (w.dim() - 1))) + 1e-9)
    dmag = torch.empty((len(act),) + (1,) * (obs_t.dim() - 1),
                       device=obs_t.device).uniform_(dlo, dhi)
    base = obs_t + dmag * w
    emb_b = encoder({"state": base}, None)
    pred_b, _ = flow_map.net(act_0, t, t, emb_b)
    d2s = []
    for _k in range(K):
        v = torch.randn_like(obs_t)
        v = v / (v.reshape(len(act), -1).norm(dim=1)
                 .reshape(-1, *([1] * (v.dim() - 1))) + 1e-9)
        emb_p = encoder({"state": base + eps * v}, None)
        pred_p, _ = flow_map.net(act_0, t, t, emb_p)
        d2s.append(((pred_p - pred_b) ** 2).reshape(len(act), -1).mean(dim=1))
    D = torch.stack(d2s, 1)
    cv2 = D.var(dim=1) / (D.mean(dim=1) ** 2 + 1e-12)
    pen = torch.relu(cv2 - tau).mean()
    loss = config.loss_scale * (mse + lam * pen)
    return loss, {"condann/mse": float(mse.detach()),
                  "condann/cv2": float(cv2.mean().detach())}


def regression_pathdamp_loss(
    config: OptimizationConfig,
    flow_map: FlowMap,
    encoder: BaseEncoder,
    interp: Interpolant,
    act: torch.Tensor,
    obs: torch.Tensor,
    delta_t: torch.Tensor,
) -> float:
    """PATH-DAMPING: plain MSE + hinge penalizing mid-path directional gain
    exceeding endpoint gains on within-batch nearest-pair interpolations —
    targets the mid-path sensitivity bulge (INTJ probe: L2 svmax mid 2.0-2.3
    vs MIP-2step 0.95). Env: PD_LAM (3e-3), PD_EPS (0.05)."""
    import os as _os

    lam = float(_os.environ.get("PD_LAM", "3e-3"))
    eps = float(_os.environ.get("PD_EPS", "0.05"))
    t = torch.zeros_like(delta_t, device=delta_t.device)
    act_0 = torch.zeros_like(act, device=act.device)
    obs_t = obs["state"] if isinstance(obs, dict) else obs
    emb = encoder({"state": obs_t} if isinstance(obs, dict) else obs_t, None)
    pred, _ = flow_map.net(act_0, t, t, emb)
    mse = ((pred - act) ** 2).mean()
    B = len(act)
    flat = obs_t.reshape(B, -1)
    with torch.no_grad():
        Dm = torch.cdist(flat, flat)
        Dm.fill_diagonal_(1e9)
        jidx = Dm.argmin(dim=1)
    xj = obs_t[jidx]
    lmb = torch.rand((B,) + (1,) * (obs_t.dim() - 1), device=obs_t.device)
    lmb = 0.2 + 0.6 * lmb
    xm = obs_t + lmb * (xj - obs_t)
    v = torch.randn_like(obs_t)
    v = v / (v.reshape(B, -1).norm(dim=1)
             .reshape(-1, *([1] * (v.dim() - 1))) + 1e-9)

    def g_at(x):
        e0 = encoder({"state": x}, None)
        p0, _ = flow_map.net(act_0, t, t, e0)
        e1 = encoder({"state": x + eps * v}, None)
        p1, _ = flow_map.net(act_0, t, t, e1)
        return ((p1 - p0) ** 2).reshape(B, -1).mean(dim=1)

    g_i = g_at(obs_t)
    g_j = g_at(xj)
    g_m = g_at(xm)
    pen = torch.relu(g_m - torch.maximum(g_i, g_j)).mean()
    loss = config.loss_scale * (mse + lam * pen)
    return loss, {"pathdamp/mse": float(mse.detach()),
                  "pathdamp/pen": float(pen.detach())}


_FOCAL_NONE = None


def regression_focal_loss(
    config: OptimizationConfig,
    flow_map: FlowMap,
    encoder: BaseEncoder,
    interp: Interpolant,
    act: torch.Tensor,
    obs: torch.Tensor,
    delta_t: torch.Tensor,
) -> float:
    """CONCURRENT-FIT CURRICULUM: per-batch difficulty-proportional weights
    w_i ~ l_i^alpha (normalized) — forces pockets to be fit concurrently
    with the bulk (MIP's measured mid-training profile) in a single-view
    objective. Env: FOCAL_A (0.5)."""
    import os as _os

    alpha = float(_os.environ.get("FOCAL_A", "0.5"))
    t = torch.zeros_like(delta_t, device=delta_t.device)
    act_0 = torch.zeros_like(act, device=act.device)
    obs_emb = encoder(obs, None)
    pred = flow_map.get_velocity(t, act_0, obs_emb)
    li = ((pred - act) ** 2).reshape(len(act), -1).mean(dim=1)
    with torch.no_grad():
        w = (li + 1e-12) ** alpha
        w = w * len(li) / w.sum()
    loss = config.loss_scale * (w * li).mean()
    return loss, {"focal/wmax": float(w.max().detach())}


_FDIST_STATE = {}


def regression_fdistill_loss(
    config: OptimizationConfig,
    flow_map: FlowMap,
    encoder: BaseEncoder,
    interp: Interpolant,
    act: torch.Tensor,
    obs: torch.Tensor,
    delta_t: torch.Tensor,
) -> float:
    """FUNCTION-SPACE DISTILLATION: single-view student = MSE on data +
    lam * MSE to a frozen two-step teacher's outputs at jittered states
    (delta ~ U(0, FD_DHI) normalized, random direction). Transfers the
    teacher's off-support function while keeping single-pass deployment.
    Env: FD_CKPT (teacher), FD_LAM (1.0), FD_DHI (0.4), FD_TTS teacher
    t_two_step (default config.t_two_step)."""
    import copy as _copy
    import os as _os

    lam = float(_os.environ.get("FD_LAM", "1.0"))
    dhi = float(_os.environ.get("FD_DHI", "0.4"))
    if "fm" not in _FDIST_STATE:
        sd = torch.load(_os.environ["FD_CKPT"], map_location=act.device,
                        weights_only=False)
        tfm = _copy.deepcopy(flow_map)
        ten = _copy.deepcopy(encoder)
        tfm.load_state_dict(sd["flow_map_ema"])
        ten.load_state_dict(sd["encoder_ema"])
        for p in list(tfm.parameters()) + list(ten.parameters()):
            p.requires_grad_(False)
        tfm.eval()
        ten.eval()
        _FDIST_STATE["fm"], _FDIST_STATE["en"] = tfm, ten
    tfm, ten = _FDIST_STATE["fm"], _FDIST_STATE["en"]
    t = torch.zeros_like(delta_t, device=delta_t.device)
    act_0 = torch.zeros_like(act, device=act.device)
    obs_t = obs["state"] if isinstance(obs, dict) else obs
    emb = encoder({"state": obs_t} if isinstance(obs, dict) else obs_t, None)
    pred = flow_map.get_velocity(t, act_0, emb)
    mse = ((pred - act) ** 2).mean()
    u = torch.randn_like(obs_t)
    u = u / (u.reshape(len(act), -1).norm(dim=1)
             .reshape(-1, *([1] * (u.dim() - 1))) + 1e-9)
    dmag = torch.rand((len(act),) + (1,) * (obs_t.dim() - 1),
                      device=obs_t.device) * dhi
    xj = obs_t + dmag * u
    with torch.no_grad():
        embt = ten({"state": xj}, None)
        s_ = torch.zeros_like(delta_t)
        tt = torch.zeros_like(delta_t) + float(
            _os.environ.get("FD_TTS", str(config.t_two_step)))
        a1 = tfm.get_velocity(s_, act_0, embt)
        ty = tfm.get_velocity(tt, a1, embt)
    embj = encoder({"state": xj}, None)
    predj = flow_map.get_velocity(t, act_0, embj)
    dl = ((predj - ty) ** 2).mean()
    loss = config.loss_scale * (mse + lam * dl)
    return loss, {"fdist/mse": float(mse.detach()),
                  "fdist/dl": float(dl.detach())}


def _jspec_penalty(flow_map, encoder, act, obs_t, t, act_0):
    """Generalized Jacobian-spectrum penalty from K FD probes.
    Env: JS_STAT cv2|cond|gmax, JS_LOC on|mid, JS_TAU, JS_EPS (.05), JS_K (6).
    cv2: relu(CV^2(g^2)-tau) [effective-rank hinge]
    cond: relu(log(gmax/gmed)-log(tau)) [condition-number hinge]
    gmax: relu(gmax-tau) [bulge cap on top directional gain]"""
    import math as _math
    import os as _os

    stat = _os.environ.get("JS_STAT", "cv2")
    loc = _os.environ.get("JS_LOC", "mid")
    tau = float(_os.environ.get("JS_TAU", "1.0"))
    eps = float(_os.environ.get("JS_EPS", "0.05"))
    K = int(_os.environ.get("JS_K", "6"))
    nbr = int(_os.environ.get("JS_NBR", "1"))
    span = _os.environ.get("JS_SPAN", "0") == "1"
    disp = float(_os.environ.get("JS_DISP", "0"))
    B = len(act)
    if loc == "randpair":
        jidx = torch.randperm(B, device=obs_t.device)
        base = 0.5 * (obs_t + obs_t[jidx])
    elif loc == "box":
        lo = obs_t.reshape(B, -1).min(0).values
        hi = obs_t.reshape(B, -1).max(0).values
        base = (lo + torch.rand((B, lo.numel()), device=obs_t.device) *
                (hi - lo)).reshape(obs_t.shape)
    elif loc == "mid":
        flat = obs_t.reshape(B, -1)
        with torch.no_grad():
            Dm = torch.cdist(flat, flat)
            Dm.fill_diagonal_(1e9)
            if nbr <= 1:
                jidx = Dm.argmin(dim=1)
            else:
                jidx = Dm.topk(nbr, dim=1, largest=False).indices[:, nbr - 1]
        if span:
            lmb = (0.2 + 0.6 * torch.rand((B,) + (1,) * (obs_t.dim() - 1),
                                          device=obs_t.device))
            base = obs_t + lmb * (obs_t[jidx] - obs_t)
        else:
            base = 0.5 * (obs_t + obs_t[jidx])
    else:
        base = obs_t
    if disp > 0:
        w = torch.randn_like(obs_t)
        w = w / (w.reshape(B, -1).norm(dim=1)
                 .reshape(-1, *([1] * (w.dim() - 1))) + 1e-9)
        base = base + disp * w
    emb_b = encoder({"state": base}, None)
    pred_b, _ = flow_map.net(act_0, t, t, emb_b)
    d2s = []
    for _k in range(K):
        v = torch.randn_like(obs_t)
        v = v / (v.reshape(B, -1).norm(dim=1)
                 .reshape(-1, *([1] * (v.dim() - 1))) + 1e-9)
        emb_p = encoder({"state": base + eps * v}, None)
        pred_p, _ = flow_map.net(act_0, t, t, emb_p)
        d2s.append(((pred_p - pred_b) ** 2).reshape(B, -1).mean(dim=1))
    D = torch.stack(d2s, 1)
    if stat == "cv2":
        cv2 = D.var(dim=1) / (D.mean(dim=1) ** 2 + 1e-12)
        return torch.relu(cv2 - tau).mean()
    g = torch.sqrt(D + 1e-12) / eps
    if stat == "cond":
        lr_ = torch.log(g.max(dim=1).values + 1e-9) - torch.log(
            g.median(dim=1).values + 1e-9)
        return torch.relu(lr_ - _math.log(tau)).mean()
    if stat == "condmin":
        lr_ = torch.log(g.max(dim=1).values + 1e-9) - torch.log(
            g.min(dim=1).values + 1e-9)
        return torch.relu(lr_ - _math.log(tau)).mean()
    if stat == "gap":
        top2 = g.topk(2, dim=1).values
        return torch.relu((top2[:, 0] - top2[:, 1]) /
                          (top2[:, 0] + 1e-9) - tau).mean()
    if stat == "gmedfloor":
        return torch.relu(tau - g.median(dim=1).values).mean()
    if stat == "logvar":
        return torch.relu(torch.log(g + 1e-9).var(dim=1) - tau).mean()
    if stat == "range":
        return torch.relu((g.max(dim=1).values - g.min(dim=1).values) /
                          (g.mean(dim=1) + 1e-9) - tau).mean()
    if stat == "cv2gmax":
        cv2b = D.var(dim=1) / (D.mean(dim=1) ** 2 + 1e-12)
        cap = float(_os.environ.get("JS_TAU2", "1.5"))
        return (torch.relu(cv2b - tau) +
                torch.relu(g.max(dim=1).values - cap)).mean()
    return torch.relu(g.max(dim=1).values - tau).mean()


def _nn_mid(obs_t):
    B = len(obs_t)
    flat = obs_t.reshape(B, -1)
    with torch.no_grad():
        Dm = torch.cdist(flat, flat)
        Dm.fill_diagonal_(1e9)
        jidx = Dm.argmin(dim=1)
    return jidx, 0.5 * (obs_t + obs_t[jidx])


def regression_hetero_t_mixnn_loss(config, flow_map, encoder, interp, act,
                                   obs, delta_t):
    """Nearest-pair label-secant (twin-restricted mixup): base +
    MX_LAM * ||f(mid) - (y_i+y_j)/2||^2. Env: MX_LAM (0.5)."""
    import os as _os

    base, info = regression_hetero_t_loss(config, flow_map, encoder, interp,
                                          act, obs, delta_t)
    lam = float(_os.environ.get("MX_LAM", "0.5"))
    t = torch.zeros_like(delta_t, device=delta_t.device)
    act_0 = torch.zeros_like(act, device=act.device)
    obs_t = obs["state"] if isinstance(obs, dict) else obs
    jidx, xm = _nn_mid(obs_t)
    emb_m = encoder({"state": xm}, None)
    pred_m, _ = flow_map.net(act_0, t, t, emb_m)
    ym = 0.5 * (act + act[jidx])
    pen = ((pred_m - ym) ** 2).mean()
    return base + config.loss_scale * lam * pen, info


def regression_hetero_t_midlin_loss(config, flow_map, encoder, interp, act,
                                    obs, delta_t):
    """Function-space convexity between twins (no labels): base +
    MX_LAM * ||f(mid) - (f(x_i)+f(x_j))/2||^2. Env: MX_LAM (0.5)."""
    import os as _os

    base, info = regression_hetero_t_loss(config, flow_map, encoder, interp,
                                          act, obs, delta_t)
    lam = float(_os.environ.get("MX_LAM", "0.5"))
    t = torch.zeros_like(delta_t, device=delta_t.device)
    act_0 = torch.zeros_like(act, device=act.device)
    obs_t = obs["state"] if isinstance(obs, dict) else obs
    jidx, xm = _nn_mid(obs_t)
    emb_all = encoder({"state": torch.cat([xm, obs_t])}, None)
    pred_all, _ = flow_map.net(act_0.repeat(2, 1, 1),
                               t.repeat(2), t.repeat(2), emb_all)
    B = len(act)
    pred_m, pred_e = pred_all[:B], pred_all[B:]
    pen = ((pred_m - 0.5 * (pred_e + pred_e[jidx])) ** 2).mean()
    return base + config.loss_scale * lam * pen, info


def regression_hetero_t_jitcons_loss(config, flow_map, encoder, interp, act,
                                     obs, delta_t):
    """Local consistency: base + MX_LAM * ||f(x+eps u) - sg(f(x))||^2.
    Env: MX_LAM (0.5), MX_EPS (0.05)."""
    import os as _os

    base, info = regression_hetero_t_loss(config, flow_map, encoder, interp,
                                          act, obs, delta_t)
    lam = float(_os.environ.get("MX_LAM", "0.5"))
    eps = float(_os.environ.get("MX_EPS", "0.05"))
    t = torch.zeros_like(delta_t, device=delta_t.device)
    act_0 = torch.zeros_like(act, device=act.device)
    obs_t = obs["state"] if isinstance(obs, dict) else obs
    u = torch.randn_like(obs_t)
    u = u / (u.reshape(len(act), -1).norm(dim=1)
             .reshape(-1, *([1] * (u.dim() - 1))) + 1e-9)
    emb0 = encoder({"state": obs_t}, None)
    p0, _ = flow_map.net(act_0, t, t, emb0)
    embj = encoder({"state": obs_t + eps * u}, None)
    pj, _ = flow_map.net(act_0, t, t, embj)
    pen = ((pj - p0.detach()) ** 2).mean()
    return base + config.loss_scale * lam * pen, info


def regression_hetero_gauss_midlin_loss(config, flow_map, encoder, interp,
                                        act, obs, delta_t):
    """HG base + midpoint convexity."""
    import os as _os

    base, info = regression_hetero_gauss_loss(config, flow_map, encoder,
                                              interp, act, obs, delta_t)
    lam = float(_os.environ.get("MX_LAM", "0.5"))
    t = torch.zeros_like(delta_t, device=delta_t.device)
    act_0 = torch.zeros_like(act, device=act.device)
    obs_t = obs["state"] if isinstance(obs, dict) else obs
    jidx, xm = _nn_mid(obs_t)
    emb_all = encoder({"state": torch.cat([xm, obs_t])}, None)
    pred_all, _ = flow_map.net(act_0.repeat(2, 1, 1),
                               t.repeat(2), t.repeat(2), emb_all)
    B = len(act)
    pred_m, pred_e = pred_all[:B], pred_all[B:]
    pen = ((pred_m - 0.5 * (pred_e + pred_e[jidx])) ** 2).mean()
    return base + config.loss_scale * lam * pen, info


def regression_jspec_loss(config, flow_map, encoder, interp, act, obs,
                          delta_t):
    """Plain MSE + generalized J-spectrum penalty (same envs as hetero
    versions: JS_STAT/JS_LOC/JS_LAM/JS_TAU/...)."""
    import os as _os

    t = torch.zeros_like(delta_t, device=delta_t.device)
    act_0 = torch.zeros_like(act, device=act.device)
    obs_t = obs["state"] if isinstance(obs, dict) else obs
    emb = encoder({"state": obs_t} if isinstance(obs, dict) else obs_t, None)
    pred, _ = flow_map.net(act_0, t, t, emb)
    mse = ((pred - act) ** 2).mean()
    lam = float(_os.environ.get("JS_LAM", "3e-3"))
    pen = _jspec_penalty(flow_map, encoder, act, obs_t, t, act_0)
    return config.loss_scale * (mse + lam * pen), {}


def regression_hetero_t_jspec_loss(config, flow_map, encoder, interp, act,
                                   obs, delta_t):
    """hetero-t + generalized J-spectrum penalty. Env: JS_LAM (3e-3) + _jspec."""
    import os as _os

    base, info = regression_hetero_t_loss(config, flow_map, encoder, interp,
                                          act, obs, delta_t)
    lam = float(_os.environ.get("JS_LAM", "3e-3"))
    t = torch.zeros_like(delta_t, device=delta_t.device)
    act_0 = torch.zeros_like(act, device=act.device)
    obs_t = obs["state"] if isinstance(obs, dict) else obs
    pen = _jspec_penalty(flow_map, encoder, act, obs_t, t, act_0)
    return base + config.loss_scale * lam * pen, info


def regression_hetero_gauss_jspec_loss(config, flow_map, encoder, interp,
                                       act, obs, delta_t):
    """hetero-gauss + generalized J-spectrum penalty."""
    import os as _os

    base, info = regression_hetero_gauss_loss(config, flow_map, encoder,
                                              interp, act, obs, delta_t)
    lam = float(_os.environ.get("JS_LAM", "3e-3"))
    t = torch.zeros_like(delta_t, device=delta_t.device)
    act_0 = torch.zeros_like(act, device=act.device)
    obs_t = obs["state"] if isinstance(obs, dict) else obs
    pen = _jspec_penalty(flow_map, encoder, act, obs_t, t, act_0)
    return base + config.loss_scale * lam * pen, info


def _midrank_penalty(flow_map, encoder, act, obs_t, t, act_0):
    """CV^2 hinge on FD gain spectrum at within-batch nearest-pair MIDPOINTS
    — targets the measured mid-path rank-1 collapse (PART CDXVIII: L2/HG PR
    1.12-1.16, svmax 2.4-3.0 between near-twins vs MIP 1.56/1.31).
    Env: RM_TAU (1.0), RM_EPS (0.05), RM_K (6)."""
    import os as _os

    tau = float(_os.environ.get("RM_TAU", "1.0"))
    eps = float(_os.environ.get("RM_EPS", "0.05"))
    K = int(_os.environ.get("RM_K", "6"))
    B = len(act)
    flat = obs_t.reshape(B, -1)
    with torch.no_grad():
        Dm = torch.cdist(flat, flat)
        Dm.fill_diagonal_(1e9)
        jidx = Dm.argmin(dim=1)
    xm = 0.5 * (obs_t + obs_t[jidx])
    emb_m = encoder({"state": xm}, None)
    pred_m, _ = flow_map.net(act_0, t, t, emb_m)
    d2s = []
    for _k in range(K):
        v = torch.randn_like(obs_t)
        v = v / (v.reshape(B, -1).norm(dim=1)
                 .reshape(-1, *([1] * (v.dim() - 1))) + 1e-9)
        emb_p = encoder({"state": xm + eps * v}, None)
        pred_p, _ = flow_map.net(act_0, t, t, emb_p)
        d2s.append(((pred_p - pred_m) ** 2).reshape(B, -1).mean(dim=1))
    D = torch.stack(d2s, 1)
    cv2 = D.var(dim=1) / (D.mean(dim=1) ** 2 + 1e-12)
    return torch.relu(cv2 - tau).mean()


def regression_hetero_t_rankmid_loss(config, flow_map, encoder, interp, act,
                                     obs, delta_t):
    """hetero-t + midpoint effective-rank hinge. Env: RM_LAM (3e-3)."""
    import os as _os

    base, info = regression_hetero_t_loss(config, flow_map, encoder, interp,
                                          act, obs, delta_t)
    lam = float(_os.environ.get("RM_LAM", "3e-3"))
    t = torch.zeros_like(delta_t, device=delta_t.device)
    act_0 = torch.zeros_like(act, device=act.device)
    obs_t = obs["state"] if isinstance(obs, dict) else obs
    pen = _midrank_penalty(flow_map, encoder, act, obs_t, t, act_0)
    return base + config.loss_scale * lam * pen, info


def regression_hetero_gauss_rankmid_loss(config, flow_map, encoder, interp,
                                         act, obs, delta_t):
    """hetero-gauss + midpoint effective-rank hinge. Env: RM_LAM (3e-3)."""
    import os as _os

    base, info = regression_hetero_gauss_loss(config, flow_map, encoder,
                                              interp, act, obs, delta_t)
    lam = float(_os.environ.get("RM_LAM", "3e-3"))
    t = torch.zeros_like(delta_t, device=delta_t.device)
    act_0 = torch.zeros_like(act, device=act.device)
    obs_t = obs["state"] if isinstance(obs, dict) else obs
    pen = _midrank_penalty(flow_map, encoder, act, obs_t, t, act_0)
    return base + config.loss_scale * lam * pen, info


def regression_condreg_loss(
    config: OptimizationConfig,
    flow_map: FlowMap,
    encoder: BaseEncoder,
    interp: Interpolant,
    act: torch.Tensor,
    obs: torch.Tensor,
    delta_t: torch.Tensor,
) -> float:
    """PART CCCLV v2: plain MSE + BOUNDED, TARGETED spectral-flatness penalty
    on the obs-Jacobian. For K random obs directions v_i, d_i =
    ||f(obs+eps v_i)-f(obs)||^2 (finite difference). For a map of effective
    rank r these are ~chi2(r), so CV^2(d) ~ 2/r: CV^2=2 means rank~1, CV^2=1
    means rank~2. Penalty = lam * relu(CV^2 - TAU): pressure STOPS once the
    target effective rank is reached, so (unlike v1 lam=1.0, which destroyed
    the fit) it cannot dominate indefinitely. Raises PR / lowers condition
    number without bounding overall gain -> discriminates the spectrum-shape
    claim from gain-magnitude control.
    Env: CONDREG_LAM (3e-3), CONDREG_TAU (1.0 ~ rank 2), CONDREG_EPS (0.05),
    CONDREG_K (6)."""
    import os as _os

    lam = float(_os.environ.get("CONDREG_LAM", "3e-3"))
    tau = float(_os.environ.get("CONDREG_TAU", "1.0"))
    eps = float(_os.environ.get("CONDREG_EPS", "0.05"))
    K = int(_os.environ.get("CONDREG_K", "6"))
    t = torch.zeros_like(delta_t, device=delta_t.device)
    act_0 = torch.zeros_like(act, device=act.device)
    obs_t = obs["state"] if isinstance(obs, dict) else obs
    emb = encoder({"state": obs_t} if isinstance(obs, dict) else obs_t, None)
    pred, _ = flow_map.net(act_0, t, t, emb)
    mse = ((pred - act) ** 2).mean()
    d2s = []
    for _k in range(K):
        v = torch.randn_like(obs_t)
        v = v / (v.reshape(len(act), -1).norm(dim=1)
                 .reshape(-1, *([1] * (v.dim() - 1))) + 1e-9)
        emb_p = encoder({"state": obs_t + eps * v}, None)
        pred_p, _ = flow_map.net(act_0, t, t, emb_p)
        d2s.append(((pred_p - pred) ** 2).reshape(len(act), -1).mean(dim=1))
    D = torch.stack(d2s, 1)  # (B, K) directional gains^2
    cv2 = D.var(dim=1) / (D.mean(dim=1) ** 2 + 1e-12)
    pen = torch.relu(cv2 - tau).mean()
    loss = config.loss_scale * (mse + lam * pen)
    return loss, {"condreg/mse": float(mse.detach()),
                  "condreg/cv2": float(cv2.mean().detach()),
                  "condreg/pen": float(pen.detach())}


def regression_condreg_axis_loss(
    config: OptimizationConfig,
    flow_map: FlowMap,
    encoder: BaseEncoder,
    interp: Interpolant,
    act: torch.Tensor,
    obs: torch.Tensor,
    delta_t: torch.Tensor,
) -> float:
    """Axis-decomposed condreg: which axis of the obs-Jacobian's effective
    rank carries the MP-200 conditioning benefit?

    dim  - probes perturb ONLY the last obs frame (K random within-frame
           directions); CV^2 over probes = dimension-wise anisotropy at
           fixed history.
    hist - M shared within-frame directions, each applied to ONE frame at
           a time; per-direction CV^2 across the To frame-responses =
           history-wise anisotropy (how unevenly the policy relies on
           frames; with To=2 this is current-frame vs velocity reliance).
    Both diagnostics are always logged; CONDREG_AXIS picks which one is
    penalized. Env: CONDREG_AXIS (dim|hist), CONDREG_LAM (3e-3),
    CONDREG_TAU (1.0), CONDREG_TAU_HIST (0.5), CONDREG_EPS (0.05),
    CONDREG_K (6), CONDREG_M (3)."""
    import os as _os

    axis = _os.environ.get("CONDREG_AXIS", "dim")
    lam = float(_os.environ.get("CONDREG_LAM", "3e-3"))
    tau_d = float(_os.environ.get("CONDREG_TAU", "1.0"))
    tau_h = float(_os.environ.get("CONDREG_TAU_HIST", "0.5"))
    eps = float(_os.environ.get("CONDREG_EPS", "0.05"))
    K = int(_os.environ.get("CONDREG_K", "6"))
    M = int(_os.environ.get("CONDREG_M", "3"))
    t = torch.zeros_like(delta_t, device=delta_t.device)
    act_0 = torch.zeros_like(act, device=act.device)
    obs_t = obs["state"] if isinstance(obs, dict) else obs
    b = len(act)
    emb = encoder({"state": obs_t} if isinstance(obs, dict) else obs_t, None)
    pred, _ = flow_map.net(act_0, t, t, emb)
    mse = ((pred - act) ** 2).mean()

    def resp(v):
        emb_p = encoder({"state": obs_t + eps * v}, None)
        pred_p, _ = flow_map.net(act_0, t, t, emb_p)
        return ((pred_p - pred) ** 2).reshape(b, -1).mean(dim=1)

    # dimension-wise probes: random directions confined to the last frame
    d_dim = []
    for _k in range(K):
        v = torch.zeros_like(obs_t)
        r = torch.randn_like(obs_t[:, -1, :])
        r = r / (r.norm(dim=1, keepdim=True) + 1e-9)
        v[:, -1, :] = r
        d_dim.append(resp(v))
    Dd = torch.stack(d_dim, 1)
    cv2_dim = Dd.var(dim=1) / (Dd.mean(dim=1) ** 2 + 1e-12)

    # history-wise probes: one shared direction, moved across frames
    n_frames = obs_t.shape[1]
    cv2_hs = []
    for _m in range(M):
        u = torch.randn_like(obs_t[:, 0, :])
        u = u / (u.norm(dim=1, keepdim=True) + 1e-9)
        ds = []
        for f in range(n_frames):
            v = torch.zeros_like(obs_t)
            v[:, f, :] = u
            ds.append(resp(v))
        Dh = torch.stack(ds, 1)
        cv2_hs.append(Dh.var(dim=1) / (Dh.mean(dim=1) ** 2 + 1e-12))
    cv2_hist = torch.stack(cv2_hs, 1).mean(dim=1)

    if axis == "hist":
        pen = torch.relu(cv2_hist - tau_h).mean()
    else:
        pen = torch.relu(cv2_dim - tau_d).mean()
    loss = config.loss_scale * (mse + lam * pen)
    return loss, {"condreg/mse": float(mse.detach()),
                  "condreg/cv2_dim": float(cv2_dim.mean().detach()),
                  "condreg/cv2_hist": float(cv2_hist.mean().detach()),
                  "condreg/pen": float(pen.detach())}


def regression_selfnorm_loss(
    config: OptimizationConfig,
    flow_map: FlowMap,
    encoder: BaseEncoder,
    interp: Interpolant,
    act: torch.Tensor,
    obs: torch.Tensor,
    delta_t: torch.Tensor,
) -> float:
    """EXACT PURE-WEIGHT MIMIC OF HETERO-GAUSS (PART CCCXXXVIII pre-reg):
    MSE with per-sample weights w = 1/(m + eps), m = detached per-sample MSE,
    eps = SN_EPS (default 1e-6 = hetero_gauss's sigma_min^2). This is the
    hg NLL's mu-gradient at sigma-stationarity (sigma^2 -> E[m|x]) with the
    instantaneous per-sample residual as the sigma estimate: absolute anchor
    (no mean-1 normalization -> no invw starvation), full ~1e4 equalization
    ratio (no WMAX=20 truncation -> no gfloor/selfsw retirement leak),
    release below eps exactly where hg's sigma floor releases."""
    import os as _os

    t = torch.zeros_like(delta_t, device=delta_t.device)
    act_0 = torch.zeros_like(act, device=act.device)
    obs_emb = encoder(obs, None)
    act_pred, _ = flow_map.net(act_0, t, t, obs_emb)
    per = ((act_pred - act) ** 2).reshape(len(act), -1).mean(dim=1)
    eps = float(_os.environ.get("SN_EPS", "1e-6"))
    with torch.no_grad():
        w = 1.0 / (per.detach() + eps)
    loss = config.loss_scale * torch.mean(per * w)
    return loss, {}


def regression_emaret_loss(
    config: OptimizationConfig,
    flow_map: FlowMap,
    encoder: BaseEncoder,
    interp: Interpolant,
    act: torch.Tensor,
    obs: torch.Tensor,
    delta_t: torch.Tensor,
) -> float:
    """ANTI-RETIREMENT CONTROL 2 (PART CCCXXXVI pre-reg): MSE + EMA-consistency
    retention on LOW-residual samples — where labels stop defending fitted
    distinctions, a slow-moving self-teacher anchors them. Defense by
    freezing rather than re-funding (dissociates the two)."""
    import copy as _copy
    import os as _os

    t = torch.zeros_like(delta_t, device=delta_t.device)
    act_0 = torch.zeros_like(act, device=act.device)
    obs_emb = encoder(obs, None)
    act_pred, _ = flow_map.net(act_0, t, t, obs_emb)
    per = ((act_pred - act) ** 2).reshape(len(act), -1).mean(dim=1)
    lam = float(_os.environ.get("ER_LAM", "1.0"))
    rate = float(_os.environ.get("ER_RATE", "0.999"))
    if "tf" not in _EMARET_STATE:
        _EMARET_STATE["tf"] = _copy.deepcopy(flow_map)
        _EMARET_STATE["te"] = _copy.deepcopy(encoder)
        for p in list(_EMARET_STATE["tf"].parameters()) + list(_EMARET_STATE["te"].parameters()):
            p.requires_grad_(False)
        _EMARET_STATE["tau"] = None
    tf, te = _EMARET_STATE["tf"], _EMARET_STATE["te"]
    with torch.no_grad():
        for pt, p in zip(tf.parameters(), flow_map.parameters()):
            pt.mul_(rate).add_((1 - rate) * p.detach())
        for pt, p in zip(te.parameters(), encoder.parameters()):
            pt.mul_(rate).add_((1 - rate) * p.detach())
        temb = te(obs, None)
        tpred, _ = tf.net(act_0, t, t, temb)
        med = per.detach().median()
        _EMARET_STATE["tau"] = (med if _EMARET_STATE["tau"] is None
                                else 0.99 * _EMARET_STATE["tau"] + 0.01 * med)
        low = (per.detach() < _EMARET_STATE["tau"]).float()
    ret = ((act_pred - tpred.detach()) ** 2).reshape(len(act), -1).mean(dim=1)
    loss = config.loss_scale * (torch.mean(per) + lam * torch.mean(low * ret))
    return loss, {}


_SNT_STATE = {}


def regression_snteach_loss(
    config: OptimizationConfig,
    flow_map: FlowMap,
    encoder: BaseEncoder,
    interp: Interpolant,
    act: torch.Tensor,
    obs: torch.Tensor,
    delta_t: torch.Tensor,
) -> float:
    """PART CCCXLI pre-reg: snorm's allocation with a LOW-VARIANCE weight
    estimator and still no NLL. Weights come from the EMA teacher's residual
    field, w = 1/(m_teacher + eps): the teacher's residuals are a slow,
    smoothed estimate of E[m|x] — exactly the role hg's learned sigma(x)
    plays — while the student objective stays plain weighted MSE. Completes
    the decomposition hg = allocation (snorm/gfq: 0.80/0.69) + estimator
    quality. Env: SNT_EPS (1e-6), SNT_RATE (0.999)."""
    import copy as _copy
    import os as _os

    t = torch.zeros_like(delta_t, device=delta_t.device)
    act_0 = torch.zeros_like(act, device=act.device)
    obs_emb = encoder(obs, None)
    act_pred, _ = flow_map.net(act_0, t, t, obs_emb)
    per = ((act_pred - act) ** 2).reshape(len(act), -1).mean(dim=1)
    eps = float(_os.environ.get("SNT_EPS", "1e-6"))
    rate = float(_os.environ.get("SNT_RATE", "0.999"))
    if "tf" not in _SNT_STATE:
        _SNT_STATE["tf"] = _copy.deepcopy(flow_map)
        _SNT_STATE["te"] = _copy.deepcopy(encoder)
        for p in list(_SNT_STATE["tf"].parameters()) + list(_SNT_STATE["te"].parameters()):
            p.requires_grad_(False)
    tf, te = _SNT_STATE["tf"], _SNT_STATE["te"]
    with torch.no_grad():
        for pt, p in zip(tf.parameters(), flow_map.parameters()):
            pt.mul_(rate).add_((1 - rate) * p.detach())
        for pt, p in zip(te.parameters(), encoder.parameters()):
            pt.mul_(rate).add_((1 - rate) * p.detach())
        tpred, _ = tf.net(act_0, t, t, te(obs, None))
        per_t = ((tpred - act) ** 2).reshape(len(act), -1).mean(dim=1)
        w = 1.0 / (per_t + eps)
    loss = config.loss_scale * torch.mean(per * w)
    return loss, {}


def regression_selfsw_loss(
    config: OptimizationConfig,
    flow_map: FlowMap,
    encoder: BaseEncoder,
    interp: Interpolant,
    act: torch.Tensor,
    obs: torch.Tensor,
    delta_t: torch.Tensor,
) -> float:
    """SELF-SIGMA ARM (PART CCCXXIX pre-reg): dynamics WITHOUT the tail.
    Plain MSE on mu, per-sample weights 1/sigma_hat^2 from the model's OWN
    scalar head, which is trained (decoupled, detached targets) to track the
    current per-sample RMS residual. Isolates the adaptive-curriculum
    component of learned sigma from the NLL/t-tail structure: prediction is
    hg-like capability with hg-like volatility if the tail is what
    stabilizes formation; ht-band if dynamics alone suffice."""
    import os as _os

    t = torch.zeros_like(delta_t, device=delta_t.device)
    act_0 = torch.zeros_like(act, device=act.device)
    obs_emb = encoder(obs, None)
    act_pred, s_raw = flow_map.net(act_0, t, t, obs_emb)
    sigma = torch.nn.functional.softplus(s_raw).reshape(len(act), -1).mean(dim=1) + 1e-3
    per = ((act_pred - act) ** 2).reshape(len(act), -1).mean(dim=1)
    _lo = float(_os.environ.get("SSW_LO", "0.02"))
    _hi = float(_os.environ.get("SSW_HI", "20.0"))
    with torch.no_grad():
        w = 1.0 / sigma.detach() ** 2
        w = torch.clamp(w / w.mean(), _lo, _hi)
    sig_fit = (torch.log(sigma) - 0.5 * torch.log(per.detach() + 1e-10)) ** 2
    loss = config.loss_scale * (torch.mean(per * w) + 0.1 * torch.mean(sig_fit))
    return loss, {}


def regression_sigmaw_loss(
    config: OptimizationConfig,
    flow_map: FlowMap,
    encoder: BaseEncoder,
    interp: Interpolant,
    act: torch.Tensor,
    obs: torch.Tensor,
    delta_t: torch.Tensor,
) -> float:
    """CAUSAL TRANSPLANT AT ROBOT SCALE (PART CCCXXIV pre-reg): plain MSE with
    FIXED per-sample weights 1/sigma^2(x) from a FROZEN trained hetero-t
    teacher (checkpoint path via env SIGMAW_TEACHER; EMA weights). Weights are
    running-mean-1 normalized then clipped [0.02, 20], mirroring the toy l2sw
    closure. If this recovers the hetero-t band, sigma-map repricing is
    causally sufficient on the real task; if not, the tail/NLL structure
    carries content beyond the weights."""
    import copy as _copy
    import os as _os

    t = torch.zeros_like(delta_t, device=delta_t.device)
    act_0 = torch.zeros_like(act, device=act.device)
    if "teacher" not in _SIGMAW_STATE:
        path = _os.environ["SIGMAW_TEACHER"]
        ck = torch.load(path, map_location=act.device)
        tf = _copy.deepcopy(flow_map)
        te = _copy.deepcopy(encoder)
        tf.load_state_dict(ck["flow_map_ema"])
        te.load_state_dict(ck["encoder_ema"])
        for p in list(tf.parameters()) + list(te.parameters()):
            p.requires_grad_(False)
        tf.eval()
        te.eval()
        _SIGMAW_STATE["teacher"] = (tf, te)
        _SIGMAW_STATE["mw"] = None
    tf, te = _SIGMAW_STATE["teacher"]
    with torch.no_grad():
        emb = te(obs, None)
        _, s_raw = tf.net(act_0, t, t, emb)
        sigma = torch.nn.functional.softplus(s_raw).reshape(len(act), -1).mean(dim=1) + 1e-3
        w = 1.0 / sigma ** 2
        m = w.mean()
        _SIGMAW_STATE["mw"] = (m if _SIGMAW_STATE["mw"] is None
                               else 0.99 * _SIGMAW_STATE["mw"] + 0.01 * m)
        w = torch.clamp(w / _SIGMAW_STATE["mw"], 0.02, 20.0)
    obs_emb = encoder(obs, None)
    act_pred, _ = flow_map.net(act_0, t, t, obs_emb)
    per = ((act_pred - act) ** 2).reshape(len(act), -1).mean(dim=1)
    loss = config.loss_scale * torch.mean(per * w)
    return loss, {}


def regression_gmm_loss(
    config: OptimizationConfig,
    flow_map: FlowMap,
    encoder: BaseEncoder,
    interp: Interpolant,
    act: torch.Tensor,
    obs: torch.Tensor,
    delta_t: torch.Tensor,
) -> float:
    """MIXTURE-AXIS CONTROL (PART CCCXVIII wave 1b): MDN/GMM NLL, K components
    (network.gmm_k), isotropic per-component sigma. Tests whether mixture
    multimodality adds anything beyond repricing: the suppression story
    predicts GMM lands at/below hetero-t. Eval uses the argmax-weight mean
    (mode), wired in ChiUNet.forward."""
    t = torch.zeros_like(delta_t, device=delta_t.device)
    act_0 = torch.zeros_like(act, device=act.device)
    obs_emb = encoder(obs, None)
    _ = flow_map.net(act_0, t, t, obs_emb)
    means, logits, logsig = flow_map.net._gmm  # (B,K,T,A), (B,K), (B,K)
    K = means.shape[1]
    D = means[0, 0].numel()
    sigma = torch.nn.functional.softplus(logsig) + 1e-3
    r2 = ((means - act.unsqueeze(1)) ** 2).sum(dim=(2, 3))  # (B,K)
    logw = torch.log_softmax(logits, dim=1)
    logp = logw - D * torch.log(sigma) - 0.5 * r2 / sigma ** 2
    loss = -torch.logsumexp(logp, dim=1).mean() / D
    return config.loss_scale * loss, {}


def regression_globalt_loss(
    config: OptimizationConfig,
    flow_map: FlowMap,
    encoder: BaseEncoder,
    interp: Interpolant,
    act: torch.Tensor,
    obs: torch.Tensor,
    delta_t: torch.Tensor,
) -> float:
    """NECESSITY CONTROL A (PART CCCXVIII): Student-t NLL with a LEARNED GLOBAL
    scale — state-INdependent. The scale is the t-MLE global sigma, maintained as
    an EMA of the EM fixed-point update (buffer; equivalent to learning the scalar
    by MLE, robust to the frozen-optimizer-param issue). Isolates whether
    HETEROscedasticity (state-conditional sigma) is necessary, or adaptive global
    scale + bounded tail suffices. Everything else mirrors regression_hetero_t."""
    t = torch.zeros_like(delta_t, device=delta_t.device)
    act_0 = torch.zeros_like(act, device=act.device)
    obs_emb = encoder(obs, None)
    act_pred, _ = flow_map.net(act_0, t, t, obs_emb)
    nu = config.student_t_df
    r2 = (act_pred - act) ** 2
    D = r2[0].numel()
    m = r2.sum(dim=tuple(range(1, r2.dim()))) / D  # per-sample mean-square
    if not hasattr(flow_map, "_gt_sig2"):
        flow_map.register_buffer(
            "_gt_sig2", torch.ones(1, device=act.device) * m.detach().mean())
    sig2 = flow_map._gt_sig2.clamp_min(1e-8)
    with torch.no_grad():
        w = (nu + 1.0) / (nu + m.detach() / sig2)
        flow_map._gt_sig2.mul_(0.99).add_(0.01 * (w * m.detach()).mean())
    sigma = sig2.sqrt()
    per = 0.5 * (nu + 1.0) * torch.log1p(m / (nu * sig2)) * D + D * torch.log(sigma)
    loss = config.loss_scale * torch.mean(per) / D
    return loss, {}


def regression_hgclip_loss(
    config: OptimizationConfig,
    flow_map: FlowMap,
    encoder: BaseEncoder,
    interp: Interpolant,
    act: torch.Tensor,
    obs: torch.Tensor,
    delta_t: torch.Tensor,
) -> float:
    """NECESSITY CONTROL B (PART CCCXVIII): heteroscedastic GAUSSIAN NLL with
    BOUNDED INFLUENCE via Huberization of the normalized squared residual
    z = |r|^2/(sigma^2 D): psi(z) = z for z <= c^2 else 2c sqrt(z) - c^2
    (gradient wrt r capped at c/sigma). Separates 'bounded influence' from the
    t density shape: if this recovers hetero-t's stability, tail = boundedness;
    if it stays hg-volatile, the t shape carries more. c via HGCLIP_C (2.0)."""
    t = torch.zeros_like(delta_t, device=delta_t.device)
    act_0 = torch.zeros_like(act, device=act.device)
    obs_emb = encoder(obs, None)
    act_pred, s_raw = flow_map.net(act_0, t, t, obs_emb)
    sigma = torch.nn.functional.softplus(s_raw).reshape(len(act), -1).mean(dim=1) + 1e-3
    import os as _os
    c = float(_os.environ.get("HGCLIP_C", "2.0"))
    r2 = (act_pred - act) ** 2
    D = r2[0].numel()
    z = r2.sum(dim=tuple(range(1, r2.dim()))) / (sigma ** 2 * D)
    psi = torch.where(z <= c * c, z, 2.0 * c * torch.sqrt(z.clamp_min(1e-12)) - c * c)
    per = 0.5 * D * psi + D * torch.log(sigma)
    loss = config.loss_scale * torch.mean(per) / D
    return loss, {}


def regression_hetero_t_fdlip_loss(
    config: OptimizationConfig,
    flow_map: FlowMap,
    encoder: BaseEncoder,
    interp: Interpolant,
    act: torch.Tensor,
    obs: torch.Tensor,
    delta_t: torch.Tensor,
) -> float:
    """DESIGNED FROM THE MECHANISM (PART CLXXIX): hetero-t NLL + finite-difference
    Lipschitz penalty at ANNULUS scale. The boundary eff-Lip gap (MLP 1.8 vs chiunet
    2.6-3.7 on support->failure paths) lives at window-radius 0.15-0.6 — an
    infinitesimal Jacobian penalty at the support does not control it. Penalty:
    lam * E ||f(obs + eps*v) - sg(f(obs))||^2 / eps^2, v random unit direction over the
    flattened window (rank-8 tangent in 106-d => random directions are ~93% normal),
    eps ~ U[JR_LO, JR_HI] in absolute window-norm units. Clean branch detached so the
    on-manifold regression target is untouched. Envs: FDLIP_LAM, JR_LO, JR_HI."""
    import os
    lam = float(os.environ.get("FDLIP_LAM", "0.1"))
    lo = float(os.environ.get("JR_LO", "0.15"))
    hi = float(os.environ.get("JR_HI", "0.6"))
    t = torch.zeros_like(delta_t, device=delta_t.device)
    act_0 = torch.zeros_like(act, device=act.device)
    obs_emb = encoder(obs, None)
    act_pred, s_raw = flow_map.net(act_0, t, t, obs_emb)
    sigma = torch.nn.functional.softplus(s_raw).reshape(len(act), -1).mean(dim=1) + 1e-3
    nu = config.student_t_df
    r2 = (act_pred - act) ** 2
    per = 0.5 * (nu + 1.0) * torch.log1p(
        r2.sum(dim=tuple(range(1, r2.dim()))) / (nu * sigma ** 2 * r2[0].numel())
    ) * r2[0].numel() + r2[0].numel() * torch.log(sigma)
    nll = torch.mean(per) / r2[0].numel()

    B = act.shape[0]
    def _perturb(v):
        g = torch.randn_like(v)
        g = g / (g.reshape(B, -1).norm(dim=1).reshape(-1, *([1] * (v.dim() - 1))) + 1e-8)
        eps = (lo + (hi - lo) * torch.rand(B, device=v.device)).reshape(
            -1, *([1] * (v.dim() - 1)))
        return v + eps * g, eps.reshape(B)
    if isinstance(obs, dict) or hasattr(obs, "keys"):
        keys = list(obs.keys())
        pert = {}
        eps_b = None
        for k in keys:
            pert[k], eps_b = _perturb(obs[k])
        obs_j = pert
    else:
        obs_j, eps_b = _perturb(obs)
    emb_j = encoder(obs_j, None)
    pred_j, _ = flow_map.net(act_0, t, t, emb_j)
    dif = (pred_j - act_pred.detach()).reshape(B, -1).norm(dim=1)
    fdlip = ((dif / eps_b) ** 2).mean()
    loss = config.loss_scale * nll + lam * fdlip
    return loss, {"fdlip": fdlip.detach()}


def regression_hetero_t_tublip_loss(
    config: OptimizationConfig,
    flow_map: FlowMap,
    encoder: BaseEncoder,
    interp: Interpolant,
    act: torch.Tensor,
    obs: torch.Tensor,
    delta_t: torch.Tensor,
) -> float:
    """Tube-boundary Lipschitz constraint (user-directed): penalize the LOCAL
    directional derivative of the obs->action map AT the trajectory-tube shell,
    not the chord from the support (that is fdlip). Shell point x_b = x + eps*g,
    eps ~ U[TUB_LO, TUB_HI] (annulus band); local FD derivative at x_b over step
    TUB_DELTA along h = g (TUB_DIR=radial: the outward ray, matching the
    boundary-eff-Lip probe) or a fresh random direction (TUB_DIR=rand).
    penalty = lam * E ||f(x_b + delta*h) - f(x_b)||^2 / delta^2. Both shell
    forwards are trained (no detach): flatness AT the shell, on-support fit
    untouched (hetero-t NLL on the clean branch). Envs: TUBLIP_LAM, TUB_LO,
    TUB_HI, TUB_DELTA, TUB_DIR."""
    import os
    lam = float(os.environ.get("TUBLIP_LAM", "0.1"))
    lo = float(os.environ.get("TUB_LO", "0.15"))
    hi = float(os.environ.get("TUB_HI", "0.6"))
    delta = float(os.environ.get("TUB_DELTA", "0.05"))
    radial = os.environ.get("TUB_DIR", "radial") == "radial"
    t = torch.zeros_like(delta_t, device=delta_t.device)
    act_0 = torch.zeros_like(act, device=act.device)
    obs_emb = encoder(obs, None)
    act_pred, s_raw = flow_map.net(act_0, t, t, obs_emb)
    sigma = torch.nn.functional.softplus(s_raw).reshape(len(act), -1).mean(dim=1) + 1e-3
    nu = config.student_t_df
    r2 = (act_pred - act) ** 2
    per = 0.5 * (nu + 1.0) * torch.log1p(
        r2.sum(dim=tuple(range(1, r2.dim()))) / (nu * sigma ** 2 * r2[0].numel())
    ) * r2[0].numel() + r2[0].numel() * torch.log(sigma)
    nll = torch.mean(per) / r2[0].numel()

    B = act.shape[0]
    def _unit(v):
        g = torch.randn_like(v)
        return g / (g.reshape(B, -1).norm(dim=1).reshape(-1, *([1] * (v.dim() - 1))) + 1e-8)
    def _shell(v):
        g = _unit(v)
        eps = (lo + (hi - lo) * torch.rand(B, device=v.device)).reshape(
            -1, *([1] * (v.dim() - 1)))
        h = g if radial else _unit(v)
        return v + eps * g, v + eps * g + delta * h
    if isinstance(obs, dict) or hasattr(obs, "keys"):
        xb, xb2 = {}, {}
        for k in obs:
            xb[k], xb2[k] = _shell(obs[k])
    else:
        xb, xb2 = _shell(obs)
    pb, _ = flow_map.net(act_0, t, t, encoder(xb, None))
    pb2, _ = flow_map.net(act_0, t, t, encoder(xb2, None))
    tublip = ((pb2 - pb).reshape(B, -1).norm(dim=1) ** 2 / delta ** 2).mean()
    loss = config.loss_scale * nll + lam * tublip
    return loss, {"tublip": tublip.detach()}


def regression_pace_loss(
    config: OptimizationConfig,
    flow_map: FlowMap,
    encoder: BaseEncoder,
    interp: Interpolant,
    act: torch.Tensor,
    obs: torch.Tensor,
    delta_t: torch.Tensor,
) -> float:
    """DESIGNED FROM THE TRANSPORT MECHANISM (PART CXCVI): pacing multimodality —
    demos agree on direction (cos .84-.91) but spread 1.4-3.1x in magnitude; the
    conditional mean is direction-correct and ~2x magnitude-shrunk; policies run at
    ~half demo speed and die of horizon exhaustion. Fix (estimator-level): decompose
    the position-channel error into components parallel/perpendicular to the target
    direction; fit perpendicular (path) with L2 but the PARALLEL speed with a
    pinball/quantile loss at tau>0.5 — an upper-quantile speed estimator instead of
    the mean (mode-committing on pacing). Non-position channels: standard L2.
    Envs: PACE_TAU (0.7), PACE_LAM (1.0), PACE_BLOCKS ("0-3,10-13" for 20-d rot6d
    dual-arm; "0-3" for tool-hang 10-d)."""
    import os
    tau = float(os.environ.get("PACE_TAU", "0.7"))
    lam = float(os.environ.get("PACE_LAM", "1.0"))
    blocks = [tuple(int(x) for x in b.split("-")) for b in os.environ.get("PACE_BLOCKS", "0-3,10-13").split(",")]
    t = torch.zeros_like(delta_t, device=delta_t.device)
    act_0 = torch.zeros_like(act, device=act.device)
    obs_emb = encoder(obs, None)
    act_pred = flow_map.get_velocity(t, act_0, obs_emb)
    r = act_pred - act
    pos_mask = torch.zeros(act.shape[-1], dtype=torch.bool, device=act.device)
    for lo, hi in blocks:
        pos_mask[lo:hi] = True
    loss_other = (r[..., ~pos_mask] ** 2).mean()
    loss_pace = 0.0
    for lo, hi in blocks:
        y = act[..., lo:hi]
        a = act_pred[..., lo:hi]
        ny = y.norm(dim=-1, keepdim=True)
        u = y / (ny + 1e-6)
        s_pred = (a * u).sum(dim=-1)
        e_par = s_pred - ny.squeeze(-1)
        pin = torch.where(e_par < 0, tau * (-e_par), (1 - tau) * e_par)
        e_perp = a - s_pred.unsqueeze(-1) * u
        loss_pace = loss_pace + lam * pin.mean() + (e_perp ** 2).mean()
    loss = config.loss_scale * (loss_other + loss_pace)
    return loss, {"pace_pin": pin.mean().detach()}


def regression_paceabs_loss(
    config: OptimizationConfig,
    flow_map: FlowMap,
    encoder: BaseEncoder,
    interp: Interpolant,
    act: torch.Tensor,
    obs: torch.Tensor,
    delta_t: torch.Tensor,
) -> float:
    """ABS-space pace loss via CHUNK-INTERNAL differences: consecutive waypoint
    increments d_t = y[t+1]-y[t] are valid displacement vectors in one normalizer's
    coordinates (unlike the abs waypoints themselves). Pacing shrinkage in abs ==
    under-spaced increments. loss = L2 on waypoint levels + on each pos-block's
    increments: pinball_tau on the parallel (speed) component + L2 on perpendicular.
    Envs: PACE_TAU (0.85), PACE_LAM (1.0), PACE_BLOCKS ("0-3,10-13")."""
    import os
    tau = float(os.environ.get("PACE_TAU", "0.85"))
    lam = float(os.environ.get("PACE_LAM", "1.0"))
    blocks = [tuple(int(x) for x in b.split("-")) for b in os.environ.get("PACE_BLOCKS", "0-3,10-13").split(",")]
    t = torch.zeros_like(delta_t, device=delta_t.device)
    act_0 = torch.zeros_like(act, device=act.device)
    obs_emb = encoder(obs, None)
    act_pred = flow_map.get_velocity(t, act_0, obs_emb)
    level = ((act_pred - act) ** 2).mean()
    pin = torch.zeros((), device=act.device)
    pace = 0.0
    for lo, hi in blocks:
        dy = act[:, 1:, lo:hi] - act[:, :-1, lo:hi]
        da = act_pred[:, 1:, lo:hi] - act_pred[:, :-1, lo:hi]
        ny = dy.norm(dim=-1, keepdim=True)
        u = dy / (ny + 1e-6)
        s_pred = (da * u).sum(dim=-1)
        e_par = s_pred - ny.squeeze(-1)
        pin = torch.where(e_par < 0, tau * (-e_par), (1 - tau) * e_par).mean()
        e_perp = da - s_pred.unsqueeze(-1) * u
        pace = pace + lam * pin + (e_perp ** 2).mean()
    loss = config.loss_scale * (level + pace)
    return loss, {"paceabs_pin": pin.detach()}


def regression_hetero_gauss_loss(
    config: OptimizationConfig,
    flow_map: FlowMap,
    encoder: BaseEncoder,
    interp: Interpolant,
    act: torch.Tensor,
    obs: torch.Tensor,
    delta_t: torch.Tensor,
) -> float:
    """ABLATION (isolates hetero-t's two ingredients): heteroscedastic GAUSSIAN NLL
    with the same learned per-sample scale head — state-conditional preconditioning
    (1/sigma^2 weighting) WITHOUT the heavy-tail/bounded-influence property
    (psi = r/sigma^2 is linear within a state). Comparing L2 / Cauchy / this /
    hetero-t factorizes {learned scale} x {bounded influence}."""
    t = torch.zeros_like(delta_t, device=delta_t.device)
    act_0 = torch.zeros_like(act, device=act.device)
    obs_emb = encoder(obs, None)
    act_pred, s_raw = flow_map.net(act_0, t, t, obs_emb)
    import os as _os
    _gsmin = float(_os.environ.get("HG_SMIN", "0"))
    sigma = torch.nn.functional.softplus(s_raw).reshape(len(act), -1).mean(dim=1) + 1e-3 + _gsmin
    r2 = (act_pred - act) ** 2
    D = r2[0].numel()
    per = 0.5 * r2.sum(dim=tuple(range(1, r2.dim()))) / (sigma ** 2) + D * torch.log(sigma)
    loss = config.loss_scale * torch.mean(per) / D
    return loss, {}


def regression_sigaux_loss(
    config: OptimizationConfig,
    flow_map: FlowMap,
    encoder: BaseEncoder,
    interp: Interpolant,
    act: torch.Tensor,
    obs: torch.Tensor,
    delta_t: torch.Tensor,
) -> float:
    """DECOUPLING control for hetero_gauss: the sigma head is trained to predict
    the (detached) residual scale — shaping shared features as an auxiliary task —
    while the action fit stays PLAIN unweighted MSE (sigma never re-prices the
    action residuals). If this arm reproduces hetero_gauss's contractive
    extension/SR, the sigma benefit is representation shaping; if it matches L2,
    the benefit is the transient 1/sigma^2 weighting path. Env: SIGAUX_LAM."""
    import os

    lam = float(os.environ.get("SIGAUX_LAM", "1.0"))
    t = torch.zeros_like(delta_t, device=delta_t.device)
    act_0 = torch.zeros_like(act, device=act.device)
    obs_emb = encoder(obs, None)
    act_pred, s_raw = flow_map.net(act_0, t, t, obs_emb)
    sigma = torch.nn.functional.softplus(s_raw).reshape(len(act), -1).mean(dim=1) + 1e-3
    r2 = (act_pred - act) ** 2
    D = r2[0].numel()
    mse = r2.sum(dim=tuple(range(1, r2.dim())))
    aux = 0.5 * mse.detach() / (sigma ** 2) + D * torch.log(sigma)
    loss = config.loss_scale * torch.mean(mse + lam * aux) / D
    return loss, {}


def regression_distinc_loss(
    config: OptimizationConfig,
    flow_map: FlowMap,
    encoder: BaseEncoder,
    interp: Interpolant,
    act: torch.Tensor,
    obs: torch.Tensor,
    delta_t: torch.Tensor,
) -> float:
    """COROLLARY-3 INSTRUMENT (PART CCXV): label-keyed distinction regularizer —
    re-prices microscopic label differences at O(1) instead of L2's O(delta^2).
    For each sample, find its nearest EMBEDDING neighbor in the batch (the pair
    most at risk of folding); if their action chunks differ by more than DIST_EPS,
    apply a hinge pushing the embeddings apart to a margin (DIST_M x median batch
    embedding distance). DIST_PRICE=const prices every real distinction equally
    (the fix); DIST_PRICE=l2 weights the same hinge by (delta/eps)^2-normalized
    L2 pricing (negative control: predicted NOT to unfold). Base loss: L2.
    Envs: DIST_LAM, DIST_EPS, DIST_M, DIST_PRICE."""
    import os
    lam = float(os.environ.get("DIST_LAM", "1.0"))
    eps = float(os.environ.get("DIST_EPS", "0.1"))
    marg = float(os.environ.get("DIST_M", "0.5"))
    price = os.environ.get("DIST_PRICE", "const")
    t = torch.zeros_like(delta_t, device=delta_t.device)
    act_0 = torch.zeros_like(act, device=act.device)
    obs_emb = encoder(obs, None)
    act_pred = flow_map.get_velocity(t, act_0, obs_emb)
    base = ((act_pred - act) ** 2).mean()

    B = act.shape[0]
    e = obs_emb.reshape(B, -1)
    x = (obs["state"] if isinstance(obs, dict) or hasattr(obs, "keys") else obs).reshape(B, -1)
    with torch.no_grad():
        Dm = torch.cdist(e, e)
        Dm.fill_diagonal_(float("inf"))
        # fold pairs are embedding-near but INPUT-far: exclude input-near pairs
        # (temporal/spatial neighbors, whose label differences are legitimate)
        Dx = torch.cdist(x, x)
        far_thr = torch.quantile(Dx[torch.isfinite(Dm)].flatten(), float(os.environ.get("DIST_OBSQ", "0.3")))
        Dm[Dx < far_thr] = float("inf")
        jstar = Dm.argmin(dim=1)
        valid = torch.isfinite(Dm.gather(1, jstar[:, None]).squeeze(1)).float()
        med = torch.median(torch.cdist(e, e)[~torch.eye(B, dtype=torch.bool, device=e.device)])
    dlab = (act - act[jstar]).reshape(B, -1).norm(dim=1)
    demb = (e - e[jstar]).norm(dim=1)
    m = marg * med
    hinge = torch.clamp(m - demb, min=0) ** 2 / (m ** 2 + 1e-8)
    mask = (dlab > eps).float() * valid
    if price == "l2":
        w = (dlab / eps) ** 2 * (eps ** 2)  # L2-priced: incentive ~ delta^2
    else:
        w = torch.ones_like(dlab)           # const-priced: O(1) per real distinction
    pen = (mask * w * hinge).sum() / (mask.sum() + 1e-6)
    loss = config.loss_scale * (base + lam * pen)
    return loss, {"distinc_pen": pen.detach(), "distinc_frac": mask.mean().detach()}


def regression_trim_loss(
    config: OptimizationConfig,
    flow_map: FlowMap,
    encoder: BaseEncoder,
    interp: Interpolant,
    act: torch.Tensor,
    obs: torch.Tensor,
    delta_t: torch.Tensor,
) -> float:
    """DESIGNED FROM THE MECHANISM (PART CLXVII): batch-trimmed L2 — zero the top-q%
    per-sample losses each batch (online despike; blocks tail influence adaptively)."""
    import os
    t = torch.zeros_like(delta_t, device=delta_t.device)
    act_0 = torch.zeros_like(act, device=act.device)
    obs_emb = encoder(obs, None)
    act_pred = flow_map.get_velocity(t, act_0, obs_emb)
    per = ((act_pred - act) ** 2).sum(dim=tuple(range(1, act.dim())))
    q = float(os.environ.get("TRIMQ", "0.10"))
    thr = torch.quantile(per.detach(), 1.0 - q)
    loss = config.loss_scale * torch.mean(per * (per.detach() <= thr))
    return loss, {}


def regression_student_t_loss(
    config: OptimizationConfig,
    flow_map: FlowMap,
    encoder: BaseEncoder,
    interp: Interpolant,
    act: torch.Tensor,
    obs: torch.Tensor,
    delta_t: torch.Tensor,
) -> float:
    """Regression with Student-t NLL — ((nu+1)/2) * log(1 + (e/sigma)^2 / nu).

    Matches a heavy-tailed (Student-t) demonstration-noise model. df=1 recovers
    the Cauchy loss; df->inf recovers MSE. sigma = config.cauchy_c (scale)."""
    t = torch.zeros_like(delta_t, device=delta_t.device)
    act_0 = torch.zeros_like(act, device=act.device)
    obs_emb = encoder(obs, None)
    act_pred = flow_map.get_velocity(t, act_0, obs_emb)

    nu = config.student_t_df
    sigma = config.cauchy_c
    r2 = ((act_pred - act) / sigma) ** 2
    loss = 0.5 * (nu + 1.0) * torch.log1p(r2 / nu)
    loss = config.loss_scale * torch.mean(loss)
    return loss, {}


def regression_loss(
    config: OptimizationConfig,
    flow_map: FlowMap,
    encoder: BaseEncoder,
    interp: Interpolant,
    act: torch.Tensor,
    obs: torch.Tensor,
    delta_t: torch.Tensor,
) -> float:
    """Standard regression loss."""
    import os

    # sample
    t = torch.zeros_like(delta_t, device=delta_t.device)
    act_0 = torch.zeros_like(act, device=act.device)

    # OBSJIT: Gaussian input jitter on the (normalized) obs window, labels fixed —
    # denoising augmentation that trains the field in the annulus around the support
    ojit = float(os.environ.get("OBSJIT", "0"))
    if ojit > 0:
        obs = {k: v + ojit * torch.randn_like(v) for k, v in obs.items()} if isinstance(obs, dict) else obs + ojit * torch.randn_like(obs)

    # get condition
    obs_emb = encoder(obs, None)

    # predict
    act_pred = flow_map.get_velocity(t, act_0, obs_emb)

    # compute loss
    err = act_pred - act
    _md = float(os.environ.get("MSE_DEAD", "0"))
    if _md > 0:
        # dead-zone MSE: per-dim squared errors below _md^2 contribute 0
        err = torch.sign(err) * torch.sqrt(torch.relu(err ** 2 - _md ** 2))
    rotw = float(os.environ.get("ROTW", "0"))
    if rotw > 0:
        # upweight the rotation channels (rot6d dims 3:9 of the 10-dim action) to test
        # whether the attenuated retry-rotation servo is a gradient-scale artifact
        w = torch.ones(err.shape[-1], device=err.device)
        w[3:9] = rotw
        err = err * w.sqrt()
    loss = get_norm(err, config.norm_type, config.cauchy_c)
    loss = config.loss_scale * torch.mean(loss)
    return loss, {}


def regression_binw_loss(
    config: OptimizationConfig,
    flow_map: FlowMap,
    encoder: BaseEncoder,
    interp: Interpolant,
    act: torch.Tensor,
    obs: torch.Tensor,
    delta_t: torch.Tensor,
) -> float:
    """C2 balancer: histogram-equalized MSE — per-sample weight proportional to
    the inverse batch-frequency of its chunk-magnitude decile. Balances CONTENT
    DENSITY (counts), orthogonal to sigma's residual-scale repricing; the
    matched cure for duration/frequency imbalance. Env: BINW_K (bins, def 10)."""
    import os

    k = int(os.environ.get("BINW_K", "10"))
    t = torch.zeros_like(delta_t, device=delta_t.device)
    act_0 = torch.zeros_like(act, device=act.device)
    obs_emb = encoder(obs, None)
    act_pred = flow_map.get_velocity(t, act_0, obs_emb)
    err = act_pred - act
    per = get_norm(err, config.norm_type, config.cauchy_c).mean(dim=tuple(range(1, err.dim() - 1)))
    with torch.no_grad():
        mag = act.reshape(len(act), -1).norm(dim=1)
        edges = torch.quantile(mag, torch.linspace(0, 1, k + 1, device=mag.device))
        bin_id = torch.bucketize(mag, edges[1:-1])
        counts = torch.bincount(bin_id, minlength=k).float().clamp(min=1)
        w = (1.0 / counts)[bin_id]
        w = w * (len(act) / w.sum())
    loss = config.loss_scale * torch.mean(w * per)
    return loss, {}


def regression_headonly_loss(
    config: OptimizationConfig,
    flow_map: FlowMap,
    encoder: BaseEncoder,
    interp: Interpolant,
    act: torch.Tensor,
    obs: torch.Tensor,
    delta_t: torch.Tensor,
) -> float:
    """EXPERIMENTUM CRUCIS for chunking-as-target-supervision vs chunking-as-
    execution (Simchowitz et al. 2503.09722): identical H-step architecture and
    identical deployment, but the loss is MASKED to the first HEADK chunk steps
    (default 2 = the obs window + executed head). If the H=10 baseline's AS=1
    performance needs the TAIL supervision (steps 2..H-1), this arm degrades to
    short-H levels despite executing nothing open-loop; the execution account
    predicts no difference. Env: HEADK."""
    import os

    k = int(os.environ.get("HEADK", "2"))
    t = torch.zeros_like(delta_t, device=delta_t.device)
    act_0 = torch.zeros_like(act, device=act.device)
    obs_emb = encoder(obs, None)
    act_pred = flow_map.get_velocity(t, act_0, obs_emb)
    err = (act_pred - act)[:, :k]
    loss = get_norm(err, config.norm_type, config.cauchy_c)
    loss = config.loss_scale * torch.mean(loss)
    return loss, {}

def regression_labelnoise_loss(
    config: OptimizationConfig,
    flow_map: FlowMap,
    encoder: BaseEncoder,
    interp: Interpolant,
    act: torch.Tensor,
    obs: torch.Tensor,
    delta_t: torch.Tensor,
) -> float:
    """Damian-Ma-Lee test arm: plain MSE with explicit Gaussian LABEL noise
    (targets act + LN_SIGMA * eps, unbiased). Any SR change vs plain L2 is
    the implicit tr(H)-flatness channel (label noise), disjoint from the
    Bishop input-noise/Jacobian channel (MIP) by construction."""
    import os as _os

    sig = float(_os.environ.get("LN_SIGMA", "0.1"))
    t = torch.zeros_like(delta_t, device=delta_t.device)
    act_0 = torch.zeros_like(act, device=act.device)
    obs_emb = encoder(obs, None)
    act_pred, _ = flow_map.net(act_0, t, t, obs_emb)
    tgt = act + sig * torch.randn_like(act)
    return config.loss_scale * ((act_pred - tgt) ** 2).mean(), {}


def regression_dcw_loss(
    config: OptimizationConfig,
    flow_map: FlowMap,
    encoder: BaseEncoder,
    interp: Interpolant,
    act: torch.Tensor,
    obs: torch.Tensor,
    delta_t: torch.Tensor,
) -> float:
    """EXPLICIT ANTI-CHUNK-BIAS MSE (PART CCCLXXXIV mediator): plain MSE plus
    an extra penalty on the CHUNK-COHERENT (DC) component of the residual,
    lambda * ||mean_t e_t||^2 per sample (DCW_LAMBDA, default 4). Targets the
    measured SR mediator (emitted DC bias) directly at the training residual
    level; the incoherent component keeps standard weight."""
    import os as _os

    lam = float(_os.environ.get("DCW_LAMBDA", "4.0"))
    t = torch.zeros_like(delta_t, device=delta_t.device)
    act_0 = torch.zeros_like(act, device=act.device)
    obs_emb = encoder(obs, None)
    act_pred, _ = flow_map.net(act_0, t, t, obs_emb)
    e = act_pred - act
    mse = (e ** 2).mean()
    dc = (e.mean(dim=1) ** 2).mean()
    return config.loss_scale * (mse + lam * dc), {}


def regression_dup100_loss(
    config: OptimizationConfig,
    flow_map: FlowMap,
    encoder: BaseEncoder,
    interp: Interpolant,
    act: torch.Tensor,
    obs: torch.Tensor,
    delta_t: torch.Tensor,
) -> float:
    """DUP0_HIGH control: same single view (t=0, zeros) but total weight 101x (1x + 100x
    duplicate). Separates loss weight from the auxiliary view. (Adam should absorb the
    global scale; this makes the prediction explicit.)"""
    t = torch.zeros_like(delta_t, device=delta_t.device)
    act_0 = torch.zeros_like(act, device=act.device)
    obs_emb = encoder(obs, None)
    act_pred = flow_map.get_velocity(t, act_0, obs_emb)
    base = get_norm(act_pred - act, config.norm_type, config.cauchy_c)
    loss = config.loss_scale * torch.mean(base + 100.0 * base)
    return loss, {}


_RW_CACHE = {"loaded": False}


def _rw_load(dev):
    import os as _os
    import numpy as _np
    a = _np.load(_os.environ["RW_ASSETS"])
    prof = [float(x) for x in _os.environ.get(
        "RW_PROFILE", "1.027,0.974,0.972,1.056,1.127").split(",")]
    _RW_CACHE.update(
        loaded=True,
        C=torch.tensor(a["center"], device=dev, dtype=torch.float32),
        SG=torch.tensor(a["sigma"], device=dev, dtype=torch.float32),
        A=torch.tensor(a["obs_A"], device=dev, dtype=torch.float32),
        B=torch.tensor(a["obs_B"], device=dev, dtype=torch.float32),
        edges=torch.tensor([0.5, 1.0, 1.5, 2.5], device=dev),
        w=torch.tensor(prof, device=dev, dtype=torch.float32),
        TG=torch.tensor(a["tangent"], device=dev, dtype=torch.float32) if "tangent" in a else None,
        Aa=torch.tensor(a["act_A"], device=dev, dtype=torch.float32) if "act_A" in a else None,
        Ba=torch.tensor(a["act_B"], device=dev, dtype=torch.float32) if "act_B" in a else None,
    )


def regression_rw_loss(
    config: OptimizationConfig,
    flow_map: FlowMap,
    encoder: BaseEncoder,
    interp: Interpolant,
    act: torch.Tensor,
    obs: torch.Tensor,
    delta_t: torch.Tensor,
) -> float:
    """MSE-RW: plain regression with per-sample DISTANCE reweighting w(d) that
    reproduces MIP's GRADW rel profile (advisor causal test: is reweighting
    sufficient?). d = nearest-knot tube distance of the last obs frame,
    computed on the fly (RW_ASSETS npz: center/sigma + affine unnorm coefs)."""
    if not _RW_CACHE["loaded"]:
        _rw_load(act.device)
    t = torch.zeros_like(delta_t, device=delta_t.device)
    act_0 = torch.zeros_like(act, device=act.device)
    with torch.no_grad():
        eef = obs[:, -1, 44:47] * _RW_CACHE["A"] + _RW_CACHE["B"]   # raw meters
        diff = eef[:, None, :] - _RW_CACHE["C"][None]                # (B,P,3)
        d = (diff.norm(dim=2) / _RW_CACHE["SG"][None]).min(dim=1).values
        w = _RW_CACHE["w"][torch.bucketize(d, _RW_CACHE["edges"])]   # (B,)
    obs_emb = encoder(obs, None)
    act_pred = flow_map.get_velocity(t, act_0, obs_emb)
    base = get_norm(act_pred - act, config.norm_type, config.cauchy_c)  # (B, Ta)
    per_sample = base.mean(dim=tuple(range(1, base.dim()))) if base.dim() > 1 else base
    loss = config.loss_scale * torch.mean(w * per_sample)
    return loss, {}


def regression_jacreg_loss(
    config: OptimizationConfig,
    flow_map: FlowMap,
    encoder: BaseEncoder,
    interp: Interpolant,
    act: torch.Tensor,
    obs: torch.Tensor,
    delta_t: torch.Tensor,
) -> float:
    """Regression + EXPLICIT obs-Jacobian penalty (double backprop, Hutchinson).
    loss = ||f(0,0,obs) - act||^2 + lambda * E_v ||d(f.v)/d obs||^2  (= lambda ||J_c f||_F^2)
    Tests the 'smoothness in c alone suffices' hypothesis WITHOUT touching targets
    (unlike obs-noise, which convolves the target function). JACREG_LAMBDA env var."""
    import os
    lam = float(os.environ.get("JACREG_LAMBDA", "1e-3"))
    t = torch.zeros_like(delta_t, device=delta_t.device)
    act_0 = torch.zeros_like(act, device=act.device)
    if isinstance(obs, dict) or hasattr(obs, "keys"):
        obs_in = {k: v.detach().requires_grad_(True) for k, v in obs.items()}
        leaves = list(obs_in.values())
    else:
        obs_in = obs.detach().requires_grad_(True)
        leaves = [obs_in]
    obs_emb = encoder(obs_in, None)
    act_pred = flow_map.get_velocity(t, act_0, obs_emb)
    data_loss = torch.mean(get_norm(act_pred - act, config.norm_type, config.cauchy_c))
    v = torch.randn_like(act_pred)
    grads = torch.autograd.grad((act_pred * v).sum(), leaves, create_graph=True)
    B = act.shape[0]
    jpen = sum(g.reshape(B, -1).pow(2).sum(dim=1) for g in grads).mean()
    loss = config.loss_scale * data_loss + lam * jpen
    return loss, {"jacobian_penalty": jpen.detach()}


def mip_eqw_loss(
    config: OptimizationConfig,
    flow_map: FlowMap,
    encoder: BaseEncoder,
    interp: Interpolant,
    act: torch.Tensor,
    obs: torch.Tensor,
    delta_t: torch.Tensor,
) -> float:
    """MIP loss WITHOUT the 1/t, 1/(1-t) scalings: both heads weighted equally.
    Isolates whether the 100x denoise-head gradient weight matters."""
    s = torch.zeros_like(delta_t, device=delta_t.device)
    t = torch.zeros_like(delta_t, device=delta_t.device) + config.t_two_step
    act_0 = torch.zeros_like(act, device=act.device)
    noise = torch.empty_like(act).normal_(0, 1)
    act_t = act + (1 - config.t_two_step) * noise
    obs_emb = encoder(obs, None)
    act_pred_0 = flow_map.get_velocity(s, act_0, obs_emb)
    act_pred_1 = flow_map.get_velocity(t, act_t, obs_emb)
    loss0 = get_norm(act_pred_0 - act, config.norm_type, config.cauchy_c)
    loss1 = get_norm(act_pred_1 - act, config.norm_type, config.cauchy_c)
    loss = config.loss_scale * torch.mean(loss0 + loss1)
    return loss, {}


def denoise_only_loss(
    config: OptimizationConfig,
    flow_map: FlowMap,
    encoder: BaseEncoder,
    interp: Interpolant,
    act: torch.Tensor,
    obs: torch.Tensor,
    delta_t: torch.Tensor,
) -> float:
    """ONLY the MIP denoising head (no regression head). Used for the NO-WEIGHT-SHARING
    ablation: train this denoiser as a SEPARATE network, then compose at inference with
    a frozen MSE policy (a = D(t_two_step, MSE(obs), obs)). If the composition fails to
    match shared-weight MIP-step1, weight sharing during training is the essential
    transmission channel."""
    t = torch.zeros_like(delta_t, device=delta_t.device) + config.t_two_step
    noise = torch.empty_like(act).normal_(0, 1)
    act_t = act + (1 - config.t_two_step) * noise
    obs_emb = encoder(obs, None)
    act_pred_1 = flow_map.get_velocity(t, act_t, obs_emb)
    loss1 = get_norm((act_pred_1 - act) / (1 - config.t_two_step), config.norm_type, config.cauchy_c)
    loss = config.loss_scale * torch.mean(loss1)
    return loss, {}


_SCRAMBLE_Q = {}


def _scramble_q(dim, device):
    if dim not in _SCRAMBLE_Q:
        rng = np.random.RandomState(0)
        q, _ = np.linalg.qr(rng.randn(dim, dim))
        _SCRAMBLE_Q[dim] = torch.tensor(q, dtype=torch.float32)
    return _SCRAMBLE_Q[dim].to(device)


def denoise_zeroin_loss(
    config: OptimizationConfig,
    flow_map: FlowMap,
    encoder: BaseEncoder,
    interp: Interpolant,
    act: torch.Tensor,
    obs: torch.Tensor,
    delta_t: torch.Tensor,
) -> float:
    """EXP3 variant: t-slice + 1/(1-t) loss weight of the denoising head, but the action
    input is ZEROS (no noisy action, no chart info). Isolates whether the corrective
    s-extension comes from the noisy-action input or merely from the slice/weight."""
    t = torch.zeros_like(delta_t, device=delta_t.device) + config.t_two_step
    act_in = torch.zeros_like(act, device=act.device)
    obs_emb = encoder(obs, None)
    pred = flow_map.get_velocity(t, act_in, obs_emb)
    loss = get_norm((pred - act) / (1 - config.t_two_step), config.norm_type, config.cauchy_c)
    return config.loss_scale * torch.mean(loss), {}


def denoise_zeroin_flat_loss(
    config: OptimizationConfig,
    flow_map: FlowMap,
    encoder: BaseEncoder,
    interp: Interpolant,
    act: torch.Tensor,
    obs: torch.Tensor,
    delta_t: torch.Tensor,
) -> float:
    """zeroin WITHOUT the 1/(1-t) weight: plain regression at the t=0.9 slice, zeros input.
    Under Adam the global scale is absorbed anyway; this makes it explicit. Isolates the
    t-slice VALUE as the only difference vs plain MSE."""
    t = torch.zeros_like(delta_t, device=delta_t.device) + config.t_two_step
    act_in = torch.zeros_like(act, device=act.device)
    obs_emb = encoder(obs, None)
    pred = flow_map.get_velocity(t, act_in, obs_emb)
    loss = get_norm(pred - act, config.norm_type, config.cauchy_c)
    return config.loss_scale * torch.mean(loss), {}


def denoise_randin_loss(
    config: OptimizationConfig,
    flow_map: FlowMap,
    encoder: BaseEncoder,
    interp: Interpolant,
    act: torch.Tensor,
    obs: torch.Tensor,
    delta_t: torch.Tensor,
) -> float:
    """EXP4 variant: action input is PURE random noise N(0,I) (uncentered nuisance, no
    action-tube structure). Tests 'randomized nuisance input' vs 'action-centered tube'."""
    t = torch.zeros_like(delta_t, device=delta_t.device) + config.t_two_step
    act_in = torch.empty_like(act).normal_(0, 1)
    obs_emb = encoder(obs, None)
    pred = flow_map.get_velocity(t, act_in, obs_emb)
    loss = get_norm((pred - act) / (1 - config.t_two_step), config.norm_type, config.cauchy_c)
    return config.loss_scale * torch.mean(loss), {}


def denoise_scramble_loss(
    config: OptimizationConfig,
    flow_map: FlowMap,
    encoder: BaseEncoder,
    interp: Interpolant,
    act: torch.Tensor,
    obs: torch.Tensor,
    delta_t: torch.Tensor,
) -> float:
    """EXP5 variant: the noisy action input is rotated by a FIXED random orthogonal Q
    (per-step 10-dim coords). Destroys the physical action-coordinate alignment while
    keeping the centered-tube information content. Distinguishes 'generic conditional
    denoising' from 'physical action-coordinate feedback prior'."""
    t = torch.zeros_like(delta_t, device=delta_t.device) + config.t_two_step
    noise = torch.empty_like(act).normal_(0, 1)
    act_t = act + (1 - config.t_two_step) * noise
    Q = _scramble_q(act.shape[-1], act.device)
    act_in = act_t @ Q.T
    obs_emb = encoder(obs, None)
    pred = flow_map.get_velocity(t, act_in, obs_emb)
    loss = get_norm((pred - act) / (1 - config.t_two_step), config.norm_type, config.cauchy_c)
    return config.loss_scale * torch.mean(loss), {}


def regression_tjitter_loss(
    config: OptimizationConfig,
    flow_map: FlowMap,
    encoder: BaseEncoder,
    interp: Interpolant,
    act: torch.Tensor,
    obs: torch.Tensor,
    delta_t: torch.Tensor,
) -> float:
    """MSE with resampled TARGET jitter: ||f(s,0,0) - (a + sigma*eps)||^2, eps fresh per
    batch. Identical minimizer and mean gradient to plain MSE (the noise term is a
    constant in expectation); injects only persistent stationary gradient dither with NO
    input-side ball constraint. Discriminates 'generic gradient noise' from 'input
    ball-flatness' as the erosion-arrest mechanism. Env: TJITTER_STD (default 0.1)."""
    import os
    std = float(os.environ.get("TJITTER_STD", "0.1"))
    t = torch.zeros_like(delta_t, device=delta_t.device)
    act_0 = torch.zeros_like(act, device=act.device)
    obs_emb = encoder(obs, None)
    act_pred = flow_map.get_velocity(t, act_0, obs_emb)
    target = act + std * torch.randn_like(act)
    loss = get_norm(act_pred - target, config.norm_type, config.cauchy_c)
    return config.loss_scale * torch.mean(loss), {}


def regression_stressreg_loss(
    config: OptimizationConfig,
    flow_map: FlowMap,
    encoder: BaseEncoder,
    interp: Interpolant,
    act: torch.Tensor,
    obs: torch.Tensor,
    delta_t: torch.Tensor,
) -> float:
    """Regression fit + isometry (stress) regularizer: match z-scored embedding pair
    distances to z-scored state pair distances over in-batch pairs. Directly defends the
    state metric (MDS-style), unlike InfoNCE uniformity. Env: STRESS_LAMBDA (default 1.0),
    STRESS_MODE (global = random in-batch pairs; local = each sample paired with its
    nearest in-batch state neighbor, aiming the constraint at the fine scale where the
    fold/NN-assignment lives)."""
    import os
    lam = float(os.environ.get("STRESS_LAMBDA", "1.0"))
    mode = os.environ.get("STRESS_MODE", "global")
    t = torch.zeros_like(delta_t, device=delta_t.device)
    act_0 = torch.zeros_like(act, device=act.device)
    obs_emb = encoder(obs, None)
    act_pred = flow_map.get_velocity(t, act_0, obs_emb)
    loss_fit = get_norm(act_pred - act, config.norm_type, config.cauchy_c)
    B = len(act)
    e = obs_emb.reshape(B, -1)
    e = e / (e.norm(dim=1, keepdim=True) + 1e-9)
    if isinstance(obs, dict) or hasattr(obs, "keys"):
        x = obs["state"].reshape(B, -1)
    else:
        x = obs.reshape(B, -1)
    if mode == "local":
        with torch.no_grad():
            dmat = torch.cdist(x, x)
            dmat.fill_diagonal_(float("inf"))
            idx = dmat.argmin(dim=1)
    else:
        idx = torch.randperm(B, device=act.device)
    de = 1.0 - (e * e[idx]).sum(1)
    dx = (x - x[idx]).norm(dim=1)
    de_z = (de - de.mean()) / (de.std() + 1e-6)
    dx_z = (dx - dx.mean()) / (dx.std() + 1e-6)
    loss_stress = ((de_z - dx_z) ** 2).mean()
    loss = config.loss_scale * torch.mean(loss_fit) + lam * loss_stress
    return loss, {}


def regression_geomreg_loss(
    config: OptimizationConfig,
    flow_map: FlowMap,
    encoder: BaseEncoder,
    interp: Interpolant,
    act: torch.Tensor,
    obs: torch.Tensor,
    delta_t: torch.Tensor,
) -> float:
    """Regression fit + explicit embedding-metric preservation (InfoNCE):
    the embedding of a noise-perturbed state must be closer to its own clean embedding
    than to other states' embeddings. Constrains the ENCODER METRIC only -- outputs free.
    Env: GEOM_LAMBDA (default 1.0), GEOM_STD (default 0.1), GEOM_TAU (default 0.1)."""
    import os
    lam = float(os.environ.get("GEOM_LAMBDA", "1.0"))
    std = float(os.environ.get("GEOM_STD", "0.1"))
    tau = float(os.environ.get("GEOM_TAU", "0.1"))
    t = torch.zeros_like(delta_t, device=delta_t.device)
    act_0 = torch.zeros_like(act, device=act.device)
    obs_emb = encoder(obs, None)
    act_pred = flow_map.get_velocity(t, act_0, obs_emb)
    loss_fit = get_norm(act_pred - act, config.norm_type, config.cauchy_c)
    if isinstance(obs, dict) or hasattr(obs, "keys"):
        obs_p = {k: v + std * torch.randn_like(v) for k, v in obs.items()}
    else:
        obs_p = obs + std * torch.randn_like(obs)
    e = obs_emb.reshape(len(act), -1)
    ep = encoder(obs_p, None).reshape(len(act), -1)
    e = e / (e.norm(dim=1, keepdim=True) + 1e-9)
    ep = ep / (ep.norm(dim=1, keepdim=True) + 1e-9)
    logits = (e @ ep.T) / tau
    labels = torch.arange(len(act), device=act.device)
    loss_geo = torch.nn.functional.cross_entropy(logits, labels)
    loss = config.loss_scale * torch.mean(loss_fit) + lam * loss_geo
    return loss, {}


_RANDAUG = {"net": None, "shape": None}
def randaug_offset(obs, act):
    """Fixed random function g(s) of the (normalized) observation, output shaped like the
    action chunk. Frozen 2-layer tanh net, deterministic from RANDAUG_SEED; output scale
    RANDAUG_GAIN. Used by regression_randaug: fit a + g(s), subtract g(s) at inference."""
    import os
    x = obs["state"] if (isinstance(obs, dict) or hasattr(obs, "keys")) else obs
    B = len(x)
    x = x.reshape(B, -1)
    out_dim = act.shape[-2] * act.shape[-1]
    if _RANDAUG["net"] is None or _RANDAUG["shape"] != (x.shape[1], out_dim):
        gain = float(os.environ.get("RANDAUG_GAIN", "0.3"))
        seed = int(os.environ.get("RANDAUG_SEED", "0"))
        gen = torch.Generator().manual_seed(seed)
        w1 = torch.randn(x.shape[1], 256, generator=gen) / (x.shape[1] ** 0.5)
        w2 = torch.randn(256, out_dim, generator=gen) / (256 ** 0.5) * gain
        _RANDAUG["net"] = {"w1": w1, "w2": w2}
        _RANDAUG["shape"] = (x.shape[1], out_dim)
    net = _RANDAUG["net"]
    if net["w1"].device != x.device:
        net["w1"] = net["w1"].to(x.device); net["w2"] = net["w2"].to(x.device)
    with torch.no_grad():
        off = torch.tanh(x @ net["w1"]) @ net["w2"]
    return off.reshape(B, act.shape[-2], act.shape[-1])


def regression_randaug_loss(
    config: OptimizationConfig,
    flow_map: FlowMap,
    encoder: BaseEncoder,
    interp: Interpolant,
    act: torch.Tensor,
    obs: torch.Tensor,
    delta_t: torch.Tensor,
) -> float:
    """Random-function label augmentation: fit a + g(s) with g a FIXED random function of the
    observation (exactly removable at inference: pi(s) = f(s) - g(s)). Same minimizer as MSE
    for the recovered policy, but the training target is state-rich everywhere — merging any
    two states g separates costs loss, incl. inside slow windows (anti-starvation through the
    label channel). Env: RANDAUG_GAIN (default 0.3), RANDAUG_SEED (default 0)."""
    t = torch.zeros_like(delta_t, device=delta_t.device)
    act_0 = torch.zeros_like(act, device=act.device)
    obs_emb = encoder(obs, None)
    act_pred = flow_map.get_velocity(t, act_0, obs_emb)
    target = act + randaug_offset(obs, act)
    loss = get_norm(act_pred - target, config.norm_type, config.cauchy_c)
    loss = config.loss_scale * torch.mean(loss)
    return loss, {}


def regression_sigreg_loss(
    config: OptimizationConfig,
    flow_map: FlowMap,
    encoder: BaseEncoder,
    interp: Interpolant,
    act: torch.Tensor,
    obs: torch.Tensor,
    delta_t: torch.Tensor,
) -> float:
    """Regression fit + SIGReg (LeJEPA, Balestriero & LeCun 2025): sketched isotropic-
    Gaussian regularization of the embedding. K random unit directions per step; each 1-D
    projection is pushed toward N(0,1) via the Epps-Pulley characteristic-function
    statistic (fixed quadrature over t). Distribution-level only — regularizes the
    embedding MARGINAL toward N(0,I); pulls no specific pairs (unlike InfoNCE/stress).
    Env: SIGREG_LAMBDA (default 0.2), SIGREG_K (default 64), SIGREG_NORM
    ("abs" = literal N(0,1) target incl. absolute scale; "global" = scale-free: embedding
    centered and divided by its global batch RMS before projection, so anisotropy and
    non-Gaussian shape are penalized but absolute scale is unconstrained)."""
    import os
    lam = float(os.environ.get("SIGREG_LAMBDA", "0.2"))
    K = int(os.environ.get("SIGREG_K", "64"))
    mode = os.environ.get("SIGREG_NORM", "abs")
    t = torch.zeros_like(delta_t, device=delta_t.device)
    act_0 = torch.zeros_like(act, device=act.device)
    obs_emb = encoder(obs, None)
    act_pred = flow_map.get_velocity(t, act_0, obs_emb)
    loss_fit = get_norm(act_pred - act, config.norm_type, config.cauchy_c)
    B = len(act)
    z = obs_emb.reshape(B, -1)
    if mode == "global":
        # entry-RMS normalization: isotropic N(0, s^2 I) of ANY scale s passes exactly;
        # anisotropy (per-direction variance != global mean variance) still fails.
        z = z - z.mean(0, keepdim=True)
        z = z / (z.pow(2).mean().sqrt() + 1e-9)
    u = torch.randn(z.shape[1], K, device=z.device)
    u = u / (u.norm(dim=0, keepdim=True) + 1e-9)
    x = z @ u                                            # (B, K) sketched projections
    tj = torch.linspace(0.5, 4.0, 8, device=z.device)    # quadrature nodes
    w = torch.exp(-tj ** 2 / 4)                          # quadrature weight
    tx = x.unsqueeze(-1) * tj                            # (B, K, T)
    c = torch.cos(tx).mean(0)                            # (K, T) empirical CF, real part
    s = torch.sin(tx).mean(0)                            # (K, T) imaginary part
    target = torch.exp(-tj ** 2 / 2)                     # CF of N(0,1)
    ep = ((c - target) ** 2 + s ** 2) * w
    loss_sig = ep.sum(-1).mean() / w.sum()
    loss = config.loss_scale * torch.mean(loss_fit) + lam * loss_sig
    return loss, {}


def regression_offmanjac_loss(
    config: OptimizationConfig,
    flow_map: FlowMap,
    encoder: BaseEncoder,
    interp: Interpolant,
    act: torch.Tensor,
    obs: torch.Tensor,
    delta_t: torch.Tensor,
) -> float:
    """Regression fit + off-manifold flatness (consistency) penalty:
      L = ||f(s) - a||^2 + lam * ||f(s + delta) - sg(f(s))||^2,  delta ~ N(0, std^2)
    The penalty acts ONLY at perturbed (off-support) states and is anchored at the
    stop-gradient on-manifold prediction: no label convolution (unlike obs-noise),
    no gradient pressure on the on-manifold Jacobian (unlike jacreg). Stochastic
    Tikhonov regularization of J_s restricted to a neighborhood of the support.
    Env: OFFJAC_LAMBDA (default 1.0), OFFJAC_STD (default 0.1, normalized obs units)."""
    import os
    lam = float(os.environ.get("OFFJAC_LAMBDA", "1.0"))
    std = float(os.environ.get("OFFJAC_STD", "0.1"))
    t = torch.zeros_like(delta_t, device=delta_t.device)
    act_0 = torch.zeros_like(act, device=act.device)
    obs_emb = encoder(obs, None)
    act_pred = flow_map.get_velocity(t, act_0, obs_emb)
    loss_fit = get_norm(act_pred - act, config.norm_type, config.cauchy_c)
    if isinstance(obs, dict) or hasattr(obs, "keys"):
        obs_p = {k: v + std * torch.randn_like(v) for k, v in obs.items()}
    else:
        obs_p = obs + std * torch.randn_like(obs)
    act_pred_p = flow_map.get_velocity(t, act_0, encoder(obs_p, None))
    loss_flat = get_norm(act_pred_p - act_pred.detach(), config.norm_type, config.cauchy_c)
    loss = config.loss_scale * (torch.mean(loss_fit) + lam * torch.mean(loss_flat))
    return loss, {}


def regression_obsnoise_loss(
    config: OptimizationConfig,
    flow_map: FlowMap,
    encoder: BaseEncoder,
    interp: Interpolant,
    act: torch.Tensor,
    obs: torch.Tensor,
    delta_t: torch.Tensor,
) -> float:
    """Identical to regression_loss but injects Gaussian noise into the OBS input during
    training (std from env OBS_NOISE_STD, default 0.1, in normalized obs units). A standard
    smoothness regularizer: tests whether making the 1-step function low-gain / smooth in
    obs (the property MIP-step1 has) recovers MIP-like SR -- WITHOUT a denoising head.
    Inference is plain regression (no noise)."""
    import os
    std = float(os.environ.get("OBS_NOISE_STD", "0.1"))
    t = torch.zeros_like(delta_t, device=delta_t.device)
    act_0 = torch.zeros_like(act, device=act.device)
    if isinstance(obs, dict) or hasattr(obs, "keys"):
        obs = {k: v + std * torch.randn_like(v) for k, v in obs.items()}
    else:
        obs = obs + std * torch.randn_like(obs)
    obs_emb = encoder(obs, None)
    act_pred = flow_map.get_velocity(t, act_0, obs_emb)
    loss = get_norm(act_pred - act, config.norm_type, config.cauchy_c)
    loss = config.loss_scale * torch.mean(loss)
    return loss, {}


def regression_manifold_loss(
    config: OptimizationConfig,
    flow_map: FlowMap,
    encoder: BaseEncoder,
    interp: Interpolant,
    act: torch.Tensor,
    obs: torch.Tensor,
    delta_t: torch.Tensor,
) -> float:
    """Regression THROUGH a frozen action manifold. Same as regression_loss, but the
    network output is passed through a frozen denoiser (pretrained on 20k actions)
    before the MSE; gradients backprop THROUGH the frozen manifold so the policy learns
    to land in the right pre-image. Tests whether a separable, data-rich, frozen action
    prior -- baked into TRAINING -- recovers MIP-like robustness."""
    from mip.action_manifold import project
    t = torch.zeros_like(delta_t, device=delta_t.device)
    act_0 = torch.zeros_like(act, device=act.device)
    obs_emb = encoder(obs, None)
    act_pred = flow_map.get_velocity(t, act_0, obs_emb)
    act_proj = project(act_pred, act.device)
    loss = get_norm(act_proj - act, config.norm_type, config.cauchy_c)
    loss = config.loss_scale * torch.mean(loss)
    return loss, {}


def straight_flow_loss(
    config: OptimizationConfig,
    flow_map: FlowMap,
    encoder: BaseEncoder,
    interp: Interpolant,
    act: torch.Tensor,
    obs: torch.Tensor,
    delta_t: torch.Tensor,
) -> float:
    """Straight flow loss."""
    # sample
    t = torch.zeros_like(delta_t, device=delta_t.device)

    # Major difference compared to regression: use random noise instead of zeros
    act_0 = torch.randn_like(act, device=act.device)

    # get condition
    obs_emb = encoder(obs, None)

    # predict
    act_pred = flow_map.get_velocity(t, act_0, obs_emb)

    # compute loss
    loss = get_norm(act_pred - act, config.norm_type, config.cauchy_c)
    loss = config.loss_scale * torch.mean(loss)
    return loss, {}

def tsd_loss(
    config: OptimizationConfig,
    flow_map: FlowMap,
    encoder: BaseEncoder,
    interp: Interpolant,
    act: torch.Tensor,
    obs: torch.Tensor,
    delta_t: torch.Tensor,
) -> float:
    """Two step denoising loss."""
    # sample
    s = torch.zeros_like(delta_t, device=delta_t.device)
    t = torch.zeros_like(delta_t, device=delta_t.device) + config.t_two_step
    act_0 = torch.empty_like(act).normal_(0, 1)
    noise = torch.empty_like(act).normal_(0, 1)
    act_t = act + (1 - config.t_two_step) * noise

    # get condition
    obs_emb = encoder(obs, None)

    # predict
    act_pred_0 = flow_map.get_velocity(s, act_0, obs_emb)
    act_pred_1 = flow_map.get_velocity(t, act_t, obs_emb)

    # compute loss
    loss0 = get_norm((act_pred_0 - act_t) / config.t_two_step, config.norm_type, config.cauchy_c)
    loss1 = get_norm((act_pred_1 - act) / (1 - config.t_two_step), config.norm_type, config.cauchy_c)
    loss = loss0 + loss1
    loss = config.loss_scale * torch.mean(loss)

    return loss, {}


def mip_heterov1_loss(
    config: OptimizationConfig,
    flow_map: FlowMap,
    encoder: BaseEncoder,
    interp: Interpolant,
    act: torch.Tensor,
    obs: torch.Tensor,
    delta_t: torch.Tensor,
) -> float:
    """DESIGNED 80+ CANDIDATE (PART CLXXIV): view-1 = hetero-t NLL (the proven best
    gradient rebalancer, learned per-sample t-scale) + view-2 = standard noise-anchored
    denoising (the measured +8..+20 iteration factor), 2-step inference."""
    t0 = torch.zeros_like(delta_t, device=delta_t.device)
    t1 = torch.zeros_like(delta_t, device=delta_t.device) + config.t_two_step
    act_0 = torch.zeros_like(act, device=act.device)
    noise = torch.empty_like(act).normal_(0, 1)
    act_t = act + (1 - config.t_two_step) * noise
    obs_emb = encoder(obs, None)
    pred0, s_raw = flow_map.net(act_0, t0, t0, obs_emb)
    pred1 = flow_map.get_velocity(t1, act_t, obs_emb)
    r0 = pred0 - act
    sb = torch.nn.functional.softplus(s_raw).reshape(len(act), -1).mean(dim=1) + 1e-3
    nu = config.student_t_df
    D = r0[0].numel()
    z2 = r0.reshape(len(act), -1).pow(2).sum(dim=1) / (sb ** 2 * D)
    loss0 = (0.5 * (nu + 1.0) * torch.log1p(z2 / nu) * D + D * torch.log(sb)) / D
    loss1 = get_norm((pred1 - act) / (1 - config.t_two_step), config.norm_type, config.cauchy_c)
    loss = config.loss_scale * (torch.mean(loss0) + torch.mean(loss1))
    return loss, {}


def mip_cauchyv1_loss(
    config: OptimizationConfig,
    flow_map: FlowMap,
    encoder: BaseEncoder,
    interp: Interpolant,
    act: torch.Tensor,
    obs: torch.Tensor,
    delta_t: torch.Tensor,
) -> float:
    """MIP with bounded-influence (Cauchy) FIRST view; denoising view-2 unchanged.
    Tests the objective-specific-rollback prediction (PART CLVIII): the unbounded L2
    view-1 drives late-training tail-chasing; clipping it should hold the ~78 peak."""
    s = torch.zeros_like(delta_t, device=delta_t.device)
    t = torch.zeros_like(delta_t, device=delta_t.device) + config.t_two_step
    act_0 = torch.zeros_like(act, device=act.device)
    noise = torch.empty_like(act).normal_(0, 1)
    act_t = act + (1 - config.t_two_step) * noise
    obs_emb = encoder(obs, None)
    act_pred_0 = flow_map.get_velocity(s, act_0, obs_emb)
    act_pred_1 = flow_map.get_velocity(t, act_t, obs_emb)
    loss0 = get_norm((act_pred_0 - act) / config.t_two_step, "cauchy", config.cauchy_c)
    loss1 = get_norm((act_pred_1 - act) / (1 - config.t_two_step), config.norm_type, config.cauchy_c)
    loss = loss0 + loss1
    loss = config.loss_scale * torch.mean(loss)
    return loss, {}


def mip_loss(
    config: OptimizationConfig,
    flow_map: FlowMap,
    encoder: BaseEncoder,
    interp: Interpolant,
    act: torch.Tensor,
    obs: torch.Tensor,
    delta_t: torch.Tensor,
) -> float:
    """Minimum iterative policy loss."""
    # sample
    s = torch.zeros_like(delta_t, device=delta_t.device)
    t = torch.zeros_like(delta_t, device=delta_t.device) + config.t_two_step
    # major difference compared to tsd: remove stochasticity in input
    act_0 = torch.zeros_like(act, device=act.device)
    noise = torch.empty_like(act).normal_(0, 1)
    act_t = act + (1 - config.t_two_step) * noise

    # NOTE: in paper, we use
    # act_t = config.t_two_step * act + (1 - config.t_two_step) * noise
    # but we found that this is not necessary when config.t_two_step close to 1.
    # feel free to use the original form if you want to, you can refer to mip_origin_loss

    # get condition
    obs_emb = encoder(obs, None)

    # predict
    # for first step, scale network output by t_two_step to match the scale of the second step
    # equivalent form: directly let first step predict act
    act_pred_0 = flow_map.get_velocity(s, act_0, obs_emb)
    act_pred_1 = flow_map.get_velocity(t, act_t, obs_emb)

    # compute loss
    # difference compared to tsd: no stochasticity in prediction
    loss0 = get_norm((act_pred_0 - act) / config.t_two_step, config.norm_type, config.cauchy_c)
    loss1 = get_norm((act_pred_1 - act) / (1 - config.t_two_step), config.norm_type, config.cauchy_c)
    loss = loss0 + loss1
    loss = config.loss_scale * torch.mean(loss)

    return loss, {}


def _xm_rep(x, k):
    return x.repeat(k, *([1] * (x.dim() - 1)))


def xm_loss(
    config: OptimizationConfig,
    flow_map: FlowMap,
    encoder: BaseEncoder,
    interp: Interpolant,
    act: torch.Tensor,
    obs: torch.Tensor,
    delta_t: torch.Tensor,
) -> float:
    """PURE Forward XM (Gladstone et al.): end-to-end one-step generator.
    The latent z ~ N(0,1) is fed as the network input at t=0; K candidates
    per sample, train on the per-sample best. Inference = one forward with
    a single z draw (xm_sampler). Multimodality is handled by latent
    exploration - no diffusion/flow steps, no anchor, no tail model."""
    k = max(int(config.xm_k), 1)
    b = act.shape[0]
    obs_emb = encoder(obs, None)
    act_rep = _xm_rep(act, k)
    z = torch.empty_like(act_rep).normal_(0, 1)
    t0 = torch.zeros(k * b, device=act.device)
    obs_emb_rep = (
        {kk: _xm_rep(v, k) for kk, v in obs_emb.items()}
        if isinstance(obs_emb, dict)
        else _xm_rep(obs_emb, k)
    )
    pred = flow_map.get_velocity(t0, z, obs_emb_rep)
    per = get_norm(pred - act_rep, config.norm_type, config.cauchy_c)
    n_el = per[0].numel()
    per_sample = per.reshape(k, b, -1).sum(dim=-1)
    best = per_sample.min(dim=0).values / n_el
    loss = config.loss_scale * torch.mean(best)
    return loss, {}


def mip_rw_loss(
    config: OptimizationConfig,
    flow_map: FlowMap,
    encoder: BaseEncoder,
    interp: Interpolant,
    act: torch.Tensor,
    obs: torch.Tensor,
    delta_t: torch.Tensor,
) -> float:
    """MIP-eq: standard mip_loss with per-sample INVERSE distance weighting
    alpha(d) = g_MSE(d)/g_MIP(d) that flattens MIP's gradient profile to MSE's
    (advisor reverse-causal test). RW_ASSETS + RW_PROFILE env as regression_rw."""
    if not _RW_CACHE["loaded"]:
        _rw_load(act.device)
    s_ = torch.zeros_like(delta_t, device=delta_t.device)
    t = torch.zeros_like(delta_t, device=delta_t.device) + config.t_two_step
    act_0 = torch.zeros_like(act, device=act.device)
    noise = torch.empty_like(act).normal_(0, 1)
    act_t = act + (1 - config.t_two_step) * noise
    with torch.no_grad():
        eef = obs[:, -1, 44:47] * _RW_CACHE["A"] + _RW_CACHE["B"]
        diff = eef[:, None, :] - _RW_CACHE["C"][None]
        d = (diff.norm(dim=2) / _RW_CACHE["SG"][None]).min(dim=1).values
        w = _RW_CACHE["w"][torch.bucketize(d, _RW_CACHE["edges"])]
    obs_emb = encoder(obs, None)
    act_pred_0 = flow_map.get_velocity(s_, act_0, obs_emb)
    act_pred_1 = flow_map.get_velocity(t, act_t, obs_emb)
    loss0 = get_norm((act_pred_0 - act) / config.t_two_step, config.norm_type, config.cauchy_c)
    loss1 = get_norm((act_pred_1 - act) / (1 - config.t_two_step), config.norm_type, config.cauchy_c)
    tot = loss0 + loss1
    per_sample = tot.mean(dim=tuple(range(1, tot.dim()))) if tot.dim() > 1 else tot
    loss = config.loss_scale * torch.mean(w * per_sample)
    return loss, {}


def _mip_two_view(config, flow_map, encoder, act, obs, delta_t, act_t, aux_target, detach_aux_emb=False):
    s_ = torch.zeros_like(delta_t, device=delta_t.device)
    t = torch.zeros_like(delta_t, device=delta_t.device) + config.t_two_step
    act_0 = torch.zeros_like(act, device=act.device)
    obs_emb = encoder(obs, None)
    act_pred_0 = flow_map.get_velocity(s_, act_0, obs_emb)
    emb_aux = obs_emb.detach() if detach_aux_emb else obs_emb
    act_pred_1 = flow_map.get_velocity(t, act_t, emb_aux)
    loss0 = get_norm((act_pred_0 - act) / config.t_two_step, config.norm_type, config.cauchy_c)
    loss1 = get_norm((act_pred_1 - aux_target) / (1 - config.t_two_step), config.norm_type, config.cauchy_c)
    return config.loss_scale * torch.mean(loss0 + loss1), {}


def mip_distill2_loss(config, flow_map, encoder, interp, act, obs, delta_t):
    """TWO-NETWORK co-train distill: denoiser = reference_net (fits data),
    step1 = main net (fits sg(denoiser output)). Shared encoder, separate
    action decoders. Requires MIP_TWONET=1."""
    t = torch.zeros_like(delta_t, device=delta_t.device) + config.t_two_step
    s_ = torch.zeros_like(delta_t, device=delta_t.device)
    noise = torch.empty_like(act).normal_(0, 1)
    act_t = act + (1 - config.t_two_step) * noise
    act_0 = torch.zeros_like(act, device=act.device)
    obs_emb = encoder(obs, None)
    y1 = flow_map.get_reference_velocity(t, act_t, obs_emb)
    y0 = flow_map.get_velocity(s_, act_0, obs_emb)
    loss1 = get_norm((y1 - act) / (1 - config.t_two_step), config.norm_type, config.cauchy_c)
    loss0 = get_norm((y0 - y1.detach()) / config.t_two_step, config.norm_type, config.cauchy_c)
    loss = config.loss_scale * torch.mean(loss0 + loss1)
    return loss, {}


def mip_lambda_loss(config, flow_map, encoder, interp, act, obs, delta_t):
    """lambda sweep: loss = ||(y0-a)/t||^2 + (lam/81)*||(y1-a)/(1-t)||^2.
    lam=81 == standard mip_loss; lam=0 == pure step1 (MSE-equivalent up to scale).
    MIPLAMBDA env."""
    import os as _os
    lam = float(_os.environ.get("MIPLAMBDA", "81"))
    s_ = torch.zeros_like(delta_t, device=delta_t.device)
    t = torch.zeros_like(delta_t, device=delta_t.device) + config.t_two_step
    act_0 = torch.zeros_like(act, device=act.device)
    noise = torch.empty_like(act).normal_(0, 1)
    act_t = act + (1 - config.t_two_step) * noise
    obs_emb = encoder(obs, None)
    y0 = flow_map.get_velocity(s_, act_0, obs_emb)
    y1 = flow_map.get_velocity(t, act_t, obs_emb)
    loss0 = get_norm((y0 - act) / config.t_two_step, config.norm_type, config.cauchy_c)
    loss1 = get_norm((y1 - act) / (1 - config.t_two_step), config.norm_type, config.cauchy_c)
    loss = config.loss_scale * torch.mean(loss0 + (lam / 81.0) * loss1)
    return loss, {}


def mip_distill_loss(config, flow_map, encoder, interp, act, obs, delta_t):
    """Decomposition test: y1 = denoiser fits DATA; y0 = f(s,0,0) fits sg(y1)
    (step-1 action sourced purely by imitating the denoising slice)."""
    t = torch.zeros_like(delta_t, device=delta_t.device) + config.t_two_step
    s_ = torch.zeros_like(delta_t, device=delta_t.device)
    noise = torch.empty_like(act).normal_(0, 1)
    act_t = act + (1 - config.t_two_step) * noise
    act_0 = torch.zeros_like(act, device=act.device)
    obs_emb = encoder(obs, None)
    y1 = flow_map.get_velocity(t, act_t, obs_emb)
    y0 = flow_map.get_velocity(s_, act_0, obs_emb)
    loss1 = get_norm((y1 - act) / (1 - config.t_two_step), config.norm_type, config.cauchy_c)
    loss0 = get_norm((y0 - y1.detach()) / config.t_two_step, config.norm_type, config.cauchy_c)
    loss = config.loss_scale * torch.mean(loss0 + loss1)
    return loss, {}


def mip_nonoise_loss(config, flow_map, encoder, interp, act, obs, delta_t):
    """EXP-B: aux slice WITHOUT noise — aux input = clean action itself."""
    return _mip_two_view(config, flow_map, encoder, act, obs, delta_t,
                         act_t=act, aux_target=act)


def mip_shufx_loss(config, flow_map, encoder, interp, act, obs, delta_t):
    """H5 discriminator: standard MIP but the AUX view's conditioning obs is
    SHUFFLED within batch — denoiser trained with wrong state context. If
    trunk-smoothing is the active ingredient, benefit persists; if the
    privileged x-conditioned decomposition matters, it collapses."""
    t = torch.zeros_like(delta_t, device=delta_t.device) + config.t_two_step
    s_ = torch.zeros_like(delta_t, device=delta_t.device)
    noise = torch.empty_like(act).normal_(0, 1)
    act_t = act + (1 - config.t_two_step) * noise
    act_0 = torch.zeros_like(act, device=act.device)
    obs_emb = encoder(obs, None)
    perm = torch.randperm(len(act), device=act.device)
    act_pred_0 = flow_map.get_velocity(s_, act_0, obs_emb)
    act_pred_1 = flow_map.get_velocity(t, act_t, obs_emb[perm])
    loss0 = get_norm((act_pred_0 - act) / config.t_two_step,
                     config.norm_type, config.cauchy_c)
    loss1 = get_norm((act_pred_1 - act) / (1 - config.t_two_step),
                     config.norm_type, config.cauchy_c)
    return config.loss_scale * torch.mean(loss0 + loss1), {}


def mip_nonoise_atk_loss(config, flow_map, encoder, interp, act, obs, delta_t):
    """ANCHOR-TIKHONOV: nonoise two-view + lam * FD estimate of the aux
    view's anchor sensitivity ||df/d(act_t)||^2 at scale ATK_EPS — the
    first-order (Bishop) deterministic equivalent of MIP's anchor noise.
    Envs: ATK_LAM (default 0.1), ATK_EPS (default 0.1)."""
    import os as _os

    lam = float(_os.environ.get("ATK_LAM", "0.1"))
    eps = float(_os.environ.get("ATK_EPS", "0.1"))
    base, info = _mip_two_view(config, flow_map, encoder, act, obs, delta_t,
                               act_t=act, aux_target=act)
    t = torch.zeros_like(delta_t, device=delta_t.device) + config.t_two_step
    obs_emb = encoder(obs, None)
    u = torch.randn_like(act)
    u = u / (u.reshape(len(u), -1).norm(dim=1).view(-1, 1, 1) + 1e-9)
    p0 = flow_map.get_velocity(t, act, obs_emb)
    p1 = flow_map.get_velocity(t, act + eps * u, obs_emb)
    pen = ((p1 - p0) / eps) ** 2
    return base + lam * pen.mean(), info


def _aux_nhat(obs, act):
    """per-sample tangent-orthogonalized outward normal (raw space) + raw pos chunk."""
    R = _RW_CACHE
    eef = obs[:, -1, 44:47] * R["A"] + R["B"]
    diff = eef[:, None, :] - R["C"][None]
    kidx = torch.argmin(diff.norm(dim=2), dim=1)
    dv = eef - R["C"][kidx]
    tg = R["TG"][kidx]
    dvp = dv - (dv * tg).sum(1, keepdim=True) * tg
    nhat = dvp / (dvp.norm(dim=1, keepdim=True) + 1e-9)          # (B,3)
    a_pos_raw = act[:, :, :3] * R["Aa"][:3] + R["Ba"][:3]        # (B,T,3)
    return nhat, a_pos_raw


def _repack(act, a_pos_new):
    R = _RW_CACHE
    tgt = act.clone()
    tgt[:, :, :3] = (a_pos_new - R["Ba"][:3]) / R["Aa"][:3]
    return tgt


def mip_auxtan_loss(config, flow_map, encoder, interp, act, obs, delta_t):
    """EXP-C1: aux TARGET stripped of its inward/outward normal component."""
    if not _RW_CACHE["loaded"]:
        _rw_load(act.device)
    with torch.no_grad():
        nhat, a_pos_raw = _aux_nhat(obs, act)
        comp = (a_pos_raw * nhat[:, None, :]).sum(-1, keepdim=True)
        tgt = _repack(act, a_pos_raw - comp * nhat[:, None, :])
    noise = torch.empty_like(act).normal_(0, 1)
    act_t = act + (1 - config.t_two_step) * noise
    return _mip_two_view(config, flow_map, encoder, act, obs, delta_t,
                         act_t=act_t, aux_target=tgt)


def mip_auxflip_loss(config, flow_map, encoder, interp, act, obs, delta_t):
    """EXP-C2: aux TARGET with inward component sign-FLIPPED."""
    if not _RW_CACHE["loaded"]:
        _rw_load(act.device)
    with torch.no_grad():
        nhat, a_pos_raw = _aux_nhat(obs, act)
        comp = (a_pos_raw * nhat[:, None, :]).sum(-1, keepdim=True)
        tgt = _repack(act, a_pos_raw - 2.0 * comp * nhat[:, None, :])
    noise = torch.empty_like(act).normal_(0, 1)
    act_t = act + (1 - config.t_two_step) * noise
    return _mip_two_view(config, flow_map, encoder, act, obs, delta_t,
                         act_t=act_t, aux_target=tgt)


def mip_auxdetach_loss(config, flow_map, encoder, interp, act, obs, delta_t):
    """EXP-D: stop-gradient from the aux slice into the state encoder."""
    noise = torch.empty_like(act).normal_(0, 1)
    act_t = act + (1 - config.t_two_step) * noise
    return _mip_two_view(config, flow_map, encoder, act, obs, delta_t,
                         act_t=act_t, aux_target=act, detach_aux_emb=True)


def mip_auxdet_loss(
    config: OptimizationConfig,
    flow_map: FlowMap,
    encoder: BaseEncoder,
    interp: Interpolant,
    act: torch.Tensor,
    obs: torch.Tensor,
    delta_t: torch.Tensor,
) -> float:
    """EXP-A1 variant 2 (AUX_GRAD_BLOCK): identical views/weights to mip_loss, but the
    auxiliary (tau-slice) loss updates ONLY the final_conv late branch — its gradients are
    blocked from the shared trunk (encoder + all UNet blocks) by temporarily setting
    requires_grad=False on those params and calling backward() here. The main (t=0) view
    trains everything as usual via the returned loss. Tests whether aux gradients into the
    shared state trunk are necessary for MIP-step1's corrective features."""
    s = torch.zeros_like(delta_t, device=delta_t.device)
    t = torch.zeros_like(delta_t, device=delta_t.device) + config.t_two_step
    act_0 = torch.zeros_like(act, device=act.device)
    noise = torch.empty_like(act).normal_(0, 1)
    act_t = act + (1 - config.t_two_step) * noise

    # ---- aux view: gradients restricted to final_conv ----
    fc = None
    for name, m in flow_map.named_modules():
        if name.endswith("final_conv"):
            fc = m
    import os as _os
    if _os.environ.get("FT_REINIT_HEAD") and not _FT_STATE.get("reinit_done"):
        # fresh MSE head on the frozen (warm-started) trunk — headmse configuration
        for m in fc.modules():
            if hasattr(m, "reset_parameters"):
                m.reset_parameters()
        _FT_STATE["reinit_done"] = True
    keep = {id(p) for p in fc.parameters()}
    flipped = [p for p in list(flow_map.parameters()) + list(encoder.parameters())
               if id(p) not in keep and p.requires_grad]
    for p in flipped:
        p.requires_grad_(False)
    obs_emb_aux = encoder(obs, None)
    act_pred_1 = flow_map.get_velocity(t, act_t, obs_emb_aux)
    loss1 = get_norm((act_pred_1 - act) / (1 - config.t_two_step), config.norm_type, config.cauchy_c)
    loss1 = config.loss_scale * torch.mean(loss1)
    loss1.backward()
    for p in flipped:
        p.requires_grad_(True)

    # ---- main view: normal ----
    obs_emb = encoder(obs, None)
    act_pred_0 = flow_map.get_velocity(s, act_0, obs_emb)
    loss0 = get_norm((act_pred_0 - act) / config.t_two_step, config.norm_type, config.cauchy_c)
    loss = config.loss_scale * torch.mean(loss0) + loss1.detach()

    return loss, {}


def mip_zeroaux_loss(
    config: OptimizationConfig,
    flow_map: FlowMap,
    encoder: BaseEncoder,
    interp: Interpolant,
    act: torch.Tensor,
    obs: torch.Tensor,
    delta_t: torch.Tensor,
) -> float:
    """EXP-B2 (TWO_VIEW_ZERO_WEIGHT_SWEEP): head0 = f(0,0,s)->a*, aux = f(tau,0,s)->a*
    (ZEROS action input in BOTH views: pure slice/weight factorial, no tube semantics).
    Aux weight w from env AUXW (default 100). Plain unscaled MSE terms:
    L = mean|f0-a|^2 + w * mean|ftau-a|^2."""
    import os as _os
    w = float(_os.environ.get("AUXW", "100"))
    s = torch.zeros_like(delta_t, device=delta_t.device)
    t = torch.zeros_like(delta_t, device=delta_t.device) + config.t_two_step
    act_0 = torch.zeros_like(act, device=act.device)

    obs_emb = encoder(obs, None)
    act_pred_0 = flow_map.get_velocity(s, act_0, obs_emb)
    act_pred_1 = flow_map.get_velocity(t, act_0, obs_emb)
    loss0 = get_norm(act_pred_0 - act, config.norm_type, config.cauchy_c)
    loss1 = get_norm(act_pred_1 - act, config.norm_type, config.cauchy_c)
    loss = config.loss_scale * (torch.mean(loss0) + w * torch.mean(loss1))

    return loss, {}


_FT_STATE = {}


def regression_frozentrunk_loss(
    config: OptimizationConfig,
    flow_map: FlowMap,
    encoder: BaseEncoder,
    interp: Interpolant,
    act: torch.Tensor,
    obs: torch.Tensor,
    delta_t: torch.Tensor,
) -> float:
    """CHECK-4 lazy/NTK control: plain regression, but ONLY final_conv trains — the trunk
    (encoder + all UNet blocks) stays at its random initialization (requires_grad=False,
    set once on first call). SGD-trained head on frozen random features: does preserving
    the random kernel preserve its corrective extension?"""
    fc = None
    for name, m in flow_map.named_modules():
        if name.endswith("final_conv"):
            fc = m
    import os as _os
    if _os.environ.get("FT_REINIT_HEAD") and not _FT_STATE.get("reinit_done"):
        # fresh MSE head on the frozen (warm-started) trunk — headmse configuration
        for m in fc.modules():
            if hasattr(m, "reset_parameters"):
                m.reset_parameters()
        _FT_STATE["reinit_done"] = True
    keep = {id(p) for p in fc.parameters()}
    for p in list(flow_map.parameters()) + list(encoder.parameters()):
        if id(p) not in keep and p.requires_grad:
            p.requires_grad_(False)
    s = torch.zeros_like(delta_t, device=delta_t.device)
    act_0 = torch.zeros_like(act, device=act.device)
    obs_emb = encoder(obs, None)
    act_pred = flow_map.get_velocity(s, act_0, obs_emb)
    loss = config.loss_scale * torch.mean(get_norm(act_pred - act, config.norm_type, config.cauchy_c))
    return loss, {}


_TEACHER_CACHE = {}


def _get_teacher(flow_map, encoder):
    """Frozen deepcopy of the nets at FIRST loss call. With INIT_CKPT warm-start this
    is exactly the starting checkpoint (e.g. MSE s10k)."""
    if "fm" not in _TEACHER_CACHE:
        import copy
        _TEACHER_CACHE["fm"] = copy.deepcopy(flow_map).eval()
        _TEACHER_CACHE["enc"] = copy.deepcopy(encoder).eval()
        for p in _TEACHER_CACHE["fm"].parameters():
            p.requires_grad_(False)
        for p in _TEACHER_CACHE["enc"].parameters():
            p.requires_grad_(False)
    return _TEACHER_CACHE["fm"], _TEACHER_CACHE["enc"]


def regression_tauteacher_loss(
    config: OptimizationConfig,
    flow_map: FlowMap,
    encoder: BaseEncoder,
    interp: Interpolant,
    act: torch.Tensor,
    obs: torch.Tensor,
    delta_t: torch.Tensor,
) -> float:
    """CLOSURE B1: L_MSE(t=0) + beta * ||f(v,0,s) - stopgrad(f_s10k(v,0,s))||^2.
    Env: BETA (default 100), TVIEW ('tau' -> t_two_step, 't0' -> 0 for the control)."""
    import os as _os
    beta = float(_os.environ.get("BETA", "100"))
    tview = _os.environ.get("TVIEW", "tau")
    tfm, tenc = _get_teacher(flow_map, encoder)
    s = torch.zeros_like(delta_t, device=delta_t.device)
    tv = s if tview == "t0" else s + config.t_two_step
    act_0 = torch.zeros_like(act, device=act.device)
    obs_emb = encoder(obs, None)
    pred0 = flow_map.get_velocity(s, act_0, obs_emb)
    loss0 = torch.mean(get_norm(pred0 - act, config.norm_type, config.cauchy_c))
    pred_v = flow_map.get_velocity(tv, act_0, obs_emb)
    with torch.no_grad():
        temb = tenc(obs, None)
        tgt_v = tfm.get_velocity(tv, act_0, temb)
    loss_anchor = torch.mean(get_norm(pred_v - tgt_v, config.norm_type, config.cauchy_c))
    return config.loss_scale * (loss0 + beta * loss_anchor), {}


def regression_taufeat_loss(
    config: OptimizationConfig,
    flow_map: FlowMap,
    encoder: BaseEncoder,
    interp: Interpolant,
    act: torch.Tensor,
    obs: torch.Tensor,
    delta_t: torch.Tensor,
) -> float:
    """CLOSURE B2: L_MSE + beta * ||P phi(tau,0,s) - stopgrad(P phi_s10k(tau,0,s))||^2
    on the penultimate feature (input of final_conv's last layer). P = identity (env FEATP
    reserved for subspace variants). Env: BETA."""
    import os as _os
    beta = float(_os.environ.get("BETA", "10"))
    tfm, tenc = _get_teacher(flow_map, encoder)
    if "hooks" not in _TEACHER_CACHE:
        buf = {}
        def find_last(net):
            fc = None
            for nm, m in net.named_modules():
                if nm.endswith("final_conv"):
                    fc = m
            return list(fc.children())[-1]
        find_last(flow_map).register_forward_pre_hook(
            lambda mod, inp: buf.__setitem__("s", inp[0]))
        find_last(tfm).register_forward_pre_hook(
            lambda mod, inp: buf.__setitem__("t", inp[0]))
        _TEACHER_CACHE["hooks"] = buf
    buf = _TEACHER_CACHE["hooks"]
    s = torch.zeros_like(delta_t, device=delta_t.device)
    tv = s + config.t_two_step
    act_0 = torch.zeros_like(act, device=act.device)
    obs_emb = encoder(obs, None)
    pred0 = flow_map.get_velocity(s, act_0, obs_emb)
    loss0 = torch.mean(get_norm(pred0 - act, config.norm_type, config.cauchy_c))
    flow_map.get_velocity(tv, act_0, obs_emb)
    f_s = buf["s"]
    with torch.no_grad():
        temb = tenc(obs, None)
        tfm.get_velocity(tv, act_0, temb)
        f_t = buf["t"]
    loss_feat = torch.mean((f_s - f_t) ** 2)
    return config.loss_scale * (loss0 + beta * loss_feat), {}


def regression_badpen_loss(
    config: OptimizationConfig,
    flow_map: FlowMap,
    encoder: BaseEncoder,
    interp: Interpolant,
    act: torch.Tensor,
    obs: torch.Tensor,
    delta_t: torch.Tensor,
) -> float:
    """CLOSURE B3: L_MSE + lam_pos*ReLU(gain@bad1)^2 + lam_rot*||drot response @ v_mseamp||^2
    via finite differences in NORMALIZED obs space. Env: LAMPOS, LAMROT, BADDIR_NPZ
    (contains dpos_dir, drot_dir: normalized-obs-space perturbations; bad1: 3d output dir),
    EPSFD (default 0.1)."""
    import os as _os
    lam_p = float(_os.environ.get("LAMPOS", "1e-2"))
    lam_r = float(_os.environ.get("LAMROT", "1e-2"))
    eps = float(_os.environ.get("EPSFD", "0.1"))
    if "baddirs" not in _TEACHER_CACHE:
        import numpy as _np
        z = _np.load(_os.environ["BADDIR_NPZ"])
        dev = act.device
        _TEACHER_CACHE["baddirs"] = (
            torch.tensor(z["dpos_dir"], dtype=torch.float32, device=dev),
            torch.tensor(z["drot_dir"], dtype=torch.float32, device=dev),
            torch.tensor(z["bad1"], dtype=torch.float32, device=dev))
    dpos, drot, bad1 = _TEACHER_CACHE["baddirs"]
    s = torch.zeros_like(delta_t, device=delta_t.device)
    act_0 = torch.zeros_like(act, device=act.device)
    obs_emb = encoder(obs, None)
    pred0 = flow_map.get_velocity(s, act_0, obs_emb)
    loss0 = torch.mean(get_norm(pred0 - act, config.norm_type, config.cauchy_c))
    st = obs["state"] if isinstance(obs, dict) else obs
    def resp(direction):
        pert = st + eps * direction[None, None, :]
        ob2 = {"state": pert} if isinstance(obs, dict) else pert
        emb2 = encoder(ob2, None)
        return (flow_map.get_velocity(s, act_0, emb2) - pred0) / eps
    r_p = resp(dpos)   # (B, H, 10)
    gain = torch.einsum("bhk,k->bh", r_p[:, :, :3], bad1).mean(1)
    loss_pos = torch.mean(torch.relu(gain) ** 2)
    r_r = resp(drot)
    loss_rot = torch.mean(torch.sum(r_r[:, :, 3:6] ** 2, dim=-1))
    return config.loss_scale * (loss0 + lam_p * loss_pos + lam_r * loss_rot), {}


def regression_opanchor_loss(
    config: OptimizationConfig,
    flow_map: FlowMap,
    encoder: BaseEncoder,
    interp: Interpolant,
    act: torch.Tensor,
    obs: torch.Tensor,
    delta_t: torch.Tensor,
) -> float:
    """CLOSURE B4 (oracle upper bound): L_MSE + gamma * || (f(s')-f(s0)) -
    stopgrad(f_s10k(s')-f_s10k(s0)) ||^2 on pre-dumped off-support pairs (NORMALIZED obs
    windows). Anchors the RESPONSE OPERATOR to the s10k one; no GT recovery labels.
    Env: GAMMA, PAIRS_NPZ (w0: (N,2,53), ws: (N,2,53) normalized windows), KPAIRS."""
    import os as _os
    gamma = float(_os.environ.get("GAMMA", "1e-1"))
    K = int(_os.environ.get("KPAIRS", "16"))
    tfm, tenc = _get_teacher(flow_map, encoder)
    if "pairs" not in _TEACHER_CACHE:
        import numpy as _np
        z = _np.load(_os.environ["PAIRS_NPZ"])
        dev = act.device
        _TEACHER_CACHE["pairs"] = (
            torch.tensor(z["w0"], dtype=torch.float32, device=dev),
            torch.tensor(z["ws"], dtype=torch.float32, device=dev))
    W0, WS = _TEACHER_CACHE["pairs"]
    s = torch.zeros_like(delta_t, device=delta_t.device)
    act_0 = torch.zeros_like(act, device=act.device)
    obs_emb = encoder(obs, None)
    pred0 = flow_map.get_velocity(s, act_0, obs_emb)
    loss0 = torch.mean(get_norm(pred0 - act, config.norm_type, config.cauchy_c))
    idx = torch.randint(0, len(W0), (K,), device=act.device)
    sk = torch.zeros(2 * K, device=act.device)
    ak = torch.zeros((2 * K, act.shape[1], act.shape[2]), device=act.device)
    stacked = {"state": torch.cat([W0[idx], WS[idx]], 0)}
    emb = encoder(stacked, None)
    out = flow_map.get_velocity(sk, ak, emb)
    dr = out[K:] - out[:K]
    with torch.no_grad():
        temb = tenc(stacked, None)
        tout = tfm.get_velocity(sk, ak, temb)
        tdr = tout[K:] - tout[:K]
    loss_op = torch.mean((dr - tdr) ** 2)
    return config.loss_scale * (loss0 + gamma * loss_op), {}


def mip_tubeaux_loss(
    config: OptimizationConfig,
    flow_map: FlowMap,
    encoder: BaseEncoder,
    interp: Interpolant,
    act: torch.Tensor,
    obs: torch.Tensor,
    delta_t: torch.Tensor,
) -> float:
    """CLOSURE V2 (TAU-TUBE-TARGET): L0 + w * ||f(0.9, a*+0.1eps, s) - a*||^2 with a PLAIN
    weight w (env AUXW): MIP's tube aux view targeting a*, without 1/tau^2 scaling."""
    import os as _os
    w = float(_os.environ.get("AUXW", "100"))
    s = torch.zeros_like(delta_t, device=delta_t.device)
    t = torch.zeros_like(delta_t, device=delta_t.device) + config.t_two_step
    act_0 = torch.zeros_like(act, device=act.device)
    noise = torch.empty_like(act).normal_(0, 1)
    act_t = act + (1 - config.t_two_step) * noise
    obs_emb = encoder(obs, None)
    pred0 = flow_map.get_velocity(s, act_0, obs_emb)
    pred1 = flow_map.get_velocity(t, act_t, obs_emb)
    loss = torch.mean(get_norm(pred0 - act, config.norm_type, config.cauchy_c)) + \
        w * torch.mean(get_norm(pred1 - act, config.norm_type, config.cauchy_c))
    return config.loss_scale * loss, {}


def mip_scramaux_loss(
    config: OptimizationConfig,
    flow_map: FlowMap,
    encoder: BaseEncoder,
    interp: Interpolant,
    act: torch.Tensor,
    obs: torch.Tensor,
    delta_t: torch.Tensor,
) -> float:
    """CLOSURE V3 (TAU-SCRAMBLE-TARGET): tube aux input in FIXED orthogonally scrambled
    action coordinates (seed-0 Q, per-step 10-dim), plain weight w (env AUXW)."""
    import os as _os
    w = float(_os.environ.get("AUXW", "100"))
    if "scramQ" not in _TEACHER_CACHE:
        q, _ = np.linalg.qr(np.random.RandomState(0).randn(10, 10))
        _TEACHER_CACHE["scramQ"] = torch.tensor(q, dtype=torch.float32, device=act.device)
    Q = _TEACHER_CACHE["scramQ"]
    s = torch.zeros_like(delta_t, device=delta_t.device)
    t = torch.zeros_like(delta_t, device=delta_t.device) + config.t_two_step
    act_0 = torch.zeros_like(act, device=act.device)
    noise = torch.empty_like(act).normal_(0, 1)
    act_t = (act + (1 - config.t_two_step) * noise) @ Q.T
    obs_emb = encoder(obs, None)
    pred0 = flow_map.get_velocity(s, act_0, obs_emb)
    pred1 = flow_map.get_velocity(t, act_t, obs_emb)
    loss = torch.mean(get_norm(pred0 - act, config.norm_type, config.cauchy_c)) + \
        w * torch.mean(get_norm(pred1 - act, config.norm_type, config.cauchy_c))
    return config.loss_scale * loss, {}


def mip_randaux_loss(
    config: OptimizationConfig,
    flow_map: FlowMap,
    encoder: BaseEncoder,
    interp: Interpolant,
    act: torch.Tensor,
    obs: torch.Tensor,
    delta_t: torch.Tensor,
) -> float:
    """CLOSURE V5 (TAU-RANDN-TARGET): L0 + w * ||f(0.9, eps, s) - a*||^2 with PURE noise
    aux input eps~N(0,I) (carries NO information about a*), plain weight w (env AUXW).
    Strongest-form test of the view-nondegeneracy hypothesis."""
    import os as _os
    w = float(_os.environ.get("AUXW", "100"))
    s_t = torch.zeros_like(delta_t, device=delta_t.device)
    t = torch.zeros_like(delta_t, device=delta_t.device) + config.t_two_step
    act_0 = torch.zeros_like(act, device=act.device)
    act_t = torch.empty_like(act).normal_(0, 1)
    obs_emb = encoder(obs, None)
    pred0 = flow_map.get_velocity(s_t, act_0, obs_emb)
    pred1 = flow_map.get_velocity(t, act_t, obs_emb)
    loss = torch.mean(get_norm(pred0 - act, config.norm_type, config.cauchy_c)) + \
        w * torch.mean(get_norm(pred1 - act, config.norm_type, config.cauchy_c))
    return config.loss_scale * loss, {}


def regression_cvu_loss(
    config: OptimizationConfig,
    flow_map: FlowMap,
    encoder: BaseEncoder,
    interp: Interpolant,
    act: torch.Tensor,
    obs: torch.Tensor,
    delta_t: torch.Tensor,
) -> float:
    """CLOSURE CVU: cross-view update-consistency regularizer. NO tau-view target.
      L = ||f(0,0,s) - a*||^2 + lam * || (f(tau,u,s)-f0(tau,u,s)) - (f(0,0,s)-f0(0,0,s)) ||^2
    f0 = frozen start checkpoint (deepcopy at first call; INIT_CKPT warm start).
    Env: LAMCVU (default 100), AUXIN in {tube, scramble, random, zero}."""
    import os as _os
    lam = float(_os.environ.get("LAMCVU", "100"))
    auxin = _os.environ.get("AUXIN", "tube")
    tfm, tenc = _get_teacher(flow_map, encoder)
    s_t = torch.zeros_like(delta_t, device=delta_t.device)
    t = torch.zeros_like(delta_t, device=delta_t.device) + config.t_two_step
    act_0 = torch.zeros_like(act, device=act.device)
    if auxin == "tube":
        u = act + 0.1 * torch.empty_like(act).normal_(0, 1)
    elif auxin == "scramble":
        if "scramQ" not in _TEACHER_CACHE:
            q, _ = np.linalg.qr(np.random.RandomState(0).randn(10, 10))
            _TEACHER_CACHE["scramQ"] = torch.tensor(q, dtype=torch.float32, device=act.device)
        u = (act + 0.1 * torch.empty_like(act).normal_(0, 1)) @ _TEACHER_CACHE["scramQ"].T
    elif auxin == "random":
        u = torch.empty_like(act).normal_(0, 1)
    else:
        u = act_0
    obs_emb = encoder(obs, None)
    y0 = flow_map.get_velocity(s_t, act_0, obs_emb)
    yt = flow_map.get_velocity(t, u, obs_emb)
    with torch.no_grad():
        temb = tenc(obs, None)
        y0b = tfm.get_velocity(s_t, act_0, temb)
        ytb = tfm.get_velocity(t, u, temb)
    loss0 = torch.mean(get_norm(y0 - act, config.norm_type, config.cauchy_c))
    r_cvu = torch.mean(get_norm((yt - ytb) - (y0 - y0b), config.norm_type, config.cauchy_c))
    return config.loss_scale * (loss0 + lam * r_cvu), {}


def regression_recgeo_loss(
    config: OptimizationConfig,
    flow_map: FlowMap,
    encoder: BaseEncoder,
    interp: Interpolant,
    act: torch.Tensor,
    obs: torch.Tensor,
    delta_t: torch.Tensor,
) -> float:
    """RECOVERY-GEOMETRY sufficiency: L_MSE + lam * || (f(s)-f(s0)) - target_delta ||^2
    on pre-dumped puredart pairs; target_delta from env REFG in {gt, mip, s10k}
    (native normalized 10d at the executed slot). NO tau view, NO aux target.
    Env: LAMREC, REFG, BAND in {near, rec, mixed}, KPAIRS, RECGEO_NPZ."""
    import os as _os
    lam = float(_os.environ.get("LAMREC", "10"))
    refg = _os.environ.get("REFG", "gt")
    band = _os.environ.get("BAND", "mixed")
    K = int(_os.environ.get("KPAIRS", "16"))
    if "recgeo" not in _TEACHER_CACHE:
        z = np.load(_os.environ.get("RECGEO_NPZ", "logs/recgeo_assets.npz"))
        dev = act.device
        b = z["band"]
        if band == "near":
            m = b == 0
        elif band == "rec":
            m = b == 1
        else:
            m = np.ones(len(b), dtype=bool)
        _TEACHER_CACHE["recgeo"] = (
            torch.tensor(z["w0"][m], dtype=torch.float32, device=dev),
            torch.tensor(z["ws"][m], dtype=torch.float32, device=dev),
            torch.tensor(z["tgt_" + refg][m], dtype=torch.float32, device=dev))
    W0, WS, TGT = _TEACHER_CACHE["recgeo"]
    s_t = torch.zeros_like(delta_t, device=delta_t.device)
    act_0 = torch.zeros_like(act, device=act.device)
    obs_emb = encoder(obs, None)
    pred0 = flow_map.get_velocity(s_t, act_0, obs_emb)
    loss0 = torch.mean(get_norm(pred0 - act, config.norm_type, config.cauchy_c))
    idx = torch.randint(0, len(W0), (K,), device=act.device)
    sk = torch.zeros(2 * K, device=act.device)
    ak = torch.zeros((2 * K, act.shape[1], act.shape[2]), device=act.device)
    emb = encoder({"state": torch.cat([W0[idx], WS[idx]], 0)} if isinstance(obs, dict)
                  else torch.cat([W0[idx], WS[idx]], 0), None)
    out = flow_map.get_velocity(sk, ak, emb)
    START = 1
    df = out[K:, START] - out[:K, START]
    l_rec = torch.mean(torch.sum((df - TGT[idx]) ** 2, dim=-1))
    return config.loss_scale * (loss0 + lam * l_rec), {}


def regression_pdprior_loss(
    config: OptimizationConfig,
    flow_map: FlowMap,
    encoder: BaseEncoder,
    interp: Interpolant,
    act: torch.Tensor,
    obs: torch.Tensor,
    delta_t: torch.Tensor,
) -> float:
    """MSE + analytic attraction-field prior (v2).
    Env: KZ, LAMPD, PDBAND {near,rec,mixed}, PDH (number of executed slots constrained,
    1..15; 16 = all), PDTGT {iso, gt, gtdamped}, PDALPHA (damping for gtdamped),
    PD_NPZ (scales + ggt_pos)."""
    import os as _os
    kz = float(_os.environ.get("KZ", "0.03"))
    lam = float(_os.environ.get("LAMPD", "100"))
    band = _os.environ.get("PDBAND", "mixed")
    pdh = int(_os.environ.get("PDH", "16"))
    tgtmode = _os.environ.get("PDTGT", "iso")
    alpha = float(_os.environ.get("PDALPHA", "1.0"))
    if "pd" not in _TEACHER_CACHE:
        z = np.load(_os.environ.get("PD_NPZ", "logs/pd_assets.npz"))
        dev = act.device
        _TEACHER_CACHE["pd"] = (
            torch.tensor(z["obs_scale"], dtype=torch.float32, device=dev),
            torch.tensor(z["act_scale"], dtype=torch.float32, device=dev),
            torch.tensor(z["ggt_pos"], dtype=torch.float32, device=dev)
            if "ggt_pos" in z.files else None)
    obs_scale, act_scale, ggt_pos = _TEACHER_CACHE["pd"]
    st = obs["state"] if isinstance(obs, dict) else obs
    B = st.shape[0]
    s_t = torch.zeros_like(delta_t, device=delta_t.device)
    act_0 = torch.zeros_like(act, device=act.device)
    obs_emb = encoder(obs, None)
    pred0 = flow_map.get_velocity(s_t, act_0, obs_emb)
    loss0 = torch.mean(get_norm(pred0 - act, config.norm_type, config.cauchy_c))
    u = torch.randn(B, 3, device=st.device)
    u = u / (u.norm(dim=1, keepdim=True) + 1e-9)
    if band == "near":
        m = torch.rand(B, 1, device=st.device) * 1.5 + 0.5
    elif band == "rec":
        m = torch.rand(B, 1, device=st.device) * 2.0 + 2.0
    else:
        m = torch.rand(B, 1, device=st.device) * 3.5 + 0.5
    dz = m * u
    pert = torch.zeros_like(st)
    pert[:, :, 44:47] = (dz * obs_scale[None, :])[:, None, :]
    ob2 = {"state": st + pert} if isinstance(obs, dict) else st + pert
    emb2 = encoder(ob2, None)
    pred2 = flow_map.get_velocity(s_t, act_0, emb2)
    dfa = pred2 - pred0
    if tgtmode == "iso":
        tgt = (-kz * dz) * act_scale[None, :]
    else:
        # fitted GT pos rows act on full 53-d dz (only POS dims nonzero here)
        dz53 = torch.zeros(B, 53, device=st.device)
        dz53[:, 44:47] = dz
        tgt = (dz53 @ ggt_pos.T) * act_scale[None, :]
        if tgtmode == "gtdamped":
            tgt = alpha * tgt
    START = 1
    hi = min(START + pdh, dfa.shape[1])
    lpd = torch.mean(torch.sum((dfa[:, START:hi, :3] - tgt[:, None, :]) ** 2, dim=-1))
    return config.loss_scale * (loss0 + lam * lpd), {}


def mip_origin_loss(
    config: OptimizationConfig,
    flow_map: FlowMap,
    encoder: BaseEncoder,
    interp: Interpolant,
    act: torch.Tensor,
    obs: torch.Tensor,
    delta_t: torch.Tensor,
) -> float:
    """Minimum iterative policy loss (original form with scale in first iteration)."""
    # sample
    s = torch.zeros_like(delta_t, device=delta_t.device)
    t = torch.zeros_like(delta_t, device=delta_t.device) + config.t_two_step
    # major difference compared to tsd: remove stochasticity in input
    act_0 = torch.zeros_like(act, device=act.device)
    noise = torch.empty_like(act).normal_(0, 1)
    act_t = config.t_two_step * act + (1 - config.t_two_step) * noise

    # get condition
    obs_emb = encoder(obs, None)

    # predict
    # for first step, scale network output by t_two_step to match the scale of the second step
    # equivalent form: directly let first step predict act
    act_pred_0 = flow_map.get_velocity(s, act_0, obs_emb)
    act_target_0 = config.t_two_step * act
    act_pred_1 = flow_map.get_velocity(t, act_t, obs_emb)
    act_target_1 = act

    # compute loss
    # difference compared to tsd: no stochasticity in prediction
    loss0 = get_norm((act_pred_0 - act_target_0) / config.t_two_step, config.norm_type, config.cauchy_c)
    loss1 = get_norm(
        (act_pred_1 - act_target_1) / (1 - config.t_two_step), config.norm_type, config.cauchy_c
    )
    loss = loss0 + loss1
    loss = config.loss_scale * torch.mean(loss)

    return loss, {}


def lmd_loss(
    config: OptimizationConfig,
    flow_map: FlowMap,
    encoder: BaseEncoder,
    interp: Interpolant,
    act: torch.Tensor,
    obs: torch.Tensor,
    delta_t: torch.Tensor,
) -> float:
    """Lagrangian map matching loss for distillation."""
    # sample
    temp_batch_1 = torch.empty_like(delta_t).uniform_(0, 1)
    temp_batch_2 = torch.empty_like(delta_t).uniform_(0, 1)
    s = torch.minimum(temp_batch_1, temp_batch_2)
    t = torch.maximum(temp_batch_1, temp_batch_2)
    s = torch.maximum(s, t - delta_t)
    act_0 = torch.empty_like(act).normal_(0, 1)
    act_1 = act

    # get condition
    label = encoder(obs, None)

    # predict
    Is = interp.calc_It(s, act_0, act_1)
    Xst_Is, dt_Xst = flow_map.jvp_t(s, t, Is, label)

    # compute the target velocity field
    b_eval = flow_map.get_reference_velocity(t, Xst_Is, label)

    # lmd loss
    loss = torch.mean(
        (dt_Xst.flatten(start_dim=1) - b_eval.flatten(start_dim=1)) ** 2, dim=-1
    )
    loss = config.loss_scale * torch.mean(loss)

    return loss, {}


def ctm_loss(
    config: OptimizationConfig,
    flow_map: FlowMap,
    encoder: BaseEncoder,
    interp: Interpolant,
    act: torch.Tensor,
    obs: torch.Tensor,
    delta_t: torch.Tensor,
) -> float:
    """Consistency trajectory model loss."""
    # sample
    temp_batch_1 = torch.empty_like(delta_t).uniform_(0, 1)
    temp_batch_2 = torch.empty_like(delta_t).uniform_(0, 1)
    s = torch.minimum(temp_batch_1, temp_batch_2)
    t = torch.maximum(temp_batch_1, temp_batch_2)
    s = torch.maximum(s, t - delta_t)
    s_plus = s + config.discrete_dt
    t = torch.maximum(t, s_plus)
    act_0 = torch.empty_like(act).normal_(0, 1)
    act_1 = act

    # get condition
    obs_emb = encoder(obs, None)

    # predict
    Is = interp.calc_It(s, act_0, act_1)

    # compute the CTM loss
    Xst_Is_pred = flow_map(s, t, Is, obs_emb)
    b_s = flow_map.get_reference_velocity(s, Is, obs_emb)
    Is_plus = Is + config.discrete_dt * b_s
    Xst_Is_target = flow_map(s_plus, t, Is_plus, obs_emb)
    # make sure loss is not too small
    loss = config.loss_scale * torch.mean(
        ((Xst_Is_target - Xst_Is_pred) / config.discrete_dt) ** 2
    )

    return loss, {}


def psd_loss(
    config: OptimizationConfig,
    flow_map: FlowMap,
    encoder: BaseEncoder,
    interp: Interpolant,
    act: torch.Tensor,
    obs: torch.Tensor,
    delta_t: torch.Tensor,
) -> float:
    """Progressive Self-Distillation loss combined with flow matching.

    This loss combines:
    1. Standard flow matching loss
    2. PSD term that encourages consistency between single-step and multi-step predictions

    The PSD term uses uniform weighting between intermediate steps.
    """
    # ========== Flow matching loss ==========
    # sample
    t_flow = torch.empty_like(delta_t).uniform_(0, 1)
    act_0 = torch.empty_like(act).normal_(0, 1)
    act_1 = act

    # get condition
    obs_emb = encoder(obs, None)

    # predict
    act_t = interp.calc_It(t_flow, act_0, act_1)
    act_t_dot = interp.calc_It_dot(t_flow, act_0, act_1)
    b_t = flow_map.get_velocity(t_flow, act_t, obs_emb)

    # compute flow loss
    flow_matching_loss = get_norm(b_t - act_t_dot, config.norm_type, config.cauchy_c)
    flow_matching_loss = config.loss_scale * torch.mean(flow_matching_loss)

    # ========== PSD term ==========
    # sample s, t, u like lmd loss
    temp_batch_1 = torch.empty_like(delta_t).uniform_(0, 1)
    temp_batch_2 = torch.empty_like(delta_t).uniform_(0, 1)
    s = torch.minimum(temp_batch_1, temp_batch_2)
    t = torch.maximum(temp_batch_1, temp_batch_2)
    s = torch.maximum(s, t - delta_t)

    # sample u uniformly between s and t
    h = torch.empty_like(delta_t).uniform_(0, 1)
    u = s + h * (t - s)

    # get interpolated starting point
    Is = interp.calc_It(s, act_0, act_1)

    # compute full jump s -> t (student)
    _, f_xst = flow_map.get_map_and_velocity(s, t, Is, obs_emb)

    # compute two-step jump s -> u -> t (teacher, no stopgrad)
    xsu, f_xsu = flow_map.get_map_and_velocity(s, u, Is, obs_emb)
    _, f_xut = flow_map.get_map_and_velocity(u, t, xsu, obs_emb)

    # uniform PSD: teacher = (1 - h) * phi_su + h * phi_ut
    # where h is the relative position of u between s and t
    student = f_xst
    # expand h to match f_xsu dimensions: [batch, horizon, act_dim]
    h_expanded = h.view(-1, 1, 1)
    teacher = (1 - h_expanded) * f_xsu + h_expanded * f_xut

    # compute PSD loss using get_norm (ignore weight_st as requested)
    psd_term = get_norm(student - teacher, config.norm_type, config.cauchy_c)
    psd_term = config.loss_scale * torch.mean(psd_term)

    # combine losses
    total_loss = flow_matching_loss + psd_term

    return total_loss, {
        "flow_loss": flow_matching_loss.item(),
        "psd_term": psd_term.item(),
    }


def lsd_loss(
    config: OptimizationConfig,
    flow_map: FlowMap,
    encoder: BaseEncoder,
    interp: Interpolant,
    act: torch.Tensor,
    obs: torch.Tensor,
    delta_t: torch.Tensor,
) -> float:
    """Lagrangian self-distillation loss combined with flow matching.

    This loss combines:
    1. Standard flow matching loss
    2. LSD term that encourages consistency in the velocity field

    The LSD term uses uniform sampling between s and t without stopgrad.
    """
    # ========== Flow matching loss ==========
    # sample
    t_flow = torch.empty_like(delta_t).uniform_(0, 1)
    act_0 = torch.empty_like(act).normal_(0, 1)
    act_1 = act

    # get condition
    obs_emb = encoder(obs, None)

    # predict
    act_t = interp.calc_It(t_flow, act_0, act_1)
    act_t_dot = interp.calc_It_dot(t_flow, act_0, act_1)
    b_t = flow_map.get_velocity(t_flow, act_t, obs_emb)

    # compute flow loss
    flow_matching_loss = get_norm(b_t - act_t_dot, config.norm_type, config.cauchy_c)
    flow_matching_loss = config.loss_scale * torch.mean(flow_matching_loss)

    # ========== LSD term ==========
    # sample s, t like lmd loss
    temp_batch_1 = torch.empty_like(delta_t).uniform_(0, 1)
    temp_batch_2 = torch.empty_like(delta_t).uniform_(0, 1)
    s = torch.minimum(temp_batch_1, temp_batch_2)
    t = torch.maximum(temp_batch_1, temp_batch_2)
    s = torch.maximum(s, t - delta_t)

    # get interpolated starting point
    Is = interp.calc_It(s, act_0, act_1)

    # compute Xst and dt_Xst using jvp_t
    xst, dt_xst = flow_map.jvp_t(s, t, Is, obs_emb)

    # compute the velocity field at the endpoint (no stopgrad)
    b_eval = flow_map.get_velocity(t, xst, obs_emb)

    # lsd loss (ignore weight_st)
    error = b_eval - dt_xst
    lsd_term = get_norm(error, config.norm_type, config.cauchy_c)
    lsd_term = config.loss_scale * torch.mean(lsd_term)

    # combine losses
    total_loss = flow_matching_loss + lsd_term

    return total_loss, {
        "flow_loss": flow_matching_loss.item(),
        "lsd_term": lsd_term.item(),
    }


def esd_loss(
    config: OptimizationConfig,
    flow_map: FlowMap,
    encoder: BaseEncoder,
    interp: Interpolant,
    act: torch.Tensor,
    obs: torch.Tensor,
    delta_t: torch.Tensor,
) -> float:
    """Euler self-distillation loss."""
    # ========== Flow matching loss ==========
    # sample
    t_flow = torch.empty_like(delta_t).uniform_(0, 1)
    act_0 = torch.empty_like(act).normal_(0, 1)
    act_1 = act

    # get condition
    obs_emb = encoder(obs, None)

    # predict
    act_t = interp.calc_It(t_flow, act_0, act_1)
    act_t_dot = interp.calc_It_dot(t_flow, act_0, act_1)
    b_t = flow_map.get_velocity(t_flow, act_t, obs_emb)

    # compute flow loss
    flow_matching_loss = get_norm(b_t - act_t_dot, config.norm_type, config.cauchy_c)
    flow_matching_loss = config.loss_scale * torch.mean(flow_matching_loss)

    # ========== ESD term ==========
    # sample s, t like lmd loss
    temp_batch_1 = torch.empty_like(delta_t).uniform_(0, 1)
    temp_batch_2 = torch.empty_like(delta_t).uniform_(0, 1)
    s = torch.minimum(temp_batch_1, temp_batch_2)
    t = torch.maximum(temp_batch_1, temp_batch_2)
    s = torch.maximum(s, t - delta_t)

    # get interpolated starting point
    Is = interp.calc_It(s, act_0, act_1)

    # compute Xst and ds_Xst using jvp_t
    xst, ds_xst = flow_map.jvp_s(s, t, Is, obs_emb)

    # compute the velocity field at the endpoint (stopgrad)
    with torch.no_grad():
        b_eval = flow_map.get_velocity(t, xst, obs_emb)

    # compute jvp
    _, grad_xst_b = flow_map.jvp_x(s, t, Is, b_eval, obs_emb)

    # esd loss
    error = ds_xst + grad_xst_b
    esd_term = get_norm(error, config.norm_type, config.cauchy_c)
    esd_term = config.loss_scale * torch.mean(esd_term)

    # combine losses
    total_loss = flow_matching_loss + esd_term

    return total_loss, {
        "flow_loss": flow_matching_loss.item(),
        "esd_term": esd_term.item(),
    }


def mf_loss(
    config: OptimizationConfig,
    flow_map: FlowMap,
    encoder: BaseEncoder,
    interp: Interpolant,
    act: torch.Tensor,
    obs: torch.Tensor,
    delta_t: torch.Tensor,
) -> float:
    """Mean flow loss."""
    # ========== Flow matching loss ==========
    # sample
    t_flow = torch.empty_like(delta_t).uniform_(0, 1)
    act_0 = torch.empty_like(act).normal_(0, 1)
    act_1 = act

    # get condition
    obs_emb = encoder(obs, None)

    # predict
    act_t = interp.calc_It(t_flow, act_0, act_1)
    act_t_dot = interp.calc_It_dot(t_flow, act_0, act_1)
    b_t = flow_map.get_velocity(t_flow, act_t, obs_emb)

    # compute flow loss
    flow_matching_loss = get_norm(b_t - act_t_dot, config.norm_type, config.cauchy_c)
    flow_matching_loss = config.loss_scale * torch.mean(flow_matching_loss)

    # ========== Mean flow term ==========
    # sample s, t
    temp_batch_1 = torch.empty_like(delta_t).uniform_(0, 1)
    temp_batch_2 = torch.empty_like(delta_t).uniform_(0, 1)
    s = torch.minimum(temp_batch_1, temp_batch_2)
    t = torch.maximum(temp_batch_1, temp_batch_2)
    s = torch.maximum(s, t - delta_t)

    # get interpolated starting point
    Is = interp.calc_It(s, act_0, act_1)
    dot_Is = interp.calc_It_dot(s, act_0, act_1)

    # compute Xst and ds_Xst using jvp_t
    xst, ds_xst = flow_map.jvp_s(s, t, Is, obs_emb)

    # compute the velocity field at the endpoint (stopgrad)
    with torch.no_grad():
        # Difference 1: use dot_Is instead of b_eval
        # Difference 2: also disable gradient for jvp_x
        # compute jvp
        _, grad_xst_b = flow_map.jvp_x(s, t, Is, dot_Is, obs_emb)

    # mf loss
    error = ds_xst + grad_xst_b
    mf_term = get_norm(error, config.norm_type, config.cauchy_c)
    mf_term = config.loss_scale * torch.mean(mf_term)

    # combine losses
    total_loss = flow_matching_loss + mf_term

    return total_loss, {
        "flow_loss": flow_matching_loss.item(),
        "mf_term": mf_term.item(),
    }


def regression_relerr_loss(
    config: OptimizationConfig,
    flow_map: FlowMap,
    encoder: BaseEncoder,
    interp: Interpolant,
    act: torch.Tensor,
    obs: torch.Tensor,
    delta_t: torch.Tensor,
) -> float:
    """Scale-free regression: per-step relative error, testing the dynamic-range
    account of MSE's fine-structure underfitting. loss_t = |e_t|^2/(|a_t|^2+c^2)."""
    t = torch.zeros_like(delta_t, device=delta_t.device)
    act_0 = torch.zeros_like(act, device=act.device)
    obs_emb = encoder(obs, None)
    act_pred = flow_map.get_velocity(t, act_0, obs_emb)
    c2 = 0.01  # c=0.1 per-dim: elementwise scale-free (settle pos dims get ~100x weight)
    per_step = torch.sum((act_pred - act) ** 2 / (act ** 2 + c2), dim=-1)
    loss = config.loss_scale * torch.mean(per_step)
    return loss, {}

def mip_quant_loss(
    config: OptimizationConfig,
    flow_map: FlowMap,
    encoder: BaseEncoder,
    interp: Interpolant,
    act: torch.Tensor,
    obs: torch.Tensor,
    delta_t: torch.Tensor,
) -> float:
    """y1-view with DETERMINISTIC scale gate: anchor = sigma*round(a/sigma) destroys
    sub-sigma info without randomness. Tests whether the noise ball's role is pure
    scale-gating. QUANT_SIGMA env (default 0.1)."""
    import os as _os
    sigma = float(_os.environ.get("QUANT_SIGMA", "0.1"))
    s = torch.zeros_like(delta_t, device=delta_t.device)
    t = torch.zeros_like(delta_t, device=delta_t.device) + config.t_two_step
    act_0 = torch.zeros_like(act, device=act.device)
    anchor = sigma * torch.round(act / sigma)
    obs_emb = encoder(obs, None)
    act_pred_0 = flow_map.get_velocity(s, act_0, obs_emb)
    act_pred_1 = flow_map.get_velocity(t, anchor, obs_emb)
    loss0 = get_norm((act_pred_0 - act) / config.t_two_step, config.norm_type, config.cauchy_c)
    loss1 = get_norm((act_pred_1 - act) / (1 - config.t_two_step), config.norm_type, config.cauchy_c)
    loss = config.loss_scale * torch.mean(loss0 + loss1)
    return loss, {}

_RESID_CACHE = {"loaded": False, "fm": None, "enc": None}

def _resid_teacher(flow_map, encoder, device):
    """Lazy-load frozen coarse teacher from TEACHER_CKPT (same architecture)."""
    import os as _os, torch as _torch
    from copy import deepcopy as _dc
    if not _RESID_CACHE["loaded"]:
        sd = _torch.load(_os.environ["TEACHER_CKPT"], map_location=device, weights_only=False)
        fm = _dc(flow_map).requires_grad_(False); fm.load_state_dict(sd["flow_map_ema"])
        enc = _dc(encoder).requires_grad_(False); enc.load_state_dict(sd["encoder_ema"])
        fm.eval(); enc.eval()
        _RESID_CACHE.update(loaded=True, fm=fm, enc=enc)
    return _RESID_CACHE["fm"], _RESID_CACHE["enc"]


def regression_residual_loss(
    config: OptimizationConfig,
    flow_map: FlowMap,
    encoder: BaseEncoder,
    interp: Interpolant,
    act: torch.Tensor,
    obs: torch.Tensor,
    delta_t: torch.Tensor,
) -> float:
    """Explicit coarse/fine decomposition (boosting): student regresses the residual
    a - teacher(s) with plain MSE. Teacher = early-stopped coarse model (TEACHER_CKPT).
    Deployment (see sampler) = teacher(s) + student(s)."""
    tfm, tenc = _resid_teacher(flow_map, encoder, act.device)
    t = torch.zeros_like(delta_t, device=delta_t.device)
    act_0 = torch.zeros_like(act, device=act.device)
    with torch.no_grad():
        coarse = tfm.get_velocity(t, act_0, tenc(obs, None))
    obs_emb = encoder(obs, None)
    pred = flow_map.get_velocity(t, act_0, obs_emb)
    loss = config.loss_scale * torch.mean(get_norm(pred - (act - coarse), config.norm_type, config.cauchy_c))
    return loss, {}

_SIGANNEAL = {"step": 0}

def mip_siganneal_loss(
    config: OptimizationConfig,
    flow_map: FlowMap,
    encoder: BaseEncoder,
    interp: Interpolant,
    act: torch.Tensor,
    obs: torch.Tensor,
    delta_t: torch.Tensor,
) -> float:
    """sigma-annealed y1 view: noise scale swept log-linearly sigma_min -> sigma_max
    over training (gate sweeps fine->coarse; responsibilities accumulate in emb).
    Identical to mip_loss otherwise (t input fixed at t_two_step; loss1 / sigma^2).
    Env: SIG_MIN (0.003), SIG_MAX (0.3), SIG_STEPS (300000)."""
    import os as _os, math as _math
    _SIGANNEAL["step"] += 1
    p = min(_SIGANNEAL["step"] / float(_os.environ.get("SIG_STEPS", "300000")), 1.0)
    smin = float(_os.environ.get("SIG_MIN", "0.003")); smax = float(_os.environ.get("SIG_MAX", "0.3"))
    sigma = smin * (smax / smin) ** p
    s = torch.zeros_like(delta_t, device=delta_t.device)
    t = torch.zeros_like(delta_t, device=delta_t.device) + config.t_two_step
    act_0 = torch.zeros_like(act, device=act.device)
    noise = torch.empty_like(act).normal_(0, 1)
    act_t = act + sigma * noise
    obs_emb = encoder(obs, None)
    act_pred_0 = flow_map.get_velocity(s, act_0, obs_emb)
    act_pred_1 = flow_map.get_velocity(t, act_t, obs_emb)
    loss0 = get_norm((act_pred_0 - act) / config.t_two_step, config.norm_type, config.cauchy_c)
    loss1 = get_norm((act_pred_1 - act) / sigma, config.norm_type, config.cauchy_c)
    loss = config.loss_scale * torch.mean(loss0 + loss1)
    return loss, {}

_FADEHINT = {"step": 0}

def regression_fadehint_loss(
    config: OptimizationConfig,
    flow_map: FlowMap,
    encoder: BaseEncoder,
    interp: Interpolant,
    act: torch.Tensor,
    obs: torch.Tensor,
    delta_t: torch.Tensor,
) -> float:
    """Single-view fading-hint regression: f(s, a + sigma(step)*eps, t=0) -> a, plain MSE,
    sigma log-swept SIG_MIN -> SIG_MAX (default 0.003 -> 3.0; end state = anchor drowned
    in noise, deployment f(s,0,0)). No y0 view, no t conditioning, no 1/sigma^2 weighting."""
    import os as _os
    _FADEHINT["step"] += 1
    p = min(_FADEHINT["step"] / float(_os.environ.get("SIG_STEPS", "300000")), 1.0)
    smin = float(_os.environ.get("SIG_MIN", "0.003")); smax = float(_os.environ.get("SIG_MAX", "3.0"))
    sigma = smin * (smax / smin) ** p
    t = torch.zeros_like(delta_t, device=delta_t.device)
    noise = torch.empty_like(act).normal_(0, 1)
    obs_emb = encoder(obs, None)
    pred = flow_map.get_velocity(t, act + sigma * noise, obs_emb)
    loss = config.loss_scale * torch.mean(get_norm(pred - act, config.norm_type, config.cauchy_c))
    return loss, {}
