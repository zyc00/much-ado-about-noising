"""Sampler for different training objectives.

Author: Chaoyi Pan
Date: 2025-10-03
"""

import numpy as np
import torch

from mip.config import OptimizationConfig
from mip.encoders import BaseEncoder
from mip.flow_map import FlowMap
from mip.torch_utils import at_least_ndim


def get_default_step_list(loss_type: str):
    if loss_type in ["flow", "ctm", "lmd", "psd", "lsd", "esd", "mf"]:
        return 3 ** np.arange(2, -1, -1)
    elif loss_type in ["regression", "regression_relerr", "regression_fadehint", "regression_residual", "regression_rw", "regression_cauchy", "regression_student_t", "regression_hetero_t", "regression_globalt", "regression_hgclip", "regression_gmm", "regression_sigmaw", "regression_selfsw", "regression_invw", "regression_gfloor", "regression_selfnorm", "regression_condreg", "regression_condnum", "regression_pathdamp", "regression_focal", "regression_hetero_t_rankmid", "regression_hetero_t_jspec", "regression_jspec", "regression_hetero_t_mixnn", "regression_hetero_t_midlin", "regression_hetero_t_jitcons", "regression_hetero_gauss_midlin", "regression_hetero_gauss_jspec", "regression_hetero_gauss_rankmid", "regression_fdistill", "regression_condann", "regression_dcr", "regression_hetero_gauss_cnd", "regression_hetero_diag", "regression_hetero_t_diag", "regression_hetero_t_cnd", "regression_dcw", "regression_labelnoise", "regression_featdrop", "regression_snteach", "regression_emaret", "regression_hetero_gauss", "regression_sigaux", "regression_headonly", "regression_binw", "regression_distinc", "regression_hetero_t_fdlip", "regression_hetero_t_tublip", "regression_pace", "regression_paceabs", "regression_trim", "regression_normed", "regression_stdt", "mip", "mip_cauchyv1", "mip_heterov1", "mip_quant", "mip_siganneal", "mip_rw", "mip_step1", "mip_nonoise", "mip_nonoise_atk", "mip_shufx", "mip_shufx", "mip_auxtan", "mip_auxflip", "mip_auxdetach", "mip_distill", "mip_distill2", "mip_lambda", "tsd", "straight_flow", "regression_manifold", "regression_obsnoise", "regression_offmanjac", "regression_geomreg", "regression_stressreg", "regression_sigreg", "regression_randaug", "regression_tjitter", "regression_jacreg", "regression_dup100", "regression_frozentrunk", "regression_cvu", "regression_recgeo", "regression_pdprior", "regression_tauteacher", "regression_taufeat", "regression_badpen", "regression_opanchor", "regression_cvu", "regression_recgeo", "regression_pdprior", "mip_tubeaux", "mip_scramaux", "mip_randaux", "mip_eqw", "denoise_only", "denoise_zeroin", "denoise_zeroin_flat", "denoise_randin", "denoise_scramble", "mip_auxdet", "mip_zeroaux", "mip_tubeaux", "mip_scramaux", "mip_randaux"]:
        return [1]
    else:
        raise NotImplementedError(f"Loss type {loss_type} not implemented.")


def get_sampler(loss_type: str):
    if loss_type == "flow":
        return ode_sampler
    elif loss_type == "regression_residual":
        return regression_residual_sampler
    elif loss_type in ["regression", "regression_relerr", "regression_fadehint", "regression_rw", "regression_cauchy", "regression_student_t", "regression_hetero_t", "regression_globalt", "regression_hgclip", "regression_gmm", "regression_sigmaw", "regression_selfsw", "regression_invw", "regression_gfloor", "regression_selfnorm", "regression_condreg", "regression_condnum", "regression_pathdamp", "regression_focal", "regression_hetero_t_rankmid", "regression_hetero_t_jspec", "regression_jspec", "regression_hetero_t_mixnn", "regression_hetero_t_midlin", "regression_hetero_t_jitcons", "regression_hetero_gauss_midlin", "regression_hetero_gauss_jspec", "regression_hetero_gauss_rankmid", "regression_fdistill", "regression_condann", "regression_dcr", "regression_hetero_gauss_cnd", "regression_hetero_diag", "regression_hetero_t_diag", "regression_hetero_t_cnd", "regression_dcw", "regression_labelnoise", "regression_featdrop", "regression_snteach", "regression_emaret", "regression_hetero_gauss", "regression_sigaux", "regression_headonly", "regression_binw", "regression_distinc", "regression_hetero_t_fdlip", "regression_hetero_t_tublip", "regression_pace", "regression_paceabs", "regression_trim", "regression_normed", "regression_stdt", "straight_flow", "regression_obsnoise", "regression_offmanjac", "regression_geomreg", "regression_stressreg", "regression_sigreg", "regression_randaug", "regression_tjitter", "regression_jacreg", "regression_dup100", "regression_frozentrunk", "regression_cvu", "regression_recgeo", "regression_pdprior", "regression_tauteacher", "regression_taufeat", "regression_badpen", "regression_opanchor"]:
        return regression_sampler
    elif loss_type in ["mip_quant", "mip_siganneal", "mip_eqw", "denoise_only", "denoise_zeroin", "denoise_zeroin_flat", "denoise_randin", "denoise_scramble"]:
        return mip_sampler
    elif loss_type == "regression_manifold":
        return regression_manifold_sampler
    elif loss_type in ["tsd", "mip", "mip_cauchyv1", "mip_heterov1", "mip_rw", "mip_nonoise", "mip_nonoise_atk", "mip_shufx", "mip_shufx", "mip_auxtan", "mip_auxflip", "mip_auxdetach", "mip_distill", "mip_distill2", "mip_lambda", "mip_auxdet", "mip_zeroaux", "mip_tubeaux", "mip_scramaux", "mip_randaux"]:
        return mip_sampler
    elif loss_type == "mip_step1":
        return mip_step1_only_sampler
    elif loss_type in ["lmd", "ctm", "psd", "lsd", "esd", "mf"]:
        return flow_map_sampler
    else:
        raise NotImplementedError(f"Loss type {loss_type} not implemented.")


def ode_sampler(
    config: OptimizationConfig,
    flow_map: FlowMap,
    encoder: BaseEncoder,
    act_0: torch.Tensor,
    obs: torch.Tensor,
):
    num_steps = config.num_steps
    sample_mode = config.sample_mode
    t_schedule = np.linspace(0, 1, num_steps + 1)
    if sample_mode == "stochastic":
        act_s = torch.randn_like(act_0, device=act_0.device)
    else:
        act_s = torch.zeros_like(act_0, device=act_0.device)
    obs_emb = encoder(obs, None)
    bs = act_0.shape[0]
    for i in range(num_steps):
        s_val = t_schedule[i]
        t_val = t_schedule[i + 1]
        s = torch.full((bs,), s_val, device=act_0.device)
        t = torch.full((bs,), t_val, device=act_0.device)
        b_s = flow_map.get_velocity(s, act_s, obs_emb)
        s_expanded = at_least_ndim(s, act_s.dim())
        t_expanded = at_least_ndim(t, act_s.dim())
        act_s = act_s + b_s * (t_expanded - s_expanded)
    act = act_s
    return act


def flow_map_sampler(
    config: OptimizationConfig,
    flow_map: FlowMap,
    encoder: BaseEncoder,
    act_0: torch.Tensor,
    obs: torch.Tensor,
):
    """This function is designed for flow map sampler, i.e. for the distilled shortcut model.

    Args:
        config (OptimizationConfig): the configuration
        flow_map (FlowMap): the flow map
        encoder (BaseEncoder): the encoder
        act_0 (torch.Tensor): the initial action
        obs (torch.Tensor): the observation

    Returns:
        torch.Tensor: the sampled action
    """
    num_steps = config.num_steps
    sample_mode = config.sample_mode
    t_schedule = np.linspace(0, 1, num_steps + 1)
    if sample_mode == "stochastic":
        act_s = torch.randn_like(act_0, device=act_0.device)
    else:
        act_s = torch.zeros_like(act_0, device=act_0.device)
    obs_emb = encoder(obs, None)
    bs = act_0.shape[0]
    for i in range(num_steps):
        s_val = t_schedule[i]
        t_val = t_schedule[i + 1]
        s = torch.full((bs,), s_val, device=act_0.device)
        t = torch.full((bs,), t_val, device=act_0.device)
        act_s = flow_map(s, t, act_s, obs_emb)
    act = act_s
    return act


def regression_sampler(
    config: OptimizationConfig,
    flow_map: FlowMap,
    encoder: BaseEncoder,
    act_0: torch.Tensor,
    obs: torch.Tensor,
):
    bs = act_0.shape[0]
    act_zeros = torch.zeros_like(act_0, device=act_0.device)
    t = torch.zeros(bs, device=act_0.device)
    obs_emb = encoder(obs, None)
    act = flow_map.get_velocity(t, act_zeros, obs_emb)
    return act


def regression_manifold_sampler(
    config: OptimizationConfig,
    flow_map: FlowMap,
    encoder: BaseEncoder,
    act_0: torch.Tensor,
    obs: torch.Tensor,
):
    """Inference for regression_manifold: predict from zeros, then pass through the same
    frozen action manifold (denoiser) used in training."""
    from mip.action_manifold import project
    bs = act_0.shape[0]
    act_zeros = torch.zeros_like(act_0, device=act_0.device)
    t = torch.zeros(bs, device=act_0.device)
    obs_emb = encoder(obs, None)
    act = flow_map.get_velocity(t, act_zeros, obs_emb)
    return project(act, act_0.device)


def mip_step1_only_sampler(
    config: OptimizationConfig,
    flow_map: FlowMap,
    encoder: BaseEncoder,
    act_0: torch.Tensor,
    obs: torch.Tensor,
):
    """MIP step 1 only — same as regression but from MIP-trained network."""
    bs = act_0.shape[0]
    s = torch.zeros((bs,), device=act_0.device)
    obs_emb = encoder(obs, None)
    act_0 = torch.zeros_like(act_0, device=act_0.device)
    act_pred_0 = flow_map.get_velocity(s, act_0, obs_emb)
    return act_pred_0


def regression_residual_sampler(
    config,
    flow_map,
    encoder,
    act_0,
    obs,
):
    """Deploy teacher(s) + student(s)."""
    from mip.losses import _resid_teacher
    bs = act_0.shape[0]
    t = torch.zeros((bs,), device=act_0.device)
    z = torch.zeros_like(act_0, device=act_0.device)
    tfm, tenc = _resid_teacher(flow_map, encoder, act_0.device)
    coarse = tfm.get_velocity(t, z, tenc(obs, None))
    fine = flow_map.get_velocity(t, z, encoder(obs, None))
    return coarse + fine


def mip_sampler(
    config: OptimizationConfig,
    flow_map: FlowMap,
    encoder: BaseEncoder,
    act_0: torch.Tensor,
    obs: torch.Tensor,
):
    """Simplified minimum iterative policy
    Note that now the first step prediction is not scaled by t_two_step,
    """
    bs = act_0.shape[0]
    s = torch.zeros((bs,), device=act_0.device)
    t = torch.full((bs,), config.t_two_step, device=act_0.device)

    obs_emb = encoder(obs, None)

    act_0 = torch.zeros_like(act_0, device=act_0.device)
    act_pred_0 = flow_map.get_velocity(s, act_0, obs_emb)
    # NOTE: now the first step prediction is not scaled by t_two_step,
    # if you want the original form, you can use the mip_origin_sampler
    act_pred_1 = flow_map.get_velocity(t, act_pred_0, obs_emb)

    act = act_pred_1
    return act


def mip_origin_sampler(
    config: OptimizationConfig,
    flow_map: FlowMap,
    encoder: BaseEncoder,
    act_0: torch.Tensor,
    obs: torch.Tensor,
):
    """Original minimum iterative policy"""
    bs = act_0.shape[0]
    s = torch.zeros((bs,), device=act_0.device)
    t = torch.full((bs,), config.t_two_step, device=act_0.device)
    obs_emb = encoder(obs, None)

    act_0 = torch.zeros_like(act_0, device=act_0.device)
    act_pred_0 = flow_map.get_velocity(s, act_0, obs_emb)
    # this is the original form in the paper
    act_pred_1 = flow_map.get_velocity(t, act_pred_0 * config.t_two_step, obs_emb)

    act = act_pred_1
    return act
