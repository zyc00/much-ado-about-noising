"""Heteroscedastic Student-t regression policy inside the Diffusion Policy
framework (Chi et al., 2023).

Drop-in replacement for DiffusionTransformerLowdimPolicy that keeps the SAME
TransformerForDiffusion backbone, optimizer grouping, EMA, dataloader, env
runner and eval protocol.  The only differences are:
  * training objective: Student-t NLL with a learned per-sample scale
    (nu fixed, default 2) instead of DDPM epsilon-MSE;
  * inference: ONE forward pass at t=0 from a zero trajectory, instead of a
    100-step denoising chain.
The network emits (action_chunk, raw_scale): output_dim = action_dim + 1.
"""
from typing import Dict, Tuple

import torch
import torch.nn.functional as F

from diffusion_policy.model.common.normalizer import LinearNormalizer
from diffusion_policy.model.diffusion.transformer_for_diffusion import (
    TransformerForDiffusion,
)
from diffusion_policy.policy.base_lowdim_policy import BaseLowdimPolicy


class HeteroTTransformerLowdimPolicy(BaseLowdimPolicy):
    def __init__(
        self,
        model: TransformerForDiffusion,
        horizon,
        obs_dim,
        action_dim,
        n_action_steps,
        n_obs_steps,
        student_t_df: float = 2.0,
        obs_as_cond: bool = True,
        pred_action_steps_only: bool = False,
        **kwargs,
    ):
        super().__init__()
        assert obs_as_cond, "hetero-t policy requires obs_as_cond=True"
        assert not pred_action_steps_only
        self.model = model
        self.normalizer = LinearNormalizer()
        self.horizon = horizon
        self.obs_dim = obs_dim
        self.action_dim = action_dim
        self.n_action_steps = n_action_steps
        self.n_obs_steps = n_obs_steps
        self.student_t_df = float(student_t_df)
        self.obs_as_cond = obs_as_cond
        self.pred_action_steps_only = False
        self.kwargs = kwargs

    # ========= shared forward =========
    def _forward(self, cond, B, device, dtype):
        traj_in = torch.zeros(
            (B, self.horizon, self.action_dim), device=device, dtype=dtype
        )
        timestep = torch.zeros((B,), device=device, dtype=torch.long)
        out = self.model(traj_in, timestep, cond)
        return out[..., : self.action_dim], out[..., self.action_dim :]

    # ========= inference =========
    def predict_action(self, obs_dict: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        assert "obs" in obs_dict
        assert "past_action" not in obs_dict
        nobs = self.normalizer["obs"].normalize(obs_dict["obs"])
        B, _, Do = nobs.shape
        assert Do == self.obs_dim
        cond = nobs[:, : self.n_obs_steps]
        naction_pred, _ = self._forward(cond, B, self.device, self.dtype)
        action_pred = self.normalizer["action"].unnormalize(naction_pred)
        start = self.n_obs_steps - 1
        end = start + self.n_action_steps
        return {"action": action_pred[:, start:end], "action_pred": action_pred}

    # ========= training =========
    def set_normalizer(self, normalizer: LinearNormalizer):
        self.normalizer.load_state_dict(normalizer.state_dict())

    def get_optimizer(
        self, weight_decay: float, learning_rate: float, betas: Tuple[float, float]
    ) -> torch.optim.Optimizer:
        return self.model.configure_optimizers(
            weight_decay=weight_decay,
            learning_rate=learning_rate,
            betas=tuple(betas),
        )

    def compute_loss(self, batch):
        assert "valid_mask" not in batch
        nbatch = self.normalizer.normalize(batch)
        obs = nbatch["obs"]
        action = nbatch["action"]
        cond = obs[:, : self.n_obs_steps, :]
        B = action.shape[0]

        pred, s_raw = self._forward(cond, B, action.device, action.dtype)

        # learned per-sample scale, pooled over the action chunk
        sigma = F.softplus(s_raw).reshape(B, -1).mean(dim=1) + 1e-3
        nu = self.student_t_df
        r2 = (pred - action) ** 2
        n_dim = r2[0].numel()
        sum_r2 = r2.reshape(B, -1).sum(dim=1)
        per = 0.5 * (nu + 1.0) * torch.log1p(
            sum_r2 / (nu * sigma**2 * n_dim)
        ) * n_dim + n_dim * torch.log(sigma)
        return per.mean() / n_dim
