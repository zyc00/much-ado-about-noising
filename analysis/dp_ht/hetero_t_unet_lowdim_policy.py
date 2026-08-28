"""Heteroscedastic Student-t regression policy on DP's ConditionalUnet1D.

CNN counterpart of hetero_t_transformer_lowdim_policy.  Same backbone,
optimizer, EMA, dataloader, env runner and eval protocol as
DiffusionUnetLowdimPolicy; only the objective (Student-t NLL with a learned
scale) and single-pass inference differ.

The UNet is shape-symmetric (output_dim == input_dim), so the scale rides in
one extra channel: the network runs on width action_dim+1 and the trajectory
input is zeros, whose content is irrelevant for a regression policy.
"""
import os
from typing import Dict

import torch
import torch.nn.functional as F

from diffusion_policy.model.common.normalizer import LinearNormalizer
from diffusion_policy.model.diffusion.conditional_unet1d import ConditionalUnet1D
from diffusion_policy.policy.base_lowdim_policy import BaseLowdimPolicy


class HeteroTUnetLowdimPolicy(BaseLowdimPolicy):
    def __init__(
        self,
        model: ConditionalUnet1D,
        horizon,
        obs_dim,
        action_dim,
        n_action_steps,
        n_obs_steps,
        student_t_df: float = 2.0,
        sigma_mode: str = "pooled",
        obs_as_local_cond: bool = False,
        obs_as_global_cond: bool = True,
        pred_action_steps_only: bool = False,
        **kwargs,
    ):
        super().__init__()
        assert obs_as_global_cond and not obs_as_local_cond, (
            "hetero-t unet policy expects obs_as_global_cond=True"
        )
        assert not pred_action_steps_only
        self.model = model
        self.normalizer = LinearNormalizer()
        self.horizon = horizon
        self.obs_dim = obs_dim
        self.action_dim = action_dim
        self.n_action_steps = n_action_steps
        self.n_obs_steps = n_obs_steps
        self.student_t_df = float(student_t_df)
        assert sigma_mode in ("pooled", "perdim")
        self.sigma_mode = sigma_mode
        self.obs_as_local_cond = False
        self.obs_as_global_cond = True
        self.pred_action_steps_only = False
        self.kwargs = kwargs

    def _forward(self, global_cond, B, device, dtype):
        width = self.action_dim * 2 if self.sigma_mode == "perdim" else self.action_dim + 1
        traj_in = torch.zeros((B, self.horizon, width), device=device, dtype=dtype)
        timestep = torch.zeros((B,), device=device, dtype=torch.long)
        out = self.model(traj_in, timestep, global_cond=global_cond)
        return out[..., : self.action_dim], out[..., self.action_dim :]

    def predict_action(self, obs_dict: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        assert "obs" in obs_dict
        assert "past_action" not in obs_dict
        nobs = self.normalizer["obs"].normalize(obs_dict["obs"])
        B, _, Do = nobs.shape
        assert Do == self.obs_dim
        global_cond = nobs[:, : self.n_obs_steps].reshape(B, -1)
        naction_pred, _ = self._forward(global_cond, B, self.device, self.dtype)
        action_pred = self.normalizer["action"].unnormalize(naction_pred)
        start = self.n_obs_steps - 1
        end = start + self.n_action_steps
        return {"action": action_pred[:, start:end], "action_pred": action_pred}

    def set_normalizer(self, normalizer: LinearNormalizer):
        self.normalizer.load_state_dict(normalizer.state_dict())
        if os.environ.get("HT_MINMAX_ACTION") == "1":
            # T12-style per-dim min-max on actions.  DP leaves delta actions
            # unscaled (identity), which leaves the rotation dims ~3x smaller
            # than position dims; regression (unlike epsilon-prediction) is
            # sensitive to that imbalance.  Recompute scale/offset so every
            # action dim maps to [-1, 1], from the stored input stats.
            pd = self.normalizer.params_dict["action"]
            in_stats = pd["input_stats"]
            mn, mx = in_stats["min"], in_stats["max"]
            rng = (mx - mn).clone()
            rng[rng < 1e-7] = 1.0
            scale = 2.0 / rng
            offset = -1.0 - scale * mn
            pd["scale"].copy_(scale)
            pd["offset"].copy_(offset)

    def compute_loss(self, batch):
        assert "valid_mask" not in batch
        nbatch = self.normalizer.normalize(batch)
        obs = nbatch["obs"]
        action = nbatch["action"]
        B = action.shape[0]
        global_cond = obs[:, : self.n_obs_steps].reshape(B, -1)

        pred, s_raw = self._forward(global_cond, B, action.device, action.dtype)

        nu = self.student_t_df
        if self.sigma_mode == "perdim":
            # classic heteroscedastic Student-t: independent scale per (t, dim)
            sigma = F.softplus(s_raw) + 1e-3
            r2 = (pred - action) ** 2
            nll = 0.5 * (nu + 1.0) * torch.log1p(
                r2 / (nu * sigma**2)
            ) + torch.log(sigma)
            return nll.mean()
        sigma = F.softplus(s_raw).reshape(B, -1).mean(dim=1) + 1e-3
        r2 = (pred - action) ** 2
        n_dim = r2[0].numel()
        sum_r2 = r2.reshape(B, -1).sum(dim=1)
        per = 0.5 * (nu + 1.0) * torch.log1p(
            sum_r2 / (nu * sigma**2 * n_dim)
        ) * n_dim + n_dim * torch.log(sigma)
        return per.mean() / n_dim
