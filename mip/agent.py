"""Torch training agent for behavior cloning."""

from __future__ import annotations

from contextlib import contextmanager
from copy import deepcopy

import os

import loguru
import torch
import torch.nn as nn
from tensordict import TensorDict

from mip.config import Config
from mip.flow_map import FlowMap
from mip.interpolant import Interpolant
from mip.losses import get_loss_fn
from mip.network_utils import get_encoder, get_network
from mip.samplers import get_sampler
from mip.torch_utils import report_parameters


class TrainingAgent:
    """Training agent for behavior cloning with flow matching."""

    def __init__(
        self,
        config: Config,
    ):
        """Initialize the training agent.

        Args:
            flow_map: The flow map model
            encoder: The observation encoder
            config: Full configuration object
        """
        self.config = config
        self.loss_fn = get_loss_fn(config.optimization.loss_type)
        self.sampler = get_sampler(config.optimization.loss_type)
        self.interpolant = Interpolant(config.optimization.interp_type)
        net = get_network(config.network, config.task)
        report_parameters(net, model_name="Action Network")
        import os as _os
        if _os.environ.get("MIP_TWONET", "0") == "1":
            import copy as _copy
            self.flow_map = FlowMap(net, reference_net=_copy.deepcopy(net)).to(config.optimization.device)
        else:
            self.flow_map = FlowMap(net).to(config.optimization.device)
        self.encoder = get_encoder(config.network, config.task).to(
            config.optimization.device
        )
        report_parameters(self.encoder, model_name="Encoder Network")
        if config.optimization.loss_type in (
            "regression_hetero_t_learnnu_cond",
            "regression_hetero_t_learnnu_cond2",
        ):
            assert config.task.obs_type == "state", "learnnu_cond: state obs only"
            with torch.no_grad():
                _dummy = torch.zeros(
                    2,
                    config.task.obs_steps,
                    config.task.obs_dim,
                    device=config.optimization.device,
                )
                _emb = self.encoder(_dummy, None)
            _in_dim = _emb.reshape(2, -1).shape[1]
            _head = torch.nn.Sequential(
                torch.nn.Linear(_in_dim, 64),
                torch.nn.SiLU(),
                torch.nn.Linear(64, 1),
            ).to(config.optimization.device)
            torch.nn.init.zeros_(_head[-1].weight)
            torch.nn.init.zeros_(_head[-1].bias)
            self.flow_map.nu_cond_head = _head
            if config.optimization.loss_type.endswith("cond2"):
                # v2: nu = floor + softplus(raw); pick raw so nu starts at
                # NU_INIT (default 2.0 = the measured demo df).
                _floor = float(os.environ.get("NU_FLOOR", "1.0"))
                _init = float(os.environ.get("NU_INIT", "2.0"))
                _target = max(_init - _floor, 1e-3)
                _raw0 = float(torch.log(torch.expm1(torch.tensor(_target))))
            else:
                # v1: sigmoid(-2.16) ~= 0.1034 -> nu ~= 1 + 29*0.1034 ~= 4.0
                _raw0 = -2.16
            self.flow_map.register_parameter(
                "nu_base_raw",
                torch.nn.Parameter(
                    torch.tensor(_raw0, device=config.optimization.device)
                ),
            )
            self.flow_map.register_buffer(
                "nu_cond_step",
                torch.tensor(0.0, device=config.optimization.device),
            )
        if config.optimization.loss_type == "regression_hetero_t_learnnu":
            # nu = 0.5 + softplus(raw); raw=3.47 -> nu ~= 4.0 start
            self.flow_map.register_parameter(
                "learn_nu_raw",
                torch.nn.Parameter(
                    torch.tensor(3.47, device=config.optimization.device)
                ),
            )
        self.encoder_ema = deepcopy(self.encoder).requires_grad_(False)
        self.flow_map_ema = deepcopy(self.flow_map).requires_grad_(False)

        # Create detached models for CUDA graphs (if enabled)
        self.use_cudagraphs = config.optimization.use_cudagraphs
        if self.use_cudagraphs:
            self.flow_map_detach = deepcopy(self.flow_map).requires_grad_(False)
            self.encoder_detach = deepcopy(self.encoder).requires_grad_(False)
            self.flow_map_ema_detach = deepcopy(self.flow_map_ema).requires_grad_(False)
            self.encoder_ema_detach = deepcopy(self.encoder_ema).requires_grad_(False)
        else:
            self.flow_map_detach = None
            self.encoder_detach = None
            self.flow_map_ema_detach = None
            self.encoder_ema_detach = None

        params = list(self.encoder.parameters()) + list(self.flow_map.parameters())
        trunk_lr_mult = float(os.environ.get("TRUNK_LR_MULT", "1"))
        if trunk_lr_mult != 1.0:
            # lazy-trunk control: trunk (encoder + all UNet blocks except final_conv)
            # trains at trunk_lr_mult * lr; the final_conv head at full lr
            fc = None
            for _n, _m in self.flow_map.named_modules():
                if _n.endswith("final_conv"):
                    fc = _m
            head_ids = {id(p) for p in fc.parameters()}
            head_p = [p for p in params if id(p) in head_ids]
            trunk_p = [p for p in params if id(p) not in head_ids]
            self.optimizer = torch.optim.AdamW(
                [{"params": trunk_p, "lr": config.optimization.lr * trunk_lr_mult},
                 {"params": head_p, "lr": config.optimization.lr}],
                weight_decay=config.optimization.weight_decay,
            )
        elif __import__("os").environ.get("OPTIM", "") == "muon":
            from mip.muon import MuonWithAdamW
            import os as _os
            named = list(self.flow_map.named_parameters()) + list(self.encoder.named_parameters())
            have = {id(p) for p in params}
            named = [(n, p) for n, p in named if id(p) in have]
            self.optimizer = MuonWithAdamW(
                named,
                muon_lr=float(_os.environ.get("MUON_LR", "0.02")),
                adamw_lr=config.optimization.lr,
                weight_decay=config.optimization.weight_decay,
            )
        elif os.environ.get("WD_ENCODER"):
            # DP-T applies weight decay 1e-3 to the policy network but only
            # 1e-6 to the observation encoder. Flat decay on a ResNet encoder
            # is not what they run, so split the groups when WD_ENCODER is set.
            # WD_NODECAY=1 additionally reproduces their minGPT-style grouping:
            # biases, norm gains and embeddings are exempt from decay.
            groups = []
            if os.environ.get("WD_NODECAY"):
                _norms = (torch.nn.LayerNorm, torch.nn.GroupNorm,
                          torch.nn.Embedding, torch.nn.BatchNorm1d,
                          torch.nn.BatchNorm2d)
                decay, no_decay, seen = [], [], set()
                for _mn, _m in self.flow_map.named_modules():
                    for _pn, _p in _m.named_parameters(recurse=False):
                        if id(_p) in seen:
                            continue
                        seen.add(id(_p))
                        if _pn.endswith("bias") or isinstance(_m, _norms):
                            no_decay.append(_p)
                        else:
                            decay.append(_p)
                groups += [
                    {"params": decay,
                     "weight_decay": config.optimization.weight_decay},
                    {"params": no_decay, "weight_decay": 0.0},
                ]
                loguru.logger.info(
                    f"WD_NODECAY: {len(decay)} decayed / {len(no_decay)} exempt"
                )
            else:
                groups.append({"params": list(self.flow_map.parameters()),
                               "weight_decay": config.optimization.weight_decay})
            groups.append({"params": list(self.encoder.parameters()),
                           "weight_decay": float(os.environ["WD_ENCODER"])})
            self.optimizer = torch.optim.AdamW(
                groups,
                lr=config.optimization.lr,
                betas=(
                    config.optimization.adam_beta1,
                    config.optimization.adam_beta2,
                ),
            )
        else:
            self.optimizer = torch.optim.AdamW(
                params,
                lr=config.optimization.lr,
                weight_decay=config.optimization.weight_decay,
                betas=(
                    config.optimization.adam_beta1,
                    config.optimization.adam_beta2,
                ),
            )

        # Store obs keys if using image observations (for CUDA graph compatibility)
        if hasattr(config.task, "shape_meta") and "obs" in config.task.shape_meta:
            self.obs_keys = sorted(config.task.shape_meta["obs"].keys())
        else:
            self.obs_keys = None

        # Compile training and sampling functions for faster execution
        self.use_compile = config.optimization.use_compile
        self.compile_mode = config.optimization.compile_mode
        self.__compile__()

    def __compile__(self):
        """Compile training and inference functions."""
        loguru.logger.info(
            f"Compile: {self.use_compile} | "
            f"Compile mode: {self.compile_mode} | "
            f"CUDA graphs: {self.use_cudagraphs}"
        )

        # Create the update function that includes optimizer operations
        self._update_impl = self._create_update_impl()
        self._sample_fn = self._sample_impl

        # Step 1: Setup CUDA graphs - copy params to detached models
        if self.use_cudagraphs:
            from tensordict import from_module

            loguru.logger.info(
                "Setting up CUDA graphs - copying parameters to detached models"
            )
            # Copy params to detached models without gradients
            from_module(self.flow_map).data.to_module(self.flow_map_detach)
            from_module(self.encoder).data.to_module(self.encoder_detach)
            from_module(self.flow_map_ema).data.to_module(self.flow_map_ema_detach)
            from_module(self.encoder_ema).data.to_module(self.encoder_ema_detach)

            # Wrap sampler with context manager to use detached models
            self._sample_fn = self._inference_mode()(self._sample_impl)

        # Step 2: Compile with torch.compile
        if self.use_compile:
            loguru.logger.info(
                "Compiling entire update loop (forward+backward+optimizer) with torch.compile"
            )
            self._compiled_update = torch.compile(
                self._update_impl, mode=self.compile_mode
            )
            loguru.logger.info("Compiling sampler with torch.compile")
            self._compiled_sampler = torch.compile(
                self._sample_fn, mode=self.compile_mode
            )
            loguru.logger.info("Successfully compiled models")
        else:
            self._compiled_update = self._update_impl
            self._compiled_sampler = self._sample_fn

        # Step 3: Wrap with CudaGraphModule for CUDA graph capture
        if self.use_cudagraphs:
            from tensordict.nn import CudaGraphModule

            loguru.logger.info(
                "Wrapping update function with CudaGraphModule for CUDA graph capture"
            )
            # Wrap the update function with CudaGraphModule
            # in_keys=[] means no input TensorDict, out_keys=[] means output TensorDict keys are inferred
            self._compiled_update = CudaGraphModule(
                self._compiled_update, in_keys=[], out_keys=[]
            )
            loguru.logger.info("CUDA graph setup complete")

    @contextmanager
    def _inference_mode(self):
        """Context manager to switch to inference models.

        For CUDA graphs: Swaps to detached models (no gradients).
        For regular mode: Sets models to eval mode temporarily.
        """
        if self.use_cudagraphs:
            # Swap to detached models for CUDA graphs
            flow_map_backup = self.flow_map
            encoder_backup = self.encoder
            flow_map_ema_backup = self.flow_map_ema
            encoder_ema_backup = self.encoder_ema

            self.flow_map = self.flow_map_detach
            self.encoder = self.encoder_detach
            self.flow_map_ema = self.flow_map_ema_detach
            self.encoder_ema = self.encoder_ema_detach

            try:
                yield
            finally:
                # Restore original models
                self.flow_map = flow_map_backup
                self.encoder = encoder_backup
                self.flow_map_ema = flow_map_ema_backup
                self.encoder_ema = encoder_ema_backup
        else:
            # For regular mode, temporarily set to eval
            was_training = self.flow_map.training
            try:
                self.flow_map.eval()
                self.encoder.eval()
                self.flow_map_ema.eval()
                self.encoder_ema.eval()
                yield
            finally:
                # Restore training mode if it was training
                if was_training:
                    self.flow_map.train()
                    self.encoder.train()

    def _sync_detached_models(self):
        """Synchronize detached models with main models for CUDA graphs."""
        if self.use_cudagraphs:
            from tensordict import from_module

            # Copy parameters without gradients using tensordict
            from_module(self.flow_map).data.to_module(self.flow_map_detach)
            from_module(self.encoder).data.to_module(self.encoder_detach)
            from_module(self.flow_map_ema).data.to_module(self.flow_map_ema_detach)
            from_module(self.encoder_ema).data.to_module(self.encoder_ema_detach)

    def _create_update_impl(self):
        """Create the update implementation function that will be compiled.

        This function includes the entire update loop: forward, backward, gradient clipping,
        optimizer step, zero_grad, and EMA update.

        Returns:
            A function that performs the complete update step
        """

        def update_impl(data: TensorDict):
            """Complete update step (can be compiled).

            Args:
                data: TensorDict containing 'act', 'delta_t', and either 'obs' or flattened 'obs_*' keys

            Returns:
                TensorDict containing loss, grad_norm, and other metrics
            """
            # Extract from TensorDict
            act = data["act"]
            obs = data["obs"]
            delta_t = data["delta_t"]

            # Forward pass and compute loss
            loss, _info = self.loss_fn(
                self.config.optimization,
                self.flow_map,
                self.encoder,
                self.interpolant,
                act,
                obs,
                delta_t,
            )

            # Backward pass
            loss.backward()

            # Gradient clipping
            params = list(self.encoder.parameters()) + list(self.flow_map.parameters())
            if self.config.optimization.grad_clip_norm:
                grad_norm = nn.utils.clip_grad_norm_(
                    params, self.config.optimization.grad_clip_norm
                )
            else:
                grad_norm = torch.tensor(0.0, device=loss.device)

            # Optimizer step
            self.optimizer.step()
            self.optimizer.zero_grad()

            # EMA update
            if self.config.optimization.ema_rate < 1:
                self._ema_update_impl()

            # Return as TensorDict for CUDA graph compatibility (static shapes)
            result_dict = {
                "loss": loss.detach(),
                "grad_norm": grad_norm.detach(),
            }
            # pass scalar aux metrics from the loss (e.g. condreg/* diagnostics)
            for _ak, _av in (_info or {}).items():
                if isinstance(_av, (int, float)):
                    result_dict[_ak] = torch.tensor(float(_av), device=loss.device)
            _nu_raw = getattr(self.flow_map, "learn_nu_raw", None)
            if _nu_raw is not None:
                result_dict["nu"] = (
                    0.5 + torch.nn.functional.softplus(_nu_raw)
                ).detach()
            if hasattr(self.flow_map, "nu_cond_step"):
                self.flow_map.nu_cond_step += 1.0
                _st = getattr(self.flow_map, "_nu_stats", None)
                if _st is not None:
                    result_dict["nu"] = _st[0]
                    result_dict["nu_p10"] = _st[1]
                    result_dict["nu_p90"] = _st[2]
            result = TensorDict(result_dict, batch_size=())
            return result

        return update_impl

    def _ema_decay(self):
        """Constant ema_rate, or DP-style power-law schedule when ema_power>0."""
        cfg = self.config.optimization
        if getattr(cfg, "ema_power", 0.0) <= 0.0:
            return cfg.ema_rate
        self._ema_step = getattr(self, "_ema_step", 0) + 1
        step = max(0, self._ema_step - 1)
        if step <= 0:
            return 0.0
        value = 1.0 - (1.0 + step / cfg.ema_inv_gamma) ** (-cfg.ema_power)
        return max(cfg.ema_min, min(value, cfg.ema_max))

    def _ema_update_impl(self):
        """EMA update implementation (can be part of compiled function)."""
        params = list(self.encoder.parameters()) + list(self.flow_map.parameters())
        params_ema = list(self.encoder_ema.parameters()) + list(
            self.flow_map_ema.parameters()
        )
        rate = self._ema_decay()
        with torch.no_grad():
            for p, p_ema in zip(params, params_ema, strict=False):
                p_ema.data.mul_(rate).add_(p.data, alpha=1.0 - rate)

    def update(
        self,
        act: torch.Tensor,
        obs: torch.Tensor | dict | TensorDict,
        delta_t: torch.Tensor,
    ):
        """Update the model parameters with a training batch.

        Args:
            act: Action tensor of shape (batch_size, Ta, act_dim)
            obs: Observation tensor of shape (batch_size, To, obs_dim) or dict of tensors for images
            delta_t: Time step differences of shape (batch_size,)

        Returns:
            Dictionary containing loss and gradient norm statistics
        """
        # Check batch size consistency for CUDA graphs
        if self.use_cudagraphs:
            if not hasattr(self, "_expected_batch_size"):
                self._expected_batch_size = act.shape[0]
            elif act.shape[0] != self._expected_batch_size:
                raise ValueError(
                    f"CUDA graphs require static batch sizes. "
                    f"Expected {self._expected_batch_size}, got {act.shape[0]}. "
                    f"Make sure your dataloader has drop_last=True."
                )

        # Mark CUDA graph step boundary if using compile
        if self.use_compile:
            torch.compiler.cudagraph_mark_step_begin()

        # Wrap inputs in TensorDict - simple flat structure only (no dicts or nested structures)
        data = TensorDict(
            {
                "act": act,
                "obs": obs,  # obs can be dict or tensor - TensorDict will handle it
                "delta_t": delta_t,
            },
            batch_size=act.shape[0],
        )
        result = self._compiled_update(data)

        # Convert TensorDict to regular dict with scalar values
        out = {
            "loss": result["loss"],
            "grad_norm": result["grad_norm"],
        }
        for _k in result.keys():
            if _k not in out:
                out[_k] = result[_k]
        return out

    def ema_update(self):
        """Update exponential moving average parameters."""
        params = list(self.encoder.parameters()) + list(self.flow_map.parameters())
        ema_params = list(self.encoder_ema.parameters()) + list(
            self.flow_map_ema.parameters()
        )
        rate = self._ema_decay()
        with torch.no_grad():
            for p, p_ema in zip(params, ema_params, strict=False):
                p_ema.data.mul_(rate).add_(p.data, alpha=1.0 - rate)


    def _sample_impl(
        self,
        config,
        flow_map,
        encoder,
        act_0: torch.Tensor,
        obs: torch.Tensor,
    ):
        """Internal sampling implementation (can be compiled).

        Args:
            config: Optimization config
            flow_map: Flow map model
            encoder: Encoder model
            act_0: Initial action tensor
            obs: Observation tensor

        Returns:
            Sampled action tensor
        """
        return self.sampler(config, flow_map, encoder, act_0, obs)

    def sample(
        self,
        act_0: torch.Tensor,
        obs: torch.Tensor,
        num_steps: int = -1,
        use_ema: bool = True,
    ):
        """Sample actions from the learned policy.

        Args:
            act_0: Initial action tensor of shape (batch_size, Ta, act_dim)
            obs: Observation tensor of shape (batch_size, To, obs_dim)
            num_steps: Number of sampling steps (default: use config value)
            use_ema: Whether to use EMA parameters for sampling

        Returns:
            Sampled action tensor of shape (batch_size, Ta, act_dim)
        """
        # Sync detached models if using CUDA graphs before inference
        if self.use_cudagraphs:
            self._sync_detached_models()

        # manually set num_steps if needed
        if num_steps >= 1:
            config = deepcopy(self.config.optimization)
            config.num_steps = int(num_steps)
        else:
            config = self.config.optimization

        # choose model
        # Note: For CUDA graphs, the _inference_mode context manager was already
        # applied during __compile__, so we use the original models here
        if self.config.optimization.ema_rate < 1:
            if use_ema:
                flow_map = self.flow_map_ema
                encoder = self.encoder_ema
            else:
                flow_map = self.flow_map
                encoder = self.encoder
        else:
            flow_map = self.flow_map
            encoder = self.encoder

        with torch.no_grad():
            # For CUDA graphs, _compiled_sampler already uses detached models
            # For regular mode, we temporarily switch to eval mode
            if not self.use_cudagraphs:
                with self._inference_mode():
                    act = self._compiled_sampler(config, flow_map, encoder, act_0, obs)
            else:
                act = self._compiled_sampler(config, flow_map, encoder, act_0, obs)
        return act

    def save(self, path: str, training_state: dict = None):
        """Save agent models to path.

        Args:
            path: Path to save checkpoint
            training_state: Optional dict with training state (n_gradient_step, best_metrics, eval_history)
        """
        # save flow map, encoder, encoder_ema, flow_map_ema, optimizer
        checkpoint = {
            "flow_map": self.flow_map.state_dict(),
            "encoder": self.encoder.state_dict(),
            "encoder_ema": self.encoder_ema.state_dict(),
            "flow_map_ema": self.flow_map_ema.state_dict(),
            "optimizer": self.optimizer.state_dict(),
        }

        # Add training state if provided
        if training_state is not None:
            checkpoint["training_state"] = training_state

        torch.save(checkpoint, path)

    def load(self, path: str, load_optimizer: bool = False):
        """Load agent models from path.

        Args:
            path: Path to load checkpoint from
            load_optimizer: Whether to load optimizer state

        Returns:
            training_state dict if available, None otherwise
        """
        # load flow map, encoder, encoder_ema, flow_map_ema
        state_dict = torch.load(
            path, map_location=self.config.optimization.device, weights_only=False
        )
        self.flow_map.load_state_dict(state_dict["flow_map"])
        self.encoder.load_state_dict(state_dict["encoder"])
        self.encoder_ema.load_state_dict(state_dict["encoder_ema"])
        self.flow_map_ema.load_state_dict(state_dict["flow_map_ema"])

        # Load optimizer state if requested and available
        if load_optimizer and "optimizer" in state_dict:
            self.optimizer.load_state_dict(state_dict["optimizer"])
            loguru.logger.info("Loaded optimizer state")

        # Recompile after loading to ensure compiled functions are up to date
        self.__compile__()

        # Return training state if available
        training_state = state_dict.get("training_state", None)
        if training_state:
            loguru.logger.info(
                f"Loaded training state from step {training_state.get('n_gradient_step', 'unknown')}"
            )

        return training_state

    def eval(self):
        """Set all models to evaluation mode."""
        self.flow_map.eval()
        self.encoder.eval()
        self.flow_map_ema.eval()
        self.encoder_ema.eval()

    def train(self):
        """Set all models to training mode."""
        self.flow_map.train()
        self.encoder.train()
        self.flow_map_ema.train()
        self.encoder_ema.train()
