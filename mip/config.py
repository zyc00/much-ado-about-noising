from dataclasses import dataclass, field


@dataclass
class LogConfig:
    log_dir: str
    wandb_mode: str
    project: str
    group: str
    exp_name: str
    eval_freq: int = 20000
    log_freq: int = 1000
    save_freq: int = 10000
    eval_episodes: int = 10
    save_video: bool = False


@dataclass
class OptimizationConfig:
    seed: int = 0
    loss_type: str = "flow"
    loss_scale: float = 100.0
    norm_type: str = "l2"
    lr: float = 1e-4
    weight_decay: float = 1e-5
    num_steps: int = 1
    sample_mode: str = "stochastic"  # "zero", "mean"
    t_two_step: float = 0.9
    discrete_dt: float = 0.01
    grad_clip_norm: float = 10.0
    ema_rate: float = 0.995
    # DP-style progressive (power-law) EMA. ema_power=0 keeps the constant
    # ema_rate above; >0 uses decay = 1-(1+step/inv_gamma)^-power capped at
    # ema_max, i.e. an averaging window that grows with training.
    ema_power: float = 0.0
    ema_inv_gamma: float = 1.0
    ema_max: float = 0.9999
    ema_min: float = 0.0
    adam_beta1: float = 0.9
    adam_beta2: float = 0.999
    batch_size: int = 1024
    gradient_steps: int = 300000
    warmup_ratio: float = 0.0
    rampup_ratio: float = 0.5
    min_value: float = 0.0
    max_value: float = 1.0
    model_path: str | None = None
    interp_type: str = "linear"  # "linear" or "trig"
    device: str = "cuda"
    use_compile: bool = True  # Whether to use torch.compile for acceleration
    compile_mode: str = (
        "default"  # Compile mode: "default", "reduce-overhead", "max-autotune"
    )
    use_cudagraphs: bool = False  # Whether to use CUDA graphs (requires static shapes)
    auto_resume: bool = True  # Whether to automatically resume from checkpoint
    cauchy_c: float = 0.2
    nu_cond_reg: float = 0.01  # learnnu_cond: deviation shrinkage weight
    nu_cond_warmup: int = 30000  # learnnu_cond: steps before nu(s) unfreezes
    xm_k: int = 4  # forward-XM best-of-K exploration (mip_xm / flow_xm)
    student_t_df: float = 2.0  # degrees of freedom for regression_student_t loss (df=1 -> Cauchy, df->inf -> Gaussian)


@dataclass
class NetworkConfig:
    network_type: str = "mlp"  # "mlp" or "cnn"
    gmm_k: int = 0  # >1 enables the MDN/GMM head on chiunet (regression_gmm)
    num_layers: int = 4
    emb_dim: int = 512
    dropout: float = 0.1
    encoder_dropout: float = 0.0
    encoder_type: str = "mlp"  # "mlp", "per_step_mlp", "identity"
    expansion_factor: int = 4
    timestep_emb_dim: int = 128
    timestep_emb_type: str = "positional"  # Type of timestep embedding
    # State encoder configs
    num_encoder_layers: int = 2  # Number of layers for MLP encoder
    # Image encoder configs
    rgb_model_name: str = "resnet18"
    use_seq: bool = True
    keep_horizon_dims: bool = True
    # Transformer specific configs
    n_heads: int = 6
    n_cond_layers: int = 0
    attn_dropout: float = 0.1
    # UNet specific configs
    model_dim: int = 256
    kernel_size: int = 5
    cond_predict_scale: bool = True
    obs_as_global_cond: bool = True
    dim_mult: list[int] | None = None
    norm_type: str = "groupnorm"
    attention: bool = False
    # RNN specific configs
    rnn_type: str = "LSTM"  # "LSTM" or "GRU"
    max_freq: float = 100.0


@dataclass
class TaskConfig:
    env_name: str = "lift"
    obs_type: str = "state"
    env_type: str = "ph"
    abs_action: bool = True
    action_type: str = "absolute"  # "absolute", "delta", or "relative"
    # Dataset configuration - either HuggingFace or local path
    dataset_repo: str | None = (
        None  # HuggingFace repository ID (e.g., "ChaoyiPan/mip-dataset")
    )
    dataset_filename: str | None = (
        None  # Path within the repository (e.g., "robomimic/lift/ph/image.hdf5")
    )
    dataset_path: str | None = (
        None  # Local path (deprecated, use dataset_repo/dataset_filename)
    )
    max_episode_steps: int = 400
    obs_keys: list[str] = field(
        default_factory=lambda: [
            "object",
            "robot0_eef_pos",
            "robot0_eef_quat",
            "robot0_gripper_qpos",
        ]
    )
    obs_dim: int = -1
    act_dim: int = 10
    phase_indicator: bool = False  # if True, append 3-dim phase one-hot to action target (aux output task)
    pose_indicator: bool = False  # if True (implies rot_indicator), also append gate-frame frame-position offset (3ch)
    normjit: bool = False  # if True, normal-direction obs jitter (annulus pull-back regularizer)
    knnsmooth: bool = False  # if True, replace action targets with kNN-conditional means (target-side smoothing)
    mixup: bool = False  # if True, local cross-demo kNN mixup (interstitial-field training)
    despike: bool = False  # if True, replace top-decile temporal-residual action steps with local median (crowding-out test)
    rot_indicator: bool = False  # if True, append in-hand orientation-error rotvec (3ch, to demo insertion frame) to action target
    progress_indicator: bool = False  # if True, append normalized episode-progress ramp t/T to action target (dense anti-starvation aux output)
    tc_indicator: bool = False  # if True, append clipped signed time-to-closure ramp to action target
    fwd_indicator: bool = False  # if True, append 8-step forward state delta (obs_dim channels) to action target
    phase_input: bool = False  # if True, append 3-dim phase one-hot to OBS (condition policy on phase)
    obs_steps: int = 2
    act_steps: int = 8
    horizon: int = 10  # Prediction horizon (typically obs_steps + act_steps)
    num_envs: int = 1
    save_video: bool = False
    shape_meta: dict = field(default_factory=dict)
    render_obs_key: str = "agentview_image"
    val_dataset_percentage: float = 0.0
    # Image observation settings
    rgb_model: str = "resnet18"
    resize_shape: list[int] | None = None
    crop_shape: list[int] | None = None
    random_crop: bool = True
    use_group_norm: bool = True
    use_seq: bool = True


@dataclass
class Config:
    optimization: OptimizationConfig
    network: NetworkConfig
    task: TaskConfig
    log: LogConfig
    mode: str = "train"  # "train" or "eval"
