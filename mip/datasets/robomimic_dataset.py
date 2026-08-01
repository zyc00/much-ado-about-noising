"""Robomimic state dataset.

Author: Chaoyi Pan
Date: 2025-10-03
"""

import concurrent.futures
import os
from collections import defaultdict

import h5py
import numpy as np


def _rj_set(z, i, v):
    z = z.copy()
    z[0, i:i + len(v)] = v
    return z



import torch
import zarr
from huggingface_hub import hf_hub_download
from loguru import logger
from scipy.spatial.transform import Rotation
from tqdm import tqdm

from mip.action_utils import action_abs_to_rel
from mip.dataset_utils import (
    CompositeNormalizer,
    IdentityNormalizer,
    ImageNormalizer,
    MinMaxNormalizer,
    ReplayBuffer,
    RotationTransformer,
    SequenceSampler,
    dict_apply,
)
from mip.datasets.base import BaseDataset
from mip.datasets.imagecodecs import register_codecs

register_codecs()


def make_dataset(task_config, mode="train"):
    # Explicit local path takes precedence over HuggingFace download
    if getattr(task_config, "dataset_path", None) is not None:
        dataset_path = os.path.expanduser(task_config.dataset_path)
        logger.info(f"Using local dataset: {dataset_path}")
    elif getattr(task_config, "dataset_repo", None) is not None and getattr(
        task_config, "dataset_filename", None
    ) is not None:
        # Auto-download from HuggingFace
        logger.info(
            f"Downloading dataset from {task_config.dataset_repo}/{task_config.dataset_filename}"
        )
        dataset_path = hf_hub_download(
            repo_id=task_config.dataset_repo,
            filename=task_config.dataset_filename,
            repo_type="dataset",
        )
        logger.info(f"Downloaded dataset to: {dataset_path}")
    else:
        raise ValueError(
            "Either dataset_repo/dataset_filename or dataset_path must be provided"
        )

    action_type = getattr(task_config, "action_type", "absolute")
    if task_config.env_name in ["can", "lift", "square", "tool_hang", "transport"]:
        if task_config.obs_type == "state":
            return RobomimicDataset(
                dataset_path,
                horizon=task_config.horizon,
                obs_keys=task_config.obs_keys,
                pad_before=task_config.obs_steps - 1,
                pad_after=task_config.act_steps - 1,
                abs_action=task_config.abs_action,
                action_type=action_type,
                obs_steps=task_config.obs_steps,
                mode=mode,
                val_dataset_percentage=task_config.val_dataset_percentage,
                phase_indicator=getattr(task_config, "phase_indicator", False),
                phase_input=getattr(task_config, "phase_input", False),
                progress_indicator=getattr(task_config, "progress_indicator", False),
                rot_indicator=getattr(task_config, "rot_indicator", False),
                despike=getattr(task_config, "despike", False),
                mixup=getattr(task_config, "mixup", False),
                knnsmooth=getattr(task_config, "knnsmooth", False),
                normjit=getattr(task_config, "normjit", False),
                pose_indicator=getattr(task_config, "pose_indicator", False),
                tc_indicator=getattr(task_config, "tc_indicator", False),
                fwd_indicator=getattr(task_config, "fwd_indicator", False),
            )
        elif task_config.obs_type == "image":
            return RobomimicImageDataset(
                dataset_path,
                horizon=task_config.horizon,
                shape_meta=task_config.shape_meta,
                n_obs_steps=task_config.obs_steps,
                pad_before=task_config.obs_steps - 1,
                pad_after=task_config.act_steps - 1,
                abs_action=task_config.abs_action,
                val_dataset_percentage=task_config.val_dataset_percentage,
                mode=mode,
            )
        else:
            raise ValueError(f"Invalid observation type: {task_config.obs_type}")
    else:
        raise ValueError(f"Environment {task_config.env_name} not supported")


class RobomimicDataset(BaseDataset):
    def __init__(
        self,
        dataset_dir,
        horizon=1,
        pad_before=0,
        pad_after=0,
        obs_keys=("object", "robot0_eef_pos", "robot0_eef_quat", "robot0_gripper_qpos"),
        abs_action=False,
        action_type="absolute",
        obs_steps=2,
        rotation_rep="rotation_6d",
        val_dataset_percentage=0.0,
        mode="train",
        use_key_state_for_val: bool = False,
        phase_indicator: bool = False,
        phase_input: bool = False,
        progress_indicator: bool = False,
        rot_indicator: bool = False,
        despike: bool = False,
        mixup: bool = False,
        knnsmooth: bool = False,
        normjit: bool = False,
        pose_indicator: bool = False,
        tc_indicator: bool = False,
        fwd_indicator: bool = False,
    ):
        super().__init__()
        self.rotation_transformer = RotationTransformer(
            from_rep="axis_angle", to_rep=rotation_rep
        )
        self.val_dataset_percentage = val_dataset_percentage
        self.mode = mode
        self.action_type = action_type
        self.phase_indicator = phase_indicator
        self.phase_input = phase_input
        self.progress_indicator = progress_indicator
        self.rot_indicator = rot_indicator
        self.despike = despike
        self.mixup = mixup
        self.knnsmooth = knnsmooth
        self.normjit = normjit
        self.pose_indicator = pose_indicator
        if pose_indicator:
            self.rot_indicator = True  # pose pre-pass reuses the rot consensus frame
        self.tc_indicator = tc_indicator
        self.fwd_indicator = fwd_indicator
        self.obs_steps = obs_steps

        self.replay_buffer = ReplayBuffer.create_empty_numpy()
        with h5py.File(dataset_dir) as file:
            demos = file["data"]
            total_demos = len(demos)

            # Calculate split indices
            if val_dataset_percentage > 0.0:
                val_count = int(total_demos * val_dataset_percentage)
                train_count = total_demos - val_count

                # Use deterministic split based on indices
                if mode == "train":
                    demo_indices = list(range(train_count))
                elif mode == "val":
                    demo_indices = list(range(train_count, total_demos))
                else:
                    raise ValueError(f"Invalid mode: {mode}. Must be 'train' or 'val'")
            else:
                # Use all data for training when no validation split
                demo_indices = list(range(total_demos))

            if use_key_state_for_val:
                import robomimic.utils.env_utils as EnvUtils
                import robomimic.utils.file_utils as FileUtils
                import robomimic.utils.obs_utils as ObsUtils

                # Initialize observation utilities with dummy spec
                dummy_spec = {
                    "obs": {
                        "low_dim": ["robot0_eef_pos"],
                        "rgb": [],
                    },
                }
                ObsUtils.initialize_obs_utils_with_obs_specs(
                    obs_modality_specs=dummy_spec
                )

                # Create environment from dataset metadata
                env_meta = FileUtils.get_env_metadata_from_dataset(
                    dataset_path=dataset_dir
                )
                env = EnvUtils.create_env_from_metadata(
                    env_meta=env_meta, render=False, render_offscreen=False
                )

                # Check if this is a robosuite environment
                is_robosuite_env = EnvUtils.is_robosuite_env(env_meta)

            if self.rot_indicator:
                # demo-consensus insertion frame: frame quat shortly before first release,
                # sign-aligned and averaged over demos (the branch-rule reference)
                _quats = []
                _offs = []
                for i in demo_indices:
                    d_ = demos[f"demo_{i}"]
                    a_ = d_["actions"][:].astype(np.float32)
                    g_ = a_[:, 6]
                    T_ = len(a_)
                    cl_ = [t for t in range(1, T_) if g_[t - 1] < 0 and g_[t] >= 0]
                    op_ = [t for t in range(1, T_) if g_[t - 1] >= 0 and g_[t] < 0]
                    if not cl_:
                        continue
                    r1_ = next((t for t in op_ if t > cl_[0]), None)
                    if r1_ is None or r1_ < 20:
                        continue
                    q_ = d_["obs"]["object"][r1_ - 20, 17:21].astype(np.float32)
                    q_ = q_ / (np.linalg.norm(q_) + 1e-9)
                    if _quats and np.dot(q_, _quats[0]) < 0:
                        q_ = -q_
                    _quats.append(q_)
                    _offs.append(
                        d_["obs"]["object"][r1_ - 1, 21:24].astype(np.float32)
                        - d_["obs"]["object"][r1_ - 1, 7:10].astype(np.float32)
                    )
                _qmu = np.mean(np.stack(_quats), axis=0)
                self._rot_qmu = _qmu / (np.linalg.norm(_qmu) + 1e-9)
                self._gate_off = np.median(np.stack(_offs), axis=0)

            for i in tqdm(demo_indices, desc=f"Loading {mode} hdf5 to ReplayBuffer"):
                demo = demos[f"demo_{i}"]

                if use_key_state_for_val:
                    states = demo["states"][:]
                    # Prepare initial state for environment reset
                    initial_state = {"states": states[0]}
                    if is_robosuite_env:
                        initial_state["model"] = demo.attrs["model_file"]
                        initial_state["ep_meta"] = demo.attrs.get("ep_meta", None)

                    # Reset environment to initial state
                    env.reset_to(initial_state)

                    # Evaluate key states in the trajectory
                    for _j, state in enumerate(states):
                        env.reset_to({"states": state})

                        # Get distance between frame and stand (example evaluation metric)
                        frame_site_name = "frame_tip_site"
                        stand_site_name = "stand_mount_site"

                        frame_site_pos = env.sim.data.site_xpos[
                            env.obj_site_id[frame_site_name]
                        ]
                        stand_site_pos = env.sim.data.site_xpos[
                            env.obj_site_id[stand_site_name]
                        ]
                        distance = np.linalg.norm(frame_site_pos - stand_site_pos)
                        logger.debug(distance)
                    exit()

                if self.despike:
                    # replace tremor spikes (top-decile |a_t - temporal median| steps,
                    # threshold computed dataset-wide on first pass) with the local median;
                    # leaves >=90% of steps untouched — data-side crowding-out test
                    raw_a = demo["actions"][:].astype(np.float32)
                    med = np.stack([np.median(raw_a[max(0, i - 2):i + 3], axis=0)
                                    for i in range(len(raw_a))])
                    dev_ = np.linalg.norm(raw_a - med, axis=1)
                    if not hasattr(self, "_spike_thr"):
                        _all = []
                        for j2 in demo_indices:
                            a2 = demos[f"demo_{j2}"]["actions"][:].astype(np.float32)
                            m2 = np.stack([np.median(a2[max(0, i - 2):i + 3], axis=0)
                                           for i in range(len(a2))])
                            _all.append(np.linalg.norm(a2 - m2, axis=1))
                        self._spike_thr = float(np.quantile(np.concatenate(_all), 0.9))
                    mask = dev_ >= self._spike_thr
                    raw_a[mask] = med[mask]
                    demo = dict(demo)
                    demo["actions"] = raw_a
                episode = data_to_obs(
                    raw_obs=demo["obs"],
                    raw_actions=demo["actions"][:].astype(np.float32),
                    obs_keys=obs_keys,
                    abs_action=abs_action,
                    rotation_transformer=self.rotation_transformer,
                )
                if self.tc_indicator:
                    # signed time-to-closure ramp as AUX OUTPUT: steepest supervision through
                    # the lethal (settle) window; clipped so far-from-closure steps saturate
                    a_ = demo["actions"][:].astype(np.float32); g_ = a_[:, 6]; T_ = len(a_)
                    cl_ = [t for t in range(1, T_) if g_[t-1] < 0 and g_[t] >= 0]
                    c1_ = cl_[0] if cl_ else T_
                    tc = np.clip((np.arange(T_, dtype=np.float32) - c1_) / 50.0, -2.0, 2.0)[:, None]
                    episode["action"] = np.concatenate([episode["action"], tc], axis=-1)
                if self.fwd_indicator:
                    # k-step forward state delta as AUX OUTPUT: dense supervision along many
                    # state directions (anti-starvation richness beyond a 1-D ramp)
                    k_ = 8
                    st_ = episode["obs"].astype(np.float32)
                    fwd = np.concatenate([st_[k_:], np.repeat(st_[-1:], k_, axis=0)], axis=0) - st_
                    episode["action"] = np.concatenate([episode["action"], fwd], axis=-1)
                if self.rot_indicator:
                    # in-hand orientation error to the insertion frame as AUX OUTPUT:
                    # rotation vector (axis*angle) from current frame quat to the demo
                    # consensus frame = the corrective rotational displacement the retry
                    # servo must realize; magnitude clipped at 45deg so the align-window
                    # variation occupies most of the normalized range
                    fq = demo["obs"]["object"][:, 17:21].astype(np.float32)
                    fq = fq / (np.linalg.norm(fq, axis=1, keepdims=True) + 1e-9)
                    rv = (
                        Rotation.from_quat(fq).inv() * Rotation.from_quat(self._rot_qmu)
                    ).as_rotvec().astype(np.float32)
                    ang = np.linalg.norm(rv, axis=1, keepdims=True)
                    rv = rv * np.minimum(1.0, 0.785 / (ang + 1e-9))
                    episode["action"] = np.concatenate([episode["action"], rv], axis=-1)
                if self.pose_indicator:
                    # gate-frame position offset of the frame as AUX OUTPUT (the lateral/
                    # vertical alignment state), norm-clipped at 80mm
                    pv = (
                        demo["obs"]["object"][:, 21:24].astype(np.float32)
                        - demo["obs"]["object"][:, 7:10].astype(np.float32)
                        - self._gate_off[None]
                    )
                    pn = np.linalg.norm(pv, axis=1, keepdims=True)
                    pv = pv * np.minimum(1.0, 0.080 / (pn + 1e-9))
                    episode["action"] = np.concatenate([episode["action"], pv], axis=-1)
                if self.progress_indicator:
                    # progress ramp as AUX OUTPUT: normalized episode time t/T appended to
                    # the action target — dense, monotone, phase-resolving supervision that
                    # stays label-rich inside slow windows (anti-starvation aux)
                    T_ = len(episode["action"])
                    prog = (np.arange(T_, dtype=np.float32) / max(T_ - 1, 1))[:, None]
                    episode["action"] = np.concatenate([episode["action"], prog], axis=-1)
                if self.phase_indicator or self.phase_input:
                    # per-timestep 3-class phase one-hot:
                    # 0=reach (init->grasp), 1=lift+align (grasp->align_done), 2=insert
                    a = demo["actions"][:].astype(np.float32)
                    ez = demo["obs"]["robot0_eef_pos"][:, 2].astype(np.float32)
                    g = a[:, 6]; T_ = len(a)
                    cl = [t for t in range(1, T_) if g[t-1] < 0 and g[t] >= 0]
                    op = [t for t in range(1, T_) if g[t-1] >= 0 and g[t] < 0]
                    ph = np.zeros((T_, 3), dtype=np.float32)
                    if cl and op:
                        c1, o1 = cl[0], op[0]
                        ad = c1 + int(np.argmax(ez[c1:o1]))
                        ph[:c1, 0] = 1.0; ph[c1:ad, 1] = 1.0; ph[ad:, 2] = 1.0
                    else:
                        ph[:, 0] = 1.0
                    if self.phase_indicator:  # phase as AUX OUTPUT (append to action target)
                        episode["action"] = np.concatenate([episode["action"], ph], axis=-1)
                    if self.phase_input:  # phase as INPUT (append to obs)
                        episode["obs"] = np.concatenate([episode["obs"], ph], axis=-1)
                # Store EEF state for relative action conversion
                if action_type == "relative":
                    eef_pos = demo["obs"]["robot0_eef_pos"][:].astype(np.float32)
                    eef_quat = demo["obs"]["robot0_eef_quat"][:].astype(np.float32)
                    episode["eef_pos"] = eef_pos
                    episode["eef_quat"] = eef_quat
                    # Dual-arm: also store robot1 EEF state
                    if "robot1_eef_pos" in demo["obs"]:
                        episode["eef_pos_1"] = (
                            demo["obs"]["robot1_eef_pos"][:].astype(np.float32)
                        )
                        episode["eef_quat_1"] = (
                            demo["obs"]["robot1_eef_quat"][:].astype(np.float32)
                        )
                self.replay_buffer.add_episode(episode)

        sampler_keys = ["obs", "action"]
        if action_type == "relative":
            sampler_keys += ["eef_pos", "eef_quat"]
            # Dual-arm: include robot1 EEF keys
            if "eef_pos_1" in self.replay_buffer:
                sampler_keys += ["eef_pos_1", "eef_quat_1"]
        self.sampler = SequenceSampler(
            replay_buffer=self.replay_buffer,
            sequence_length=horizon,
            pad_before=pad_before,
            pad_after=pad_after,
            keys=sampler_keys,
        )

        self.horizon = horizon
        self.pad_before = pad_before
        self.pad_after = pad_after
        self.abs_action = abs_action
        self.obs_keys = list(obs_keys)

        # Store obs key dimensions for EEF state extraction during eval
        with h5py.File(dataset_dir) as file:
            demo0_obs = file["data"]["demo_0"]["obs"]
            self.obs_key_dims = {key: demo0_obs[key].shape[-1] for key in obs_keys}

        self.normalizer = self.get_normalizer()
        if getattr(self, "mixup", False):
            self._build_mixup_pairs()
        if getattr(self, "knnsmooth", False):
            self._build_knn_targets()
        if getattr(self, "normjit", False):
            self._build_knn_targets(k=16, store_only=True)

    def _build_knn_targets(self, k=8, store_only=False):
        """kNN-conditional-mean action targets: for each sequence, average the action
        chunks of its k=8 nearest obs-window neighbors (cross-demo candidates included,
        self included). Hands the network the smooth conditional mean instead of raw
        noisy labels — the target-side version of the MLP's implicit smoothing."""
        import torch as _t
        n = len(self.sampler)
        obs_w = []
        for i in range(n):
            smp = self.sampler.sample_sequence(i)
            obs_w.append(smp["obs"][: self.obs_steps].reshape(-1))
        X = _t.tensor(np.stack(obs_w), dtype=_t.float32)
        X = (X - X.mean(0)) / (X.std(0) + 1e-6)
        cand = np.arange(0, n, max(1, n // 16000))
        dev_ = "cuda" if _t.cuda.is_available() else "cpu"
        Xc = X[cand].to(dev_)
        nbrs = np.zeros((n, k), dtype=np.int64)
        B = 2048
        for b in range(0, n, B):
            d = _t.cdist(X[b:b + B].to(dev_), Xc)
            top = d.topk(k, largest=False).indices.cpu().numpy()
            nbrs[b:b + B] = cand[top]
        self._knn_nbrs = nbrs
        self._obs_std = np.asarray(X.std(0))
        logger.info(f"knn neighbors built over {n} sequences (k={k}, store_only={store_only})")

    def _build_mixup_pairs(self):
        """Cross-demo kNN pairing over sampler indices: for each sequence, the nearest
        obs-window among candidates from OTHER demos. Enables local mixup: interpolated
        (obs, action) pairs that determine the field BETWEEN thin support points."""
        import torch as _t
        n = len(self.sampler)
        obs_w, demo_ids = [], []
        for i in range(n):
            smp = self.sampler.sample_sequence(i)
            obs_w.append(smp["obs"][: self.obs_steps].reshape(-1))
            demo_ids.append(int(smp.get("demo_id", i) if isinstance(smp, dict) and "demo_id" in smp else -1))
        X = _t.tensor(np.stack(obs_w), dtype=_t.float32)
        X = (X - X.mean(0)) / (X.std(0) + 1e-6)
        cand = np.arange(0, n, max(1, n // 16000))
        Xc = X[cand]
        pair = np.zeros(n, dtype=np.int64)
        B = 2048
        dev_ = "cuda" if _t.cuda.is_available() else "cpu"
        Xc_d = Xc.to(dev_)
        for b in range(0, n, B):
            d = _t.cdist(X[b:b + B].to(dev_), Xc_d)
            # exclude near-identical (same trajectory point): distance floor
            d[d < 1e-3] = 1e9
            top = d.topk(4, largest=False).indices.cpu().numpy()
            for r in range(len(top)):
                pair[b + r] = cand[top[r][np.random.randint(1, 4)]]
        self._mix_pair = pair
        logger.info(f"mixup pairs built over {n} sequences ({len(cand)} candidates)")


    def density_resample_weights(self):
        """DENSITY_RESAMPLE hook: inverse-density sampling weights — windows
        in SPARSE regions of obs space (far cross-episode neighbors) are
        upsampled, dense regions downsampled, total count unchanged.
        w_i = clip((r_i/median_r)^DR_ALPHA, 1/DR_WMAX, DR_WMAX), r_i =
        cross-episode nearest-neighbor distance in the z-scored window
        metric (the D(r_nn) quantity of the epistemic-pocket analysis)."""
        import os as _os

        from scipy.spatial import cKDTree

        alpha = float(_os.environ.get("DR_ALPHA", "1.0"))
        wmax = float(_os.environ.get("DR_WMAX", "3.0"))
        ends = np.asarray(self.replay_buffer.episode_ends[:])
        try:
            S_all = self.replay_buffer["obs"]["state"][:]
        except (TypeError, IndexError, KeyError):
            S_all = self.replay_buffer["obs"][:]
        eid_step = np.searchsorted(ends, np.arange(len(S_all)), side="right")
        idx = np.clip(np.asarray([r[0] for r in self.sampler.indices]),
                      0, len(S_all) - 2)
        W = np.stack([S_all[idx], S_all[np.minimum(idx + 1, len(S_all) - 1)]],
                     axis=1).reshape(len(idx), -1)
        mu, sd = W.mean(0), W.std(0) + 1e-6
        Wn = (W - mu) / sd
        eid = eid_step[idx]
        tree = cKDTree(Wn)
        d, nb = tree.query(Wn, k=16)
        r = np.full(len(Wn), np.nan)
        for i in range(len(Wn)):
            m = eid[nb[i]] != eid[i]
            r[i] = d[i][m][0] if m.any() else d[i][-1]
        w = np.clip((r / np.median(r)) ** alpha, 1.0 / wmax, wmax)
        logger.info(f"density_resample: {len(w)} windows, w p10/50/90 = "
                    f"{np.percentile(w,10):.2f}/{np.median(w):.2f}/"
                    f"{np.percentile(w,90):.2f}, alpha={alpha} wmax={wmax}")
        return w


    def phase_resample_weights(self):
        """PHASE_RESAMPLE hook: per-window sampling weights that BOOST the
        align/insert parts and CUT reach/pick/lift, at unchanged total
        sample count (use with WeightedRandomSampler, num_samples=len(ds)).

        Phases from the gripper channel of the raw actions: open segments =
        reach/retreat; closed segments = lift/carry then align/insert. The
        last PR_TAIL fraction (default 0.4) of each CLOSED segment (the
        align+insert/hang part) gets weight PR_UP (default 3.0); everything
        else PR_DOWN (default 0.4)."""
        import os as _os

        up = float(_os.environ.get("PR_UP", "3.0"))
        down = float(_os.environ.get("PR_DOWN", "0.4"))
        tail = float(_os.environ.get("PR_TAIL", "0.4"))
        grip = np.asarray(self.replay_buffer["action"][:, -1])
        ends = np.asarray(self.replay_buffer.episode_ends[:])
        starts = np.concatenate([[0], ends[:-1]])
        w_step = np.full(len(grip), down, dtype=np.float64)
        for s0, e0 in zip(starts, ends):
            g = grip[s0:e0] > 0
            # closed segments
            i = 0
            while i < len(g):
                if g[i]:
                    j = i
                    while j < len(g) and g[j]:
                        j += 1
                    k0 = i + int((j - i) * (1.0 - tail))
                    w_step[s0 + k0:s0 + j] = up
                    i = j
                else:
                    i += 1
        # window weight = weight at the window's first executed step
        idx = np.asarray([r[0] for r in self.sampler.indices])
        idx = np.clip(idx, 0, len(w_step) - 1)
        w = w_step[idx]
        frac_up = float((w >= up).mean())
        logger.info(f"phase_resample: {len(w)} windows, boosted frac "
                    f"{frac_up:.3f}, up={up} down={down} tail={tail}")
        return w

    def undo_transform_action(self, action):
        # drop any appended phase-indicator dims (act_dim 13 -> 10); keep 20 (dual arm)
        if action.shape[-1] > 10 and action.shape[-1] != 20:
            action = action[..., :10]
        raw_shape = action.shape
        if raw_shape[-1] == 20:
            # dual arm
            action = action.reshape(-1, 2, 10)

        d_rot = action.shape[-1] - 4
        pos = action[..., :3]
        rot = action[..., 3 : 3 + d_rot]
        gripper = action[..., [-1]]
        rot = self.rotation_transformer.inverse(rot)
        uaction = np.concatenate([pos, rot, gripper], axis=-1)

        if raw_shape[-1] == 20:
            # dual arm
            uaction = uaction.reshape(*raw_shape[:-1], 14)

        return uaction

    def get_normalizer(self):
        state_normalizer = MinMaxNormalizer(
            self.replay_buffer["obs"][:]
        )  # (N, obs_dim)
        # CHAN_SCALE: channel-leverage rebalancing at fixed data. Multiplies
        # the NORMALIZED obs per channel group (object, eef_pos, eef_quat,
        # grip), consistently at train and eval (env var must be set for
        # both). The first Linear can absorb any diagonal, so the function
        # class is unchanged — only the training-dynamics leverage moves.
        _cs = os.environ.get("CHAN_SCALE", "")
        if _cs:
            _c = [float(v) for v in _cs.split(",")]
            _dim = state_normalizer.range.shape[0]
            assert _dim == 53 and len(_c) == 4, (_dim, _c)
            _m = np.ones(_dim, dtype=np.float32)
            _m[0:44], _m[44:47], _m[47:51], _m[51:53] = _c[0], _c[1], _c[2], _c[3]

            class _ChanScaled:
                def __init__(self, base, mult):
                    self.base, self.mult = base, mult

                def normalize(self, x):
                    return self.base.normalize(x) * self.mult

                def unnormalize(self, x):
                    return self.base.unnormalize(
                        np.asarray(x, dtype=np.float32) / self.mult)

            state_normalizer = _ChanScaled(state_normalizer, _m)

        if self.action_type == "relative":
            # For relative actions, compute relative action stats for normalizer.
            # Convert all actions abs->rel to get the data range, then use
            # CompositeNormalizer: MinMax for pos(0:3) and grip(9:10),
            # Identity for rot6d(3:9) (naturally bounded ~[-1,1]).
            all_actions = self.replay_buffer["action"][:]  # (N, 10)
            all_eef_pos = self.replay_buffer["eef_pos"][:]  # (N, 3)
            all_eef_quat = self.replay_buffer["eef_quat"][:]  # (N, 4)

            # Convert all actions to relative for normalizer fitting
            all_rotmat = Rotation.from_quat(all_eef_quat).as_matrix()  # (N, 3, 3)
            act_dim = all_actions.shape[-1]
            rel_actions = np.empty_like(all_actions)

            if act_dim == 20 and "eef_pos_1" in self.replay_buffer:
                # Dual-arm: convert each arm with its own EEF reference
                all_eef_pos_1 = self.replay_buffer["eef_pos_1"][:]
                all_eef_quat_1 = self.replay_buffer["eef_quat_1"][:]
                all_rotmat_1 = Rotation.from_quat(all_eef_quat_1).as_matrix()
                for i in range(len(all_actions)):
                    rel_actions[i, :10] = action_abs_to_rel(
                        all_eef_pos[i], all_rotmat[i], all_actions[i, :10]
                    )
                    rel_actions[i, 10:] = action_abs_to_rel(
                        all_eef_pos_1[i], all_rotmat_1[i], all_actions[i, 10:]
                    )
            else:
                for i in range(len(all_actions)):
                    rel_actions[i] = action_abs_to_rel(
                        all_eef_pos[i], all_rotmat[i], all_actions[i]
                    )

            if act_dim == 10:
                # Single arm: pos(3) + rot6d(6) + grip(1)
                action_normalizer = CompositeNormalizer(
                    normalizers=[
                        MinMaxNormalizer(rel_actions[..., :3]),
                        IdentityNormalizer(),
                        MinMaxNormalizer(rel_actions[..., 9:10]),
                    ],
                    dim_slices=[(0, 3), (3, 9), (9, 10)],
                )
            elif act_dim == 20:
                # Dual arm: two stacked 10D
                action_normalizer = CompositeNormalizer(
                    normalizers=[
                        MinMaxNormalizer(rel_actions[..., :3]),
                        IdentityNormalizer(),
                        MinMaxNormalizer(rel_actions[..., 9:10]),
                        MinMaxNormalizer(rel_actions[..., 10:13]),
                        IdentityNormalizer(),
                        MinMaxNormalizer(rel_actions[..., 19:20]),
                    ],
                    dim_slices=[
                        (0, 3),
                        (3, 9),
                        (9, 10),
                        (10, 13),
                        (13, 19),
                        (19, 20),
                    ],
                )
            else:
                raise ValueError(
                    f"Unsupported action dim {act_dim} for relative actions"
                )
        else:
            action_normalizer = MinMaxNormalizer(
                self.replay_buffer["action"][:]
            )  # (N, action_dim)

        return {"obs": {"state": state_normalizer}, "action": action_normalizer}

    def sample_to_data(self, sample):
        state = sample["obs"].astype(np.float32)
        state = self.normalizer["obs"]["state"].normalize(state)

        action = sample["action"].astype(np.float32)

        # On-the-fly abs→rel conversion for relative action type
        if self.action_type == "relative":
            # Reference frame: last observed step (UMI Convention B)
            ref_idx = self.obs_steps - 1
            eef_pos = sample["eef_pos"][ref_idx].astype(np.float64)
            eef_quat = sample["eef_quat"][ref_idx].astype(np.float64)
            eef_rotmat = Rotation.from_quat(eef_quat).as_matrix()

            act_dim = action.shape[-1]
            if act_dim == 20 and "eef_pos_1" in sample:
                # Dual arm: each arm uses its own EEF reference
                eef_pos_1 = sample["eef_pos_1"][ref_idx].astype(np.float64)
                eef_quat_1 = sample["eef_quat_1"][ref_idx].astype(np.float64)
                eef_rotmat_1 = Rotation.from_quat(eef_quat_1).as_matrix()
                action[:, :10] = action_abs_to_rel(eef_pos, eef_rotmat, action[:, :10])
                action[:, 10:] = action_abs_to_rel(
                    eef_pos_1, eef_rotmat_1, action[:, 10:]
                )
            elif act_dim == 20:
                # Dual arm fallback: same EEF for both
                action_2arm = action.reshape(-1, 2, 10)
                for arm_idx in range(2):
                    action_2arm[:, arm_idx] = action_abs_to_rel(
                        eef_pos, eef_rotmat, action_2arm[:, arm_idx]
                    )
                action = action_2arm.reshape(-1, 20)
            else:
                action = action_abs_to_rel(eef_pos, eef_rotmat, action)

        action = self.normalizer["action"].normalize(action)
        data = {
            "obs": {"state": state},
            "action": action,
        }
        return data

    def __str__(self) -> str:
        return f"Keys: {self.replay_buffer.keys()} Steps: {self.replay_buffer.n_steps} Episodes: {self.replay_buffer.n_episodes}"

    def __len__(self) -> int:
        return len(self.sampler)

    def __getitem__(self, idx: int) -> dict[str, torch.Tensor]:
        sample = self.sampler.sample_sequence(idx)
        data = self.sample_to_data(sample)
        if getattr(self, "normjit", False) and np.random.rand() < float(os.environ.get("NJ_P", "0.5")):
            # NORMAL-DIRECTION JITTER: perturb the obs window orthogonally to the local
            # data-manifold tangent (top-8 PCs of kNN neighbor differences), keep the
            # label. Trains annulus pull-back without touching on-manifold gains.
            o0 = data["obs"]["state"] if isinstance(data["obs"], dict) else data["obs"]
            base = o0[: self.obs_steps].reshape(-1).astype(np.float64)
            nb = []
            for j in self._knn_nbrs[idx][1:]:
                d2 = self.sampler.sample_sequence(int(j))
                nb.append(d2["obs"][: self.obs_steps].reshape(-1).astype(np.float64))
            NB = np.stack(nb) - base
            U, S, Vt = np.linalg.svd(NB, full_matrices=False)
            V = Vt[:8]
            g = np.random.randn(base.shape[0])
            g = g - V.T @ (V @ g)
            g = g / (np.linalg.norm(g) + 1e-9)
            eps = np.random.uniform(float(os.environ.get("NJ_LO", "0.1")), float(os.environ.get("NJ_HI", "0.4"))) * (
                float(np.linalg.norm(NB[0])) if os.environ.get("NJ_ABS", "0") == "0" else 1.0)
            pert = (eps * g).reshape(self.obs_steps, -1).astype(np.float32)
            if isinstance(data["obs"], dict):
                data["obs"]["state"] = data["obs"]["state"].copy()
                data["obs"]["state"][: self.obs_steps] += pert
            else:
                data["obs"] = data["obs"].copy()
                data["obs"][: self.obs_steps] += pert
        if getattr(self, "knnsmooth", False):
            accs = [data["action"]]
            for j in self._knn_nbrs[idx][1:]:
                d2 = self.sample_to_data(self.sampler.sample_sequence(int(j)))
                if d2["action"].shape == data["action"].shape:
                    accs.append(d2["action"])
            data["action"] = np.mean(accs, axis=0)
        if os.environ.get("RECOVJIT", "0") == "1" and np.random.rand() < float(os.environ.get("RJ_P", "0.5")):
            # RECOVERY JITTER: displace the eef-pos obs dims by a finite raw
            # offset (annulus scale) and correct the position deltas of the
            # first executed chunk steps so the label commands returning to
            # the demonstrated path — supervises a restoring annulus field
            # (the measured MIP property) instead of flat replay (OBSJIT).
            if not hasattr(self, "_rj_scale"):
                z_o, z_a = np.zeros((1, 53), np.float32), np.zeros((1, 10), np.float32)
                no_, na_ = self.normalizer["obs"]["state"], self.normalizer["action"]
                self._rj_scale = (
                    lambda v: (no_.normalize(_rj_set(z_o, 44, v)) - no_.normalize(z_o))[0, 44:47],
                    lambda v: (na_.normalize(_rj_set(z_a, 0, v)) - na_.normalize(z_a))[0, 0:3],
                )
            u = np.random.randn(3)
            u /= np.linalg.norm(u) + 1e-9
            if os.environ.get("RJ_NORMAL", "0") == "1":
                # project displacement orthogonal to the local motion tangent
                # (raw eef delta across the obs window); tangent-restoring
                # supervision is anti-progress (PART CDIV/CDV)
                _no = self.normalizer["obs"]["state"]
                _raw = _no.unnormalize(data["obs"]["state"][:2])
                tvec = _raw[1, 44:47] - _raw[0, 44:47]
                tn = np.linalg.norm(tvec)
                if tn > 1e-5:
                    tvec = tvec / tn
                    u = u - float(u @ tvec) * tvec
                    un = np.linalg.norm(u)
                    if un > 1e-6:
                        u = u / un
            dlt = np.random.uniform(float(os.environ.get("RJ_LO", "0.005")),
                                    float(os.environ.get("RJ_HI", "0.05"))) * u
            data["obs"]["state"] = data["obs"]["state"].copy()
            data["obs"]["state"][:, 44:47] += self._rj_scale[0](dlt)
            if os.environ.get("RJ_NOCORR", "0") != "1":
                k_ = int(os.environ.get("RJ_K", "4"))
                st_ = self.obs_steps - 1
                data["action"] = data["action"].copy()
                corr = self._rj_scale[1](dlt / k_)
                data["action"][st_:st_ + k_, 0:3] -= corr
        if getattr(self, "mixup", False) and np.random.rand() < 0.5:
            j = int(self._mix_pair[idx])
            data2 = self.sample_to_data(self.sampler.sample_sequence(j))
            lam = 0.5 + 0.5 * np.random.rand()
            def _mix(a, b):
                if isinstance(a, dict):
                    return {kk: _mix(a[kk], b[kk]) for kk in a}
                return lam * a + (1.0 - lam) * b if getattr(a, "shape", None) == getattr(b, "shape", None) else a
            for k in ("obs", "action"):
                if k in data and k in data2:
                    data[k] = _mix(data[k], data2[k])
        torch_data = dict_apply(data, torch.tensor)
        return torch_data


def data_to_obs(raw_obs, raw_actions, obs_keys, abs_action, rotation_transformer):
    obs = np.concatenate([raw_obs[key] for key in obs_keys], axis=-1).astype(np.float32)

    if abs_action:
        is_dual_arm = False
        if raw_actions.shape[-1] == 14:
            # dual arm
            raw_actions = raw_actions.reshape(-1, 2, 7)
            is_dual_arm = True

        pos = raw_actions[..., :3]
        rot = raw_actions[..., 3:6]
        gripper = raw_actions[..., 6:]
        rot = rotation_transformer.forward(rot)
        raw_actions = np.concatenate([pos, rot, gripper], axis=-1).astype(np.float32)

        if is_dual_arm:
            raw_actions = raw_actions.reshape(-1, 20)

    data = {"obs": obs, "action": raw_actions}
    return data


class RobomimicImageDataset(BaseDataset):
    def __init__(
        self,
        dataset_dir,
        shape_meta: dict,
        n_obs_steps=None,
        horizon=1,
        pad_before=0,
        pad_after=0,
        abs_action=False,
        rotation_rep="rotation_6d",
        val_dataset_percentage=0.0,
        mode="train",
    ):
        super().__init__()
        self.rotation_transformer = RotationTransformer(
            from_rep="axis_angle", to_rep=rotation_rep
        )
        self.val_dataset_percentage = val_dataset_percentage
        self.mode = mode

        self.replay_buffer = _convert_robomimic_to_replay(
            store=zarr.storage.MemoryStore(),
            shape_meta=shape_meta,
            dataset_path=dataset_dir,
            abs_action=abs_action,
            rotation_transformer=self.rotation_transformer,
            val_dataset_percentage=val_dataset_percentage,
            mode=mode,
        )

        rgb_keys = []
        lowdim_keys = []
        obs_shape_meta = shape_meta["obs"]
        for key, attr in obs_shape_meta.items():
            type = attr.get("type", "low_dim")
            if type == "rgb":
                rgb_keys.append(key)
            elif type == "low_dim":
                lowdim_keys.append(key)

        key_first_k = {}
        if n_obs_steps is not None:
            # only take first k obs from images
            for key in rgb_keys + lowdim_keys:
                key_first_k[key] = n_obs_steps
        self.sampler = SequenceSampler(
            replay_buffer=self.replay_buffer,
            sequence_length=horizon,
            pad_before=pad_before,
            pad_after=pad_after,
            key_first_k=key_first_k,
        )

        self.shape_meta = shape_meta
        self.rgb_keys = rgb_keys
        self.lowdim_keys = lowdim_keys
        self.abs_action = abs_action
        self.horizon = horizon
        self.pad_before = pad_before
        self.pad_after = pad_after
        self.n_obs_steps = n_obs_steps

        self.normalizer = self.get_normalizer()
        if getattr(self, "mixup", False):
            self._build_mixup_pairs()
        if getattr(self, "knnsmooth", False):
            self._build_knn_targets()
        if getattr(self, "normjit", False):
            self._build_knn_targets(k=16, store_only=True)

    def _build_knn_targets(self, k=8, store_only=False):
        """kNN-conditional-mean action targets: for each sequence, average the action
        chunks of its k=8 nearest obs-window neighbors (cross-demo candidates included,
        self included). Hands the network the smooth conditional mean instead of raw
        noisy labels — the target-side version of the MLP's implicit smoothing."""
        import torch as _t
        n = len(self.sampler)
        obs_w = []
        for i in range(n):
            smp = self.sampler.sample_sequence(i)
            obs_w.append(smp["obs"][: self.obs_steps].reshape(-1))
        X = _t.tensor(np.stack(obs_w), dtype=_t.float32)
        X = (X - X.mean(0)) / (X.std(0) + 1e-6)
        cand = np.arange(0, n, max(1, n // 16000))
        dev_ = "cuda" if _t.cuda.is_available() else "cpu"
        Xc = X[cand].to(dev_)
        nbrs = np.zeros((n, k), dtype=np.int64)
        B = 2048
        for b in range(0, n, B):
            d = _t.cdist(X[b:b + B].to(dev_), Xc)
            top = d.topk(k, largest=False).indices.cpu().numpy()
            nbrs[b:b + B] = cand[top]
        self._knn_nbrs = nbrs
        self._obs_std = np.asarray(X.std(0))
        logger.info(f"knn neighbors built over {n} sequences (k={k}, store_only={store_only})")

    def _build_mixup_pairs(self):
        """Cross-demo kNN pairing over sampler indices: for each sequence, the nearest
        obs-window among candidates from OTHER demos. Enables local mixup: interpolated
        (obs, action) pairs that determine the field BETWEEN thin support points."""
        import torch as _t
        n = len(self.sampler)
        obs_w, demo_ids = [], []
        for i in range(n):
            smp = self.sampler.sample_sequence(i)
            obs_w.append(smp["obs"][: self.obs_steps].reshape(-1))
            demo_ids.append(int(smp.get("demo_id", i) if isinstance(smp, dict) and "demo_id" in smp else -1))
        X = _t.tensor(np.stack(obs_w), dtype=_t.float32)
        X = (X - X.mean(0)) / (X.std(0) + 1e-6)
        cand = np.arange(0, n, max(1, n // 16000))
        Xc = X[cand]
        pair = np.zeros(n, dtype=np.int64)
        B = 2048
        dev_ = "cuda" if _t.cuda.is_available() else "cpu"
        Xc_d = Xc.to(dev_)
        for b in range(0, n, B):
            d = _t.cdist(X[b:b + B].to(dev_), Xc_d)
            # exclude near-identical (same trajectory point): distance floor
            d[d < 1e-3] = 1e9
            top = d.topk(4, largest=False).indices.cpu().numpy()
            for r in range(len(top)):
                pair[b + r] = cand[top[r][np.random.randint(1, 4)]]
        self._mix_pair = pair
        logger.info(f"mixup pairs built over {n} sequences ({len(cand)} candidates)")

    def get_normalizer(self):
        normalizer = defaultdict(dict)
        for key in self.lowdim_keys:
            normalizer["obs"][key] = MinMaxNormalizer(self.replay_buffer[key][:])
        for key in self.rgb_keys:
            normalizer["obs"][key] = ImageNormalizer()
        normalizer["action"] = MinMaxNormalizer(self.replay_buffer["action"][:])

        return normalizer

    def __str__(self) -> str:
        return f"Keys: {self.replay_buffer.keys()} Steps: {self.replay_buffer.n_steps} Episodes: {self.replay_buffer.n_episodes}"

    def __len__(self) -> int:
        return len(self.sampler)

    def __getitem__(self, idx: int) -> dict[str, torch.Tensor]:
        sample = self.sampler.sample_sequence(idx)

        # obs
        # to save RAM, only return first n_obs_steps of OBS
        # since the rest will be discarded anyway.
        # when self.n_obs_steps is None
        # this slice does nothing (takes all)
        T_slice = slice(self.n_obs_steps)

        obs_dict = {}
        for key in self.rgb_keys:
            # move channel last to channel first
            # T,H,W,C
            # convert uint8 image to float32
            obs_dict[key] = (
                np.moveaxis(sample[key][T_slice], -1, 1).astype(np.float32) / 255.0
            )
            # T,C,H,W
            del sample[key]
            obs_dict[key] = self.normalizer["obs"][key].normalize(obs_dict[key])

        for key in self.lowdim_keys:
            obs_dict[key] = sample[key][T_slice].astype(np.float32)
            del sample[key]
            obs_dict[key] = self.normalizer["obs"][key].normalize(obs_dict[key])

        # action
        action = sample["action"].astype(np.float32)
        action = self.normalizer["action"].normalize(action)

        torch_data = {
            "obs": dict_apply(obs_dict, torch.tensor),
            "action": torch.tensor(action),
        }
        return torch_data

    def undo_transform_action(self, action):
        raw_shape = action.shape
        if raw_shape[-1] == 20:
            # dual arm
            action = action.reshape(-1, 2, 10)

        d_rot = action.shape[-1] - 4
        pos = action[..., :3]
        rot = action[..., 3 : 3 + d_rot]
        gripper = action[..., [-1]]
        rot = self.rotation_transformer.inverse(rot)
        uaction = np.concatenate([pos, rot, gripper], axis=-1)

        if raw_shape[-1] == 20:
            # dual arm
            uaction = uaction.reshape(*raw_shape[:-1], 14)

        return uaction


def _convert_actions(raw_actions, abs_action, rotation_transformer):
    actions = raw_actions
    if abs_action:
        is_dual_arm = False
        if raw_actions.shape[-1] == 14:
            # dual arm
            raw_actions = raw_actions.reshape(-1, 2, 7)
            is_dual_arm = True

        pos = raw_actions[..., :3]
        rot = raw_actions[..., 3:6]
        gripper = raw_actions[..., 6:]
        rot = rotation_transformer.forward(rot)
        raw_actions = np.concatenate([pos, rot, gripper], axis=-1).astype(np.float32)

        if is_dual_arm:
            raw_actions = raw_actions.reshape(-1, 20)
        actions = raw_actions
    return actions


def _convert_robomimic_to_replay(
    store,
    shape_meta,
    dataset_path,
    abs_action,
    rotation_transformer,
    n_workers=None,
    max_inflight_tasks=None,
    val_dataset_percentage=0.0,
    mode="train",
):
    """Convert Robomimic dataset to ReplayBuffer.

    A ReplayBuffer is a `zarr.Group` or Dict[str, dict] that contains the following keys:
    - data: zarr.Group or Dict[str, dict]
        Contains the data. All data should be stored as numpy arrays with the same length.
    - meta: zarr.Group or Dict[str, dict]
        Contains key "episode_ends", which is a numpy array of shape (n_episodes,) that contains the
        end index of each episode in the data.

    Args:
    - store: zarr.Store
        zarr.MemoryStore()
    - shape_meta: dict
        Shape metadata of the dataset. Should contain keys 'obs', 'action'.
        For example:
        shape_meta = {
            "action": {"shape": [10, ]},
            "obs": {
                "agentview_image": {"shape": [84, 84, 3], "type": "rgb"},
                "robot0_eef_pos":  {"shape": [3, ],       "type": "low_dim"},
            }}
    - dataset_path: str
        Path to the Robomimic dataset
    - abs_action: bool
        Whether to use position or velocity control
    - rotation_transformer: RotationTransformer
        Rotation transformer to convert rotation representation
    """
    """ Dataset structure of Can-PH, as an example:
    - data
        - demo_0
            - actions  (118, 7)
            - dones     (118, )
            - next_obs
                - agentview_image  (118, 84, 84, 3)
                - object            (118, 14)
                - robot0_eef_pos   (118, 3)
                - robot0_eef_quat
                - robot0_eef_vel_ang
                - robot0_eef_vel_lin
                - robot0_eye_in_hand_image
                - robot0_gripper_qpos
                - robot0_gripper_qvel
                - robot0_joint_pos
                - robot0_joint_pos_cos
                - robot0_joint_pos_sin
                - robot0_joint_vel
            - obs
                ...
            - rewards   (118, )
            - states    (118, 71)
        - demo_1
        ...(x200 demos)
    - mask
        - 20_percent
        - 20_percent_train
        - 20_percent_valid
        - 50_percent
        - 50_percent_train
        - 50_percent_valid
        - train (180,)
        - valid (20,)

    Suppose that the `shape_meta` is:
    shape_meta = {
    "action": {"shape": [10, ]},
    "obs": {
        "agentview_image": {
            "shape": [3, 84, 84], "type": "rgb", },
        "robot0_eye_in_hand_image": {
            "shape": [3, 84, 84], "type": "rgb", },
        "robot0_eef_pos": {
            "shape": [3, ], "type": "low_dim", },
        "robot0_eef_quat": {
            "shape": [4, ], "type": "low_dim", },
        "robot0_gripper_qpos": {
            "shape": [2, ], "type": "low_dim", }, }}
    """

    import multiprocessing

    if n_workers is None:
        n_workers = multiprocessing.cpu_count()
    if max_inflight_tasks is None:
        max_inflight_tasks = n_workers * 5

    # parse shape_meta
    rgb_keys = []
    lowdim_keys = []
    # construct compressors and chunks
    obs_shape_meta = shape_meta["obs"]
    for key, attr in obs_shape_meta.items():
        shape = attr["shape"]
        type = attr.get("type", "low_dim")
        if type == "rgb":
            rgb_keys.append(key)
        elif type == "low_dim":
            lowdim_keys.append(key)
    # rgb_keys = ['agentview_image', 'robot0_eye_in_hand_image']
    # lowdim_keys = ['robot0_eef_pos', 'robot0_eef_quat', 'robot0_gripper_qpos']

    # create zarr group
    root = zarr.group(store)
    data_group = root.require_group("data", overwrite=True)
    meta_group = root.require_group("meta", overwrite=True)

    with h5py.File(dataset_path) as file:
        # count total steps
        demos = file["data"]
        total_demos = len(demos)

        # Calculate split indices
        if val_dataset_percentage > 0.0:
            val_count = int(total_demos * val_dataset_percentage)
            train_count = total_demos - val_count

            # Use deterministic split based on indices
            if mode == "train":
                demo_indices = list(range(train_count))
            elif mode == "val":
                demo_indices = list(range(train_count, total_demos))
            else:
                raise ValueError(f"Invalid mode: {mode}. Must be 'train' or 'val'")
        else:
            # Use all data for training when no validation split
            demo_indices = list(range(total_demos))

        episode_ends = []
        prev_end = 0
        for i in demo_indices:
            demo = demos[f"demo_{i}"]
            episode_length = demo["actions"].shape[0]
            episode_end = prev_end + episode_length
            prev_end = episode_end
            episode_ends.append(episode_end)
        n_steps = episode_ends[-1] if episode_ends else 0
        episode_starts = [0] + episode_ends[:-1]
        _ = meta_group.create_array(
            name="episode_ends",
            data=np.array(episode_ends, dtype=np.int64),
            compressor=None,
            overwrite=True,
        )

        # save lowdim data
        for key in tqdm(lowdim_keys + ["action"], desc=f"Loading {mode} lowdim data"):
            data_key = "obs/" + key
            if key == "action":
                data_key = "actions"
            this_data = []
            for i in demo_indices:
                demo = demos[f"demo_{i}"]
                this_data.append(demo[data_key][:].astype(np.float32))
            this_data = np.concatenate(this_data, axis=0) if this_data else np.array([])
            if key == "action":
                this_data = _convert_actions(
                    raw_actions=this_data,
                    abs_action=abs_action,
                    rotation_transformer=rotation_transformer,
                )
                assert this_data.shape == (n_steps,) + tuple(
                    shape_meta["action"]["shape"]
                )
            else:
                assert this_data.shape == (n_steps,) + tuple(
                    shape_meta["obs"][key]["shape"]
                )
            _ = data_group.create_array(
                name=key,
                data=this_data,
                chunks=this_data.shape,
                compressor=None,
                overwrite=True,
            )

        def img_copy(zarr_arr, zarr_idx, hdf5_arr, hdf5_idx):
            try:
                zarr_arr[zarr_idx] = hdf5_arr[hdf5_idx]
                # make sure we can successfully decode
                _ = zarr_arr[zarr_idx]
                return True
            except Exception:
                return False

        with tqdm(
            total=n_steps * len(rgb_keys),
            desc=f"Loading {mode} image data",
            mininterval=1.0,
        ) as pbar:
            # one chunk per thread, therefore no synchronization needed
            with concurrent.futures.ThreadPoolExecutor(
                max_workers=n_workers
            ) as executor:
                futures = set()
                for key in rgb_keys:
                    data_key = "obs/" + key
                    shape = tuple(shape_meta["obs"][key]["shape"])
                    c, h, w = shape
                    # Use None compressor for zarr v3 compatibility in tests
                    img_arr = data_group.require_dataset(
                        name=key,
                        shape=(n_steps, h, w, c),
                        chunks=(1, h, w, c),
                        compressor=None,
                        dtype=np.uint8,
                    )
                    for demo_list_idx, episode_idx in enumerate(demo_indices):
                        demo = demos[f"demo_{episode_idx}"]
                        hdf5_arr = demo["obs"][key]
                        for hdf5_idx in range(hdf5_arr.shape[0]):
                            if len(futures) >= max_inflight_tasks:
                                # limit number of inflight tasks
                                completed, futures = concurrent.futures.wait(
                                    futures,
                                    return_when=concurrent.futures.FIRST_COMPLETED,
                                )
                                for f in completed:
                                    if not f.result():
                                        raise RuntimeError("Failed to encode image!")
                                pbar.update(len(completed))

                            zarr_idx = episode_starts[demo_list_idx] + hdf5_idx
                            futures.add(
                                executor.submit(
                                    img_copy, img_arr, zarr_idx, hdf5_arr, hdf5_idx
                                )
                            )
                completed, futures = concurrent.futures.wait(futures)
                for f in completed:
                    if not f.result():
                        raise RuntimeError("Failed to encode image!")
                pbar.update(len(completed))

    replay_buffer = ReplayBuffer(root)
    return replay_buffer
