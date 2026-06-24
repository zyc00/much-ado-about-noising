import gymnasium as gym
import numpy as np
from gymnasium.spaces import Box

from robomimic.envs.env_robosuite import EnvRobosuite


class RobomimicLowdimWrapper(gym.Env):
    def __init__(
        self,
        env: EnvRobosuite,
        obs_keys: list[str] = None,
        init_state: np.ndarray | None = None,
        render_hw=(256, 256),
        render_camera_name="agentview",
    ):
        if obs_keys is None:
            obs_keys = [
                "object",
                "robot0_eef_pos",
                "robot0_eef_quat",
                "robot0_gripper_qpos",
            ]
        self.env = env
        self.obs_keys = obs_keys
        self.init_state = init_state
        self.render_hw = render_hw
        self.render_camera_name = render_camera_name
        self.seed_state_map = {}
        self._seed = None

        # setup spaces
        low = np.full(env.action_dimension, fill_value=-1)
        high = np.full(env.action_dimension, fill_value=1)
        self.action_space = Box(low=low, high=high, shape=low.shape, dtype=low.dtype)
        obs_example = self.get_observation()
        low = np.full_like(obs_example, fill_value=-1)
        high = np.full_like(obs_example, fill_value=1)
        self.observation_space = Box(
            low=low, high=high, shape=low.shape, dtype=low.dtype
        )

    def get_observation(self):
        raw_obs = self.env.get_observation()
        obs = np.concatenate([raw_obs[key] for key in self.obs_keys], axis=0)
        import os as _os
        if _os.environ.get("EVAL_PHASE_INPUT", "0") == "1":
            obs = np.concatenate([obs, self._oracle_phase()]).astype(obs.dtype)
        return obs

    def _oracle_phase(self):
        # sim-based 3-class phase for the phase_input eval (privileged oracle):
        # 0=reach (frame not lifted), 1=lift/align (lifted, far from hole),
        # 2=insert (lifted, frame_mount above hole xy). Mirrors the training phase.
        import numpy as _np
        ph = _np.zeros(3, dtype=_np.float32)
        try:
            sim = self.env.env.sim
            fz = sim.data.site_xpos[sim.model.site_name2id("frame_mount_site")]
            hc = _np.mean([sim.data.geom_xpos[sim.model.geom_name2id(f"stand_wall{i}")]
                           for i in range(4)], axis=0)
            lifted = fz[2] > 0.86
            near = _np.linalg.norm((fz - hc)[:2]) < 0.05
            ph[2 if (lifted and near) else (1 if lifted else 0)] = 1.0
        except Exception:
            ph[0] = 1.0
        return ph

    def seed(self, seed=None):
        np.random.seed(seed=seed)
        self._seed = seed

    def reset(self, seed=None, options=None):
        # Handle seed parameter from Gymnasium API
        if seed is not None:
            self._seed = seed

        if self.init_state is not None:
            # always reset to the same state
            # to be compatible with gym
            self.env.reset_to({"states": self.init_state})
        elif self._seed is not None:
            # reset to a specific seed
            seed = self._seed
            if seed in self.seed_state_map:
                # env.reset is expensive, use cache
                self.env.reset_to({"states": self.seed_state_map[seed]})
            else:
                # robosuite's initializes all use numpy global random state
                np.random.seed(seed=seed)
                self.env.reset()
                state = self.env.get_state()["states"]
                self.seed_state_map[seed] = state
            self._seed = None
        else:
            # random reset
            self.env.reset()

        # Reproduce the scripted demos' leading settle (frame free-falls on reset,
        # 10x zero actions, NOT recorded -> training init is the SETTLED state).
        # Without this the policy starts mid-fall (OOD) and brittle models score ~0.
        import os as _os
        n_settle = int(_os.environ.get("EVAL_SETTLE_STEPS", "0"))
        if n_settle > 0:
            for _ in range(n_settle):
                self.env.step(np.zeros(self.env.action_dimension))

        # return obs and info (Gymnasium API requires both)
        obs = self.get_observation()
        info = {}
        return obs, info

    def step(self, action):
        raw_obs, reward, done, info = self.env.step(action)
        obs = np.concatenate([raw_obs[key] for key in self.obs_keys], axis=0)
        # expose frame-assembled (insertion success) for ToolHang so eval can
        # report insertion SR separately from the full-task (hang) reward
        base = getattr(self.env, "env", None)
        if base is not None and hasattr(base, "_check_frame_assembled"):
            try:
                info = dict(info)
                info["frame_assembled"] = bool(base._check_frame_assembled())
            except Exception:
                pass
        return obs, reward, done, info

    def render(self, mode="rgb_array"):
        h, w = self.render_hw
        return self.env.render(
            mode=mode, height=h, width=w, camera_name=self.render_camera_name
        )
