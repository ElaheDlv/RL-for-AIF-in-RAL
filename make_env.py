
from typing import Optional, Tuple
import numpy as np

def try_make_carla_env(obs_mode: str, action_space: str, steer_bins, seed: int = 0):
    """
    Attempt to construct user's CarEnv with a minimal config.
    Expected signature: CarEnv(cfg_dict)
    """
    try:
        from car_env import CarEnv  # user's env
    except Exception as e:
        return None, e
    cfg = {
        "image_shape": (160,160,1) if obs_mode=="grayroad" else (160,160,3),
        "obs_mode": obs_mode,             # expected to switch camera modality
        "discrete_actions": (action_space == "disc"),
        "steer_bins": steer_bins.tolist(),
        "max_steps": 600,
        "seed": seed,
    }
    if action_space == "cont":
        cfg["action_space"] = "continuous"
    try:
        env = CarEnv(cfg)
        return env, None
    except Exception as e:
        return None, e

# ---------------- Toy envs -----------------

def _rgb_from_gray(gray):
    rgb = np.repeat(gray, 3, axis=2)
    return rgb

def make_toy_env(obs_mode: str, action_space: str, steer_bins, seed: int = 0):
    import gymnasium as gym

    class ToyLaneKeep(gym.Env):
        metadata = {"render_modes": []}
        def __init__(self, seed=None):
            super().__init__()
            self.rng = np.random.default_rng(seed)
            self.max_steps = 300
            self.t = 0
            self.lateral_offset = 0.0
            self.heading = 0.0
            self.prev_steer = 0.0
            self.obs_mode = obs_mode
            self.action_space_kind = action_space
            if obs_mode == "grayroad":
                self.observation_space = gym.spaces.Box(low=0.0, high=1.0, shape=(160,160,1), dtype=np.float32)
            else:
                self.observation_space = gym.spaces.Box(low=0.0, high=1.0, shape=(160,160,3), dtype=np.float32)
            if action_space == "disc":
                self.action_space = gym.spaces.Discrete(len(steer_bins))
            else:
                self.action_space = gym.spaces.Box(low=-1.0, high=1.0, shape=(1,), dtype=np.float32)
            self.steer_bins = steer_bins

        def reset(self, *, seed=None, options=None):
            if seed is not None:
                self.rng = np.random.default_rng(seed)
            self.t = 0
            self.lateral_offset = self.rng.uniform(-0.5, 0.5)
            self.heading = self.rng.uniform(-0.05, 0.05)
            self.prev_steer = 0.0
            return self._obs(), {}

        def step(self, action):
            if self.action_space_kind == "disc":
                steer = float(self.steer_bins[int(action)])
            else:
                steer = float(np.clip(action, -1.0, 1.0)[0])
            self.heading += 0.02*steer + self.rng.normal(0, 0.001)
            self.lateral_offset += self.heading + self.rng.normal(0, 0.005)

            reward = -abs(self.lateral_offset) - 0.01*abs(steer - self.prev_steer)
            self.prev_steer = steer

            self.t += 1
            terminated = abs(self.lateral_offset) > 2.0
            truncated = self.t >= self.max_steps
            info = {}
            if terminated or truncated:
                info["success"] = bool(abs(self.lateral_offset) < 0.5)
                info["lane_deviation"] = float(abs(self.lateral_offset))
            return self._obs(), reward, terminated, truncated, info

        def _obs(self):
            img = np.zeros((160,160,1), dtype=np.float32)
            x = int(np.clip(80 + self.lateral_offset * 20.0, 0, 159))
            img[:, x, 0] = 1.0
            img[:, 80:81, 0] = np.maximum(img[:, 80:81, 0], 0.3)
            if self.obs_mode == "grayroad":
                return img
            return _rgb_from_gray(img)

    return ToyLaneKeep(seed=seed)

def make_env(which: str, obs_mode: str, action_space: str, steer_bins, seed: int = 0):
    if which.lower() == "carla":
        env, err = try_make_carla_env(obs_mode, action_space, steer_bins, seed=seed)
        if env is not None:
            print("[INFO] Using user's CarEnv.")
            return env
        print(f"[WARN] Could not construct CarEnv ({err}). Falling back to ToyLaneKeep.")
    return make_toy_env(obs_mode, action_space, steer_bins, seed=seed)
