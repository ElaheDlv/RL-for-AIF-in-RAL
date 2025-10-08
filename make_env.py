
from typing import Optional, Tuple
import numpy as np
import gymnasium as gym
from gymnasium import spaces

class ChannelFirstObservation(gym.ObservationWrapper):
    def __init__(self, env):
        super().__init__(env)
        obs_space = env.observation_space
        assert isinstance(obs_space, spaces.Box)
        assert len(obs_space.shape) == 3
        self.observation_space = spaces.Box(
            low=np.transpose(obs_space.low, (2, 0, 1)),
            high=np.transpose(obs_space.high, (2, 0, 1)),
            dtype=obs_space.dtype,
        )

    def observation(self, observation):
        return np.transpose(observation, (2, 0, 1))


def ensure_channel_first(env):
    obs_space = env.observation_space
    if isinstance(obs_space, spaces.Box) and len(obs_space.shape) == 3:
        # If channel is last, move to channel-first for SB3 CNN policies
        if obs_space.shape[0] not in (1, 3):
            env = ChannelFirstObservation(env)
    return env


def try_make_carla_env(
    obs_mode: str,
    action_space: str,
    steer_bins,
    seed: int = 0,
    show_cam: bool = False,
    routes=None,
    route_goal_tolerance: float = 5.0,
    carla_host: str = "localhost",
    carla_port: int = 2000,
    town: Optional[str] = None,
):
    """
    Attempt to construct user's CarEnv with a minimal config.
    Expected signature: CarEnv(cfg_dict)
    """
    try:
        from car_env import CarEnv  # user's env
    except Exception as e:
        return None, e
    camera_hw = (160, 160)
    cfg = {
        "image_shape": (1, camera_hw[0], camera_hw[1]) if obs_mode == "grayroad" else (3, camera_hw[0], camera_hw[1]),
        "camera_resolution": camera_hw,
        "obs_mode": obs_mode,             # expected to switch camera modality
        "discrete_actions": (action_space == "disc"),
        "steer_bins": steer_bins.tolist(),
        "max_steps": 600,
        "seed": seed,
        "show_cam": show_cam,
        "routes": routes,
        "route_goal_tolerance": route_goal_tolerance,
        "carla_host": carla_host,
        "carla_port": int(carla_port),
    }
    if town:
        cfg["town"] = str(town)
    if action_space == "cont":
        cfg["action_space"] = "continuous"
    try:
        env = CarEnv(cfg)
        env = ensure_channel_first(env)
        return env, None
    except Exception as e:
        return None, e

# ---------------- Toy envs -----------------

def _rgb_from_gray(gray):
    rgb = np.repeat(gray, 3, axis=2)
    return rgb

def make_toy_env(
    obs_mode: str,
    action_space: str,
    steer_bins,
    seed: int = 0,
    show_cam: bool = False,
    routes=None,
    route_goal_tolerance: float = 5.0,
):
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
            self.show_cam = show_cam
            self.window_name = f"ToyLaneKeep-{obs_mode}-{action_space}"
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

        def render(self):
            if not self.show_cam:
                return
            frame = self._obs()
            frame = np.clip(frame * 255.0, 0, 255).astype(np.uint8)
            if frame.shape[-1] == 1:
                frame = np.repeat(frame, 3, axis=2)
            import cv2
            cv2.imshow(self.window_name, frame)
            cv2.waitKey(1)

        def close(self):
            if self.show_cam:
                import cv2
                cv2.destroyWindow(self.window_name)

    env = ToyLaneKeep(seed=seed)
    return ensure_channel_first(env)

def make_env(
    which: str,
    obs_mode: str,
    action_space: str,
    steer_bins,
    seed: int = 0,
    show_cam: bool = False,
    routes=None,
    route_goal_tolerance: float = 5.0,
    carla_host: str = "localhost",
    carla_port: int = 2000,
    town: Optional[str] = None,
):
    which = which.lower()
    if which == "carla":
        env, err = try_make_carla_env(
            obs_mode,
            action_space,
            steer_bins,
            seed=seed,
            show_cam=show_cam,
            routes=routes,
            route_goal_tolerance=route_goal_tolerance,
            carla_host=carla_host,
            carla_port=carla_port,
            town=town,
        )
        if env is None:
            raise RuntimeError(
                "Failed to construct CarEnv; please ensure CARLA is running and matches the expected API."
                f" Details: {err}"
            )
        print("[INFO] Using user's CarEnv.")
        return env
    if which == "toy":
        return make_toy_env(
            obs_mode,
            action_space,
            steer_bins,
            seed=seed,
            show_cam=show_cam,
            routes=routes,
            route_goal_tolerance=route_goal_tolerance,
        )
    raise ValueError(f"Unknown environment kind '{which}'.")
