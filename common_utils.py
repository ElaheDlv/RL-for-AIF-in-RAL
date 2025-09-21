
import time
import numpy as np
from typing import Dict, Any, Tuple, Optional

from stable_baselines3.common.callbacks import BaseCallback

def make_steering_bins(n_bins: int = 20):
    import numpy as np
    return np.linspace(-1.0, 1.0, n_bins, dtype=np.float32)

STEER_BINS = make_steering_bins(20)

def choose_policy_for_obs_space(obs_space) -> str:
    try:
        from gymnasium.spaces import Dict as DictSpace, Box
        if isinstance(obs_space, DictSpace):
            return "MultiInputPolicy"
        if isinstance(obs_space, Box) and len(obs_space.shape) == 3:
            return "CnnPolicy"
    except Exception:
        pass
    return "MlpPolicy"

def read_metrics_from_info(info: Dict[str, Any]) -> Tuple[Any, Any]:
    success = info.get("success", None)
    dev = info.get("lane_deviation", None)
    return success, dev

class StepTimer:
    def __init__(self):
        self._t0 = None
        self.samples = []

    def start(self):
        self._t0 = time.perf_counter()

    def stop(self):
        if self._t0 is None:
            return
        dt = time.perf_counter() - self._t0
        self.samples.append(dt)
        self._t0 = None

    @property
    def mean_latency_ms(self):
        import numpy as np
        if not self.samples:
            return float('nan')
        return float(np.mean(self.samples) * 1000.0)

    @property
    def mean_fps(self):
        import numpy as np
        if not self.samples:
            return float('nan')
        mean_dt = float(np.mean(self.samples))
        return 1.0 / mean_dt if mean_dt > 0 else float('inf')


class LiveRenderCallback(BaseCallback):
    """Render the first environment at a configurable frequency using matplotlib."""

    def __init__(self, vec_env, freq: int = 1, verbose: int = 0):
        super().__init__(verbose)
        self.vec_env = vec_env
        self.freq = max(1, int(freq))

    def _on_step(self) -> bool:
        if self.num_timesteps % self.freq == 0:
            envs = getattr(self.vec_env, "envs", None)
            if envs:
                base_env = envs[0]
                if hasattr(base_env, "render"):
                    base_env.render()
        return True
