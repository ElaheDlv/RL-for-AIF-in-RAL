from stable_baselines3.common.callbacks import BaseCallback
import numpy as np


class DrivingMetricsCallback(BaseCallback):
    """Logs extra driving metrics to TensorBoard for CARLA lane-keeping RL."""

    def __init__(self, verbose: int = 0):
        super().__init__(verbose)
        self.episode_successes = []
        self.episode_deviations = []

    def _on_step(self) -> bool:
        infos = self.locals.get("infos", [])
        rewards = self.locals.get("rewards", [])
        dones = self.locals.get("dones", [])

        if rewards:
            self.logger.record("custom/step_reward", float(np.mean(rewards)))

        if infos:
            info = infos[0]
            if "lane_deviation" in info:
                self.logger.record("custom/lane_deviation_step", float(info["lane_deviation"]))
            if "success" in info:
                self.logger.record("custom/success_step", float(info["success"]))

        # Collect per-episode metrics when environments terminate.
        for done, info in zip(dones, infos):
            if done:
                episode_data = info.get("episode")
                if episode_data:
                    self.logger.record("custom/episode_reward", float(episode_data.get("r", 0.0)))
                    self.logger.record("custom/episode_length", float(episode_data.get("l", 0.0)))
                if "success" in info:
                    self.episode_successes.append(float(info["success"]))
                if "lane_deviation" in info:
                    self.episode_deviations.append(float(info["lane_deviation"]))

        return True

    def _on_rollout_end(self) -> None:
        ep_info_buffer = getattr(self.model, "ep_info_buffer", None)
        if ep_info_buffer and len(ep_info_buffer) > 0:
            mean_r = np.mean([ep.get("r", 0.0) for ep in ep_info_buffer])
            mean_len = np.mean([ep.get("l", 0.0) for ep in ep_info_buffer])
            self.logger.record("custom/episode_reward_mean", float(mean_r))
            self.logger.record("custom/episode_length_mean", float(mean_len))

        if self.episode_successes:
            self.logger.record("custom/episode_success_rate", float(np.mean(self.episode_successes)))
            self.episode_successes.clear()
        if self.episode_deviations:
            self.logger.record("custom/episode_avg_deviation", float(np.mean(self.episode_deviations)))
            self.episode_deviations.clear()
