import math
from pathlib import Path

import pytest

pytest.importorskip("torch", reason="Torch is required for PPO training pipeline test")
pytest.importorskip("stable_baselines3", reason="Stable-Baselines3 is required for PPO training pipeline test")

from train_ppo import train_and_eval


@pytest.mark.slow
def test_toy_ppo_training_pipeline(tmp_path):
    """Train PPO on the lightweight toy env and ensure artefacts are produced."""
    out_dir = tmp_path / "ppo_output"
    out_dir.mkdir()

    metrics = train_and_eval(
        env_kind="toy",
        obs_mode="grayroad",
        action_space="disc",
        timesteps=1024,
        eval_episodes=1,
        seed=7,
        out_dir=str(out_dir),
    )

    # The training helper should always return these keys.
    expected_keys = {
        "success_rate",
        "avg_deviation_m",
        "fps",
        "latency_ms",
        "timesteps",
        "log_tag",
    }
    assert expected_keys.issubset(metrics.keys())

    assert 0.0 <= metrics["success_rate"] <= 100.0
    assert math.isfinite(metrics["avg_deviation_m"])
    assert metrics["timesteps"] >= 1024

    model_path = Path(out_dir) / f"ppo_{metrics['log_tag']}.zip"
    assert model_path.exists()

    csv_path = Path(out_dir) / "results_ppo.csv"
    assert csv_path.exists()
