"""End-to-end regression tests for the trained RL agent across canonical CARLA routes.

This suite targets the PPO/DQN agents saved via Stable-Baselines3. It replays the
policy over a curated list of start/goal pairs drawn from Town04 and Town06 route
benchmarks. The loop mirrors the evaluation settings used by the imitation and
Active Inference scripts (stationary timeout, moving-away guard, 5 m goal
tolerance) so the resulting metrics are directly comparable across agents. The
model path and other environment details are provided via environment variables
so the test can be opt-in (CARLA must be running).

Required environment variable:
    RL_ROUTE_MODEL  → path to the Stable-Baselines3 ``.zip`` checkpoint to load.

Optional overrides:
    RL_ROUTE_ALGO        → ``ppo`` (default) or ``dqn``
    RL_ROUTE_OBS         → observation mode (``rgb`` or ``grayroad``; default ``grayroad``)
    RL_ROUTE_ACTION      → action space (``disc`` or ``cont``; default ``disc``)
    RL_ROUTE_MAX_STEPS   → max environment steps per route (default: 3000)
    RL_ROUTE_GOAL_TOL    → success distance threshold in metres (default: 5.0)
    RL_ROUTE_SEED        → base RNG seed (default: 7)
    RL_ROUTE_HOST / PORT → CARLA server address (default: localhost / 2000)

Skip behaviour:
    - The test module is skipped entirely if Stable-Baselines3 is unavailable or
      if ``RL_ROUTE_MODEL`` is not defined.
    - Each parameterised route will be skipped gracefully if CARLA is not reachable.
"""

from __future__ import annotations

import math
import os
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, List

import numpy as np
import pytest

import csv

import argparse

pytest.importorskip(
    "stable_baselines3",
    reason="Stable-Baselines3 is required to replay trained RL policies",
)

from stable_baselines3 import PPO, DQN

from common_utils import STEER_BINS
from make_env import make_env


MODEL_PATH_ENV = "RL_ROUTE_MODEL"
MODEL_PATH_VALUE = os.environ.get(MODEL_PATH_ENV)

if not MODEL_PATH_VALUE:
    pytest.skip(
        f"Set {MODEL_PATH_ENV} to the SB3 checkpoint you wish to evaluate",
        allow_module_level=True,
    )

parser = argparse.ArgumentParser(description="Replay a trained RL agent on predefined CARLA routes")

parser.add_argument("--model", required=True, help="Path to the trained RL model (.zip)")
parser.add_argument("--algo", default="ppo", choices=["ppo", "dqn"], help="RL algorithm (ppo or dqn)")
parser.add_argument("--obs", default="grayroad", choices=["grayroad", "rgb"], help="Observation mode")
parser.add_argument("--action", default="disc", choices=["disc", "cont"], help="Action space")
parser.add_argument("--max_steps", type=int, default=3000, help="Max steps per route")
parser.add_argument("--goal_tol", type=float, default=5.0, help="Goal distance tolerance (m)")
parser.add_argument("--seed", type=int, default=7, help="Base RNG seed")
parser.add_argument("--host", default="localhost", help="CARLA host address")
parser.add_argument("--port", type=int, default=2000, help="CARLA port")

args, unknown = parser.parse_known_args()

MODEL_PATH = Path(args.model).expanduser().resolve()
ALGO_NAME = args.algo.lower()
OBS_MODE = args.obs.lower()
ACTION_SPACE = args.action.lower()
MAX_STEPS = args.max_steps
ROUTE_GOAL_TOL = args.goal_tol
BASE_SEED = args.seed
CARLA_HOST = args.host
CARLA_PORT = args.port
# ALGO_NAME = os.environ.get("RL_ROUTE_ALGO", "ppo").strip().lower()
# OBS_MODE = os.environ.get("RL_ROUTE_OBS", "grayroad").strip().lower()
# ACTION_SPACE = os.environ.get("RL_ROUTE_ACTION", "disc").strip().lower()
# MAX_STEPS = int(os.environ.get("RL_ROUTE_MAX_STEPS", "3000"))
# ROUTE_GOAL_TOL = float(os.environ.get("RL_ROUTE_GOAL_TOL", "5.0"))
# BASE_SEED = int(os.environ.get("RL_ROUTE_SEED", "7"))
# CARLA_HOST = os.environ.get("RL_ROUTE_HOST", "localhost")
# CARLA_PORT = int(os.environ.get("RL_ROUTE_PORT", "2000"))

if ALGO_NAME not in {"ppo", "dqn"}:
    raise ValueError(f"Unsupported RL_ROUTE_ALGO '{ALGO_NAME}'. Expected 'ppo' or 'dqn'.")
if OBS_MODE not in {"rgb", "grayroad"}:
    raise ValueError(f"Unsupported RL_ROUTE_OBS '{OBS_MODE}'.")
if ACTION_SPACE not in {"disc", "cont"}:
    raise ValueError(f"Unsupported RL_ROUTE_ACTION '{ACTION_SPACE}'.")
if ALGO_NAME == "dqn" and ACTION_SPACE != "disc":
    raise ValueError("DQN agents currently support only discrete steering actions.")


RAW_ROUTE_CODES = [
    "town06_straight(416_252)",
    "town06_straight(417_253)",
    "town06_straight(419_255)",
    "town06_straight(420-256)",
    "town06_one_turn(425_318)",
    "town06_one_turn(426_319)",
    "town06_one_turn(428_321)",
    "town06_one_turn(429_322)",
    "town06_two_turns(60-386)",
    "town06_two_turns(61-385)",
    "town06_two_turns(62-385)",
    "town06_two_turns(63_386)",
    "town04_one_turn(212-352)",
    "town04_one_turn(213-351)",
    "town04_one_turn(214-350)",
    "town04_one_turn(215_349)",
    "town04_straight(325-135)",
    "town04_straight(326-136)",
    "town04_straight(327-137)",
    "town04_straight(328-138)",
    "town04_two_turn(37-207)",
    "town04_two_turn(38-206)",
    "town04_two_turn(39-205)",
    "town04_two_turn(40-204)",
]


ROUTE_PATTERN = re.compile(
    r"town(?P<town>\d{2})_(?P<label>[a-z0-9_]+)\((?P<start>\d+)[_-](?P<goal>\d+)\)",
    re.IGNORECASE,
)


@dataclass(frozen=True)
class RouteSpec:
    town: str
    label: str
    start: int
    goal: int

    @property
    def id(self) -> str:
        return f"{self.town}-{self.label}-{self.start}-{self.goal}"


def _parse_routes(codes: Iterable[str]) -> List[RouteSpec]:
    specs: List[RouteSpec] = []
    for code in codes:
        match = ROUTE_PATTERN.fullmatch(code.strip())
        if not match:
            raise ValueError(f"Route code '{code}' does not match expected pattern")
        town_code = match.group("town")
        label = match.group("label")
        start_idx = int(match.group("start"))
        goal_idx = int(match.group("goal"))
        specs.append(
            RouteSpec(
                town=f"Town{int(town_code):02d}",
                label=label,
                start=start_idx,
                goal=goal_idx,
            )
        )
    return specs


ROUTE_SPECS = _parse_routes(RAW_ROUTE_CODES)


def _route_params():
    for spec in ROUTE_SPECS:
        yield pytest.param(spec, id=spec.id)


MODEL_PATH = Path(MODEL_PATH_VALUE).expanduser().resolve()
MODEL_CLS = PPO if ALGO_NAME == "ppo" else DQN


@pytest.fixture(scope="session")
def rl_model():
    if not MODEL_PATH.exists():
        pytest.skip(f"Model checkpoint not found at {MODEL_PATH}")
    return MODEL_CLS.load(str(MODEL_PATH), device="auto")


def _ensure_env(spec: RouteSpec, seed: int):
    try:
        env = make_env(
            which="carla",
            obs_mode=OBS_MODE,
            action_space=ACTION_SPACE,
            steer_bins=STEER_BINS,
            seed=seed,
            show_cam=False,
            routes=[{"start": spec.start, "goal": spec.goal}],
            route_goal_tolerance=ROUTE_GOAL_TOL,
            carla_host=CARLA_HOST,
            carla_port=CARLA_PORT,
            town=spec.town,
        )
        return env
    except RuntimeError as exc:
        pytest.skip(f"CARLA environment unavailable for {spec.id}: {exc}")


def _compute_goal_distance(env_obj) -> float:
    if hasattr(env_obj, "_goal_distance"):
        dist = env_obj._goal_distance()
        if dist is None:
            return math.inf
        return float(dist)
    return math.inf


def _compute_lane_deviation(env_obj) -> float:
    if hasattr(env_obj, "_lane_deviation"):
        return float(env_obj._lane_deviation())
    return float("nan")


def _compute_speed_kmh(env_obj) -> float:
    vehicle = getattr(env_obj, "vehicle", None)
    if vehicle is None:
        return float("nan")
    vel = vehicle.get_velocity()
    return 3.6 * (vel.x ** 2 + vel.y ** 2 + vel.z ** 2) ** 0.5


@pytest.mark.slow
@pytest.mark.parametrize("spec", list(_route_params()))
def test_trained_rl_agent_handles_route(spec: RouteSpec, rl_model):
    env = _ensure_env(spec, seed=BASE_SEED)
    base_env = getattr(env, "unwrapped", env)
    route_idx = ROUTE_SPECS.index(spec)

    world = getattr(base_env, "world", None)
    delta_seconds = None
    if world is not None:
        delta_seconds = getattr(world.get_settings(), "fixed_delta_seconds", None)
    if not delta_seconds:
        delta_seconds = 0.05
    stationary_limit = int(300.0 / delta_seconds)
    moving_away_limit = int(5.0 / delta_seconds)

    total_reward = 0.0
    trajectory: List[dict] = []
    stationary_steps = 0
    moving_away_steps = 0
    best_goal_distance = math.inf
    termination_reason = "max_steps"
    success = False
    step_idx = -1
    final_goal_distance = math.inf

    try:
        obs, _ = env.reset()

        for step_idx in range(MAX_STEPS):
            action, _ = rl_model.predict(obs, deterministic=True)
            if ACTION_SPACE == "disc":
                step_action = action
            else:
                step_action = np.array(action, dtype=np.float32)
            obs, reward, terminated, truncated, info = env.step(step_action)
            total_reward += float(reward)

            lane_dev = _compute_lane_deviation(base_env)
            goal_distance = _compute_goal_distance(base_env)
            final_goal_distance = goal_distance
            trajectory.append({
                "lane_deviation": lane_dev,
                "goal_distance": goal_distance,
            })

            speed_kmh = _compute_speed_kmh(base_env)

            if speed_kmh < 1.0:
                stationary_steps += 1
                if stationary_steps >= stationary_limit:
                    termination_reason = "stationary_5min"
                    break
            else:
                stationary_steps = 0

            if math.isfinite(goal_distance):
                if goal_distance + 0.1 < best_goal_distance:
                    best_goal_distance = goal_distance
                    moving_away_steps = 0
                elif goal_distance > best_goal_distance + 5.0:
                    moving_away_steps += 1
                    if moving_away_steps >= moving_away_limit:
                        termination_reason = "moving_away"
                        break
                else:
                    moving_away_steps = max(0, moving_away_steps - 1)

            if goal_distance <= ROUTE_GOAL_TOL:
                success = True
                termination_reason = "goal_reached"
                break

            if terminated or truncated:
                break
        else:
            step_idx = MAX_STEPS - 1
    finally:
        env.close()

    steps_taken = step_idx + 1
    if termination_reason == "max_steps" and success:
        termination_reason = "goal_reached"

    avg_lane_dev = float(
        np.nanmean([row["lane_deviation"] for row in trajectory])
    ) if trajectory else float("nan")
    final_goal_distance = trajectory[-1]["goal_distance"] if trajectory else math.inf

    metrics_row = {
        "route_index": route_idx,
        "start": spec.start,
        "goal": spec.goal,
        "steps": steps_taken,
        "success": float(success),
        "goal_distance": final_goal_distance,
        "lane_deviation": avg_lane_dev,
        "termination_reason": termination_reason,
    }
    
    results_path = "rl_eval_results.csv"
    header = [
    "route_index", "start", "goal", "steps", "success",
    "goal_distance", "lane_deviation", "termination_reason"
        ]

    # Append results to a CSV file
    file_exists = os.path.exists(results_path)
    with open(results_path, "a", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=header)
        if not file_exists:
            writer.writeheader()
        writer.writerow(metrics_row)

    print(
        f"[RESULT] {spec.id}: steps={metrics_row['steps']} | success={bool(metrics_row['success'])} | "
        f"goal_dist={metrics_row['goal_distance']:.2f} | lane_dev={metrics_row['lane_deviation']:.3f} | "
        f"reason={metrics_row['termination_reason']}"
    )

    assert success, (
        f"RL agent failed route {spec.id}: termination_reason={termination_reason}, "
        f"goal_dist={final_goal_distance:.2f}, steps={steps_taken}, total_reward={total_reward:.2f}"
    )
