#!/usr/bin/env python3
"""
Replay a trained RL agent across CARLA benchmark routes and log metrics.
Converted from pytest-based tests to a standalone Python script.
"""

import os
import re
import math
import csv
import sys
import argparse
from pathlib import Path
from dataclasses import dataclass
from typing import Iterable, List

import numpy as np
from stable_baselines3 import PPO, DQN

from common_utils import STEER_BINS
from make_env import make_env


# ------------------ CLI ARGUMENTS ------------------

parser = argparse.ArgumentParser(description="Evaluate a trained RL agent on CARLA routes")

parser.add_argument("--model", required=True, help="Path to the trained RL model (.zip)")
parser.add_argument("--algo", default="ppo", choices=["ppo", "dqn"])
parser.add_argument("--obs", default="grayroad", choices=["grayroad", "rgb"])
parser.add_argument("--action", default="disc", choices=["disc", "cont"])
parser.add_argument("--max_steps", type=int, default=3000)
parser.add_argument("--goal_tol", type=float, default=5.0)
parser.add_argument("--seed", type=int, default=7)
parser.add_argument("--host", default="localhost")
parser.add_argument("--port", type=int, default=2000)
parser.add_argument("--render", action="store_true", help="Enable live camera preview during evaluation")
parser.add_argument(
    "--render_freq",
    type=int,
    default=1,
    help="Render every N environment steps (only used when --render is set)",
)
def _lane_limit_arg(value: str):
    if value is None:
        return None
    lowered = value.lower()
    if lowered in {"none", "null", "off", "disable"}:
        return None
    try:
        return float(value)
    except ValueError as exc:
        raise argparse.ArgumentTypeError(f"Invalid lane_dev_limit '{value}': {exc}") from exc

parser.add_argument(
    "--lane_dev_limit",
    type=_lane_limit_arg,
    default=2.0,
    help="Lane deviation termination threshold in meters; use 'none' to disable termination.",
)

args = parser.parse_args()

MODEL_PATH = Path(args.model).expanduser().resolve()
ALGO_NAME = args.algo
OBS_MODE = args.obs
ACTION_SPACE = args.action
MAX_STEPS = args.max_steps
ROUTE_GOAL_TOL = args.goal_tol
BASE_SEED = args.seed
CARLA_HOST = args.host
CARLA_PORT = args.port
RENDER_ENABLED = args.render
RENDER_FREQ = max(1, args.render_freq)
LANE_DEV_LIMIT = args.lane_dev_limit

MODEL_CLS = PPO if ALGO_NAME == "ppo" else DQN


# ------------------ ROUTE SETUP ------------------

# RAW_ROUTE_CODES = [
#     "town06_straight(416_252)", "town06_straight(417_253)", "town06_straight(419_255)",
#     "town06_one_turn(425_318)", "town06_one_turn(426_319)", "town04_straight(325-135)"
# ]
RAW_ROUTE_CODES = [
    # "town01_straight(168-76)",
    # "town01_straight(168-76)",
    # "town01_straight(168-76)",
    # "town01_straight(168-76)",
    # "town01_one_turn(125-163)",
    # "town01_one_turn(125-163)",
    # "town01_one_turn(125-163)",
    # "town01_one_turn(125-163)",
    # "town01_two_turns(206-51)",
    # "town01_two_turns(206-51)",
    # "town01_two_turns(206-51)",
    # "town01_two_turns(206-51)",
    # "town06_straight(416_252)",
    # "town06_straight(417_253)",
    # "town06_straight(419_255)",
    # "town06_straight(420-256)",
    # "town06_one_turn(425_318)",
    # "town06_one_turn(426_319)",
    # "town06_one_turn(428_321)",
    # "town06_one_turn(429_322)",
    # "town06_two_turns(60-386)",
    # "town06_two_turns(61-385)",
    # "town06_two_turns(62-385)",
    # "town06_two_turns(63_386)",
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
    def id(self):
        return f"{self.town}-{self.label}-{self.start}-{self.goal}"


def _parse_routes(codes: Iterable[str]) -> List[RouteSpec]:
    specs = []
    for code in codes:
        m = ROUTE_PATTERN.fullmatch(code.strip())
        if not m:
            raise ValueError(f"Invalid route code: {code}")
        specs.append(
            RouteSpec(
                town=f"Town{int(m.group('town')):02d}",
                label=m.group("label"),
                start=int(m.group("start")),
                goal=int(m.group("goal")),
            )
        )
    return specs


ROUTE_SPECS = _parse_routes(RAW_ROUTE_CODES)


# ------------------ MODEL & ENV ------------------

def _safe_route_filename(route_id: str) -> str:
    safe = re.sub(r"[^A-Za-z0-9_.-]", "_", route_id)
    return f"rl_eval_step_metrics_{safe}.csv"

def load_rl_model():
    if not MODEL_PATH.exists():
        print(f"❌ Model checkpoint not found at {MODEL_PATH}")
        sys.exit(1)
    print(f"Loading model from {MODEL_PATH}")
    return MODEL_CLS.load(str(MODEL_PATH), device="auto")


def _ensure_env(spec: RouteSpec, seed: int):
    try:
        env = make_env(
            which="carla",
            obs_mode=OBS_MODE,
            action_space=ACTION_SPACE,
            steer_bins=STEER_BINS,
            seed=seed,
            show_cam=RENDER_ENABLED,
            routes=[{"start": spec.start, "goal": spec.goal}],
            route_goal_tolerance=ROUTE_GOAL_TOL,
            carla_host=CARLA_HOST,
            carla_port=CARLA_PORT,
            town=spec.town,
            max_steps=MAX_STEPS,
            lane_deviation_terminate=LANE_DEV_LIMIT,
        )
        return env
    except RuntimeError as e:
        print(f"⚠️ Skipping route {spec.id}: {e}")
        return None


# ------------------ EVALUATION ------------------

def evaluate_single_route(model, spec: RouteSpec):
    env = _ensure_env(spec, BASE_SEED)
    if env is None:
        return

    total_reward = 0.0
    success = False
    step_idx = 0
    termination_reason = "max_steps"
    trajectory = []
    step_metrics = []
    last_info = {}

    obs, reset_info = env.reset()
    if reset_info:
        last_info = reset_info
    if RENDER_ENABLED:
        env.render()

    for step_idx in range(MAX_STEPS):
        action, _ = model.predict(obs, deterministic=True)
        obs, reward, terminated, truncated, info = env.step(action)
        info = info or {}
        if RENDER_ENABLED and ((step_idx + 1) % RENDER_FREQ == 0 or terminated or truncated):
            env.render()
        total_reward += reward
        if info:
            last_info = info

        # compute distance and deviation
        if hasattr(env, "_goal_distance"):
            dist = env._goal_distance() or math.inf
        else:
            dist = math.inf

        lane_dev = info.get("lane_deviation")
        yaw_err = info.get("yaw_error")
        trajectory.append({"goal_distance": dist})
        step_metrics.append(
            {
                "route": spec.id,
                "step": step_idx + 1,
                "lane_deviation": lane_dev,
                "goal_distance": dist,
                "yaw_error": yaw_err,
                "vehicle_location_x": info.get("vehicle_location_x"),
                "vehicle_location_y": info.get("vehicle_location_y"),
                "vehicle_location_z": info.get("vehicle_location_z"),
                "vehicle_rotation_pitch": info.get("vehicle_rotation_pitch"),
                "vehicle_rotation_yaw": info.get("vehicle_rotation_yaw"),
                "vehicle_rotation_roll": info.get("vehicle_rotation_roll"),
                "vehicle_speed_kmh": info.get("vehicle_speed_kmh"),
                "waypoint_location_x": info.get("waypoint_location_x"),
                "waypoint_location_y": info.get("waypoint_location_y"),
                "waypoint_location_z": info.get("waypoint_location_z"),
            }
        )

        if dist <= ROUTE_GOAL_TOL:
            success = True
            termination_reason = "goal_reached"
            break

        if terminated or truncated:
            if terminated and step_idx + 1 >= MAX_STEPS and not info.get("terminated_reason"):
                termination_reason = "max_steps"
            else:
                termination_reason = info.get("terminated_reason") or (
                    "truncated" if truncated else "terminated"
                )
            break

    env.close()

    avg_distance = float(np.nanmean([r["goal_distance"] for r in trajectory])) if trajectory else math.inf
    lane_devs = [m["lane_deviation"] for m in step_metrics if m["lane_deviation"] is not None]
    avg_lane_dev = float(np.nanmean(lane_devs)) if lane_devs else math.inf
    result = {
        "route": spec.id,
        "steps": step_idx + 1,
        "success": success,
        "avg_goal_distance": avg_distance,
        "termination_reason": termination_reason,
        "final_lane_deviation": last_info.get("lane_deviation"),
        "final_yaw_error": last_info.get("yaw_error"),
        "avg_lane_deviation": avg_lane_dev,
    }

    with open("rl_eval_results.csv", "a", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=result.keys())
        if f.tell() == 0:
            writer.writeheader()
        writer.writerow(result)

    if step_metrics:
        step_fieldnames = list(step_metrics[0].keys())
        metrics_path = _safe_route_filename(spec.id)
        with open(metrics_path, "a", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=step_fieldnames)
            if f.tell() == 0:
                writer.writeheader()
            writer.writerows(step_metrics)

    print(f"[RESULT] {spec.id}: success={success} | steps={step_idx+1} | reason={termination_reason}")


def evaluate_routes(model):
    for spec in ROUTE_SPECS:
        evaluate_single_route(model, spec)


# ------------------ MAIN ------------------

if __name__ == "__main__":
    model = load_rl_model()
    evaluate_routes(model)
    print("\n✅ All routes evaluated. Results saved to rl_eval_results.csv.")
