#!/usr/bin/env python3
"""Route-based evaluation script for trained RL agents in CARLA."""

import argparse
import os
from typing import List, Dict

import numpy as np
from stable_baselines3 import PPO, DQN

from common_utils import STEER_BINS
from make_env import make_env
from route_utils import combine_route_sources


def main():
    parser = argparse.ArgumentParser(description="Evaluate trained RL policies on preset CARLA routes")
    parser.add_argument("--algo", choices=["ppo", "dqn"], required=True, help="Algorithm of the trained model")
    parser.add_argument("--model", required=True, help="Path to the trained SB3 model file")
    parser.add_argument("--obs", choices=["rgb", "grayroad"], default="rgb")
    parser.add_argument("--action", choices=["disc", "cont"], default="disc", help="Action space (PPO only)")
    parser.add_argument("--routes", nargs="*", help="Route list as start:goal entries (e.g. 416:252 120:78)")
    parser.add_argument("--routes-json", help="Optional JSON file with a 'routes' list containing start/goal indices")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--goal-tol", type=float, default=5.0, help="Goal distance tolerance in meters")
    parser.add_argument("--render", action="store_true", help="Enable live camera preview during rollout")
    parser.add_argument("--render-freq", type=int, default=1, help="Render every N environment steps")
    parser.add_argument("--max-steps", type=int, default=3000, help="Safety cap on steps per route")
    parser.add_argument("--out", default=None, help="Optional CSV file to append route metrics")
    parser.add_argument("--log-dir", default=None, help="Directory to store per-route trajectory CSVs")
    args = parser.parse_args()

    if args.algo == "dqn" and args.action != "disc":
        parser.error("DQN supports only discrete actions")

    try:
        routes = combine_route_sources(args.routes, args.routes_json)
    except ValueError as exc:
        parser.error(str(exc))

    env = make_env(
        which="carla",
        obs_mode=args.obs,
        action_space=args.action,
        steer_bins=STEER_BINS,
        seed=args.seed,
        show_cam=args.render,
        routes=routes,
        route_goal_tolerance=args.goal_tol,
    )

    base_env = getattr(env, "unwrapped", env)

    algo_cls = {"ppo": PPO, "dqn": DQN}[args.algo]
    model = algo_cls.load(args.model, device="auto")
    print(f"[INFO] Loaded {args.algo.upper()} model from {args.model}")

    traj_dir = None
    if args.log_dir:
        traj_dir = os.path.abspath(args.log_dir)
        os.makedirs(traj_dir, exist_ok=True)
        print(f"[INFO] Trajectory CSVs will be written to {traj_dir}")

    world = getattr(base_env, "world", None)
    vehicle = getattr(base_env, "vehicle", None)
    world_map = getattr(base_env, "map", None)
    delta_seconds = None
    if world is not None:
        delta_seconds = getattr(world.get_settings(), "fixed_delta_seconds", None)
    if not delta_seconds:
        delta_seconds = 0.05
    stationary_limit = int(300.0 / delta_seconds)
    moving_away_limit = int(5.0 / delta_seconds)

    metrics: List[Dict[str, float]] = []
    for route_idx in range(len(routes)):
        obs, info = env.reset()
        current_route = info.get("route", routes[route_idx])
        done = False
        total_reward = 0.0
        step_count = 0
        last_info: Dict[str, float] = {}
        trajectory: List[Dict[str, float]] = []
        stationary_steps = 0
        moving_away_steps = 0
        best_goal_distance = float("inf")
        termination_reason = "max_steps"

        while not done and step_count < args.max_steps:
            if args.render and (step_count % max(1, args.render_freq) == 0):
                env.render()
            action, _ = model.predict(obs, deterministic=True)
            obs, reward, terminated, truncated, last_info = env.step(
                action if args.action == "disc" else np.array(action)
            )
            total_reward += float(reward)
            step_count += 1
            lane_dev = float(last_info.get("lane_deviation", np.nan))
            goal_distance = float(last_info.get("goal_distance", np.nan))

            transform = None
            speed_kmh = float("nan")
            lane_width = float("nan")
            if vehicle is not None:
                transform = vehicle.get_transform()
                vel = vehicle.get_velocity()
                speed_kmh = 3.6 * (vel.x ** 2 + vel.y ** 2 + vel.z ** 2) ** 0.5
                if world_map is not None:
                    waypoint = world_map.get_waypoint(transform.location, project_to_road=True)
                    lane_width = waypoint.lane_width if waypoint else float("nan")
            trajectory.append(
                {
                    "step": step_count,
                    "x": transform.location.x if transform else float("nan"),
                    "y": transform.location.y if transform else float("nan"),
                    "z": transform.location.z if transform else float("nan"),
                    "pitch": transform.rotation.pitch if transform else float("nan"),
                    "yaw": transform.rotation.yaw if transform else float("nan"),
                    "roll": transform.rotation.roll if transform else float("nan"),
                    "speed_kmh": speed_kmh,
                    "goal_distance": goal_distance,
                    "lane_deviation": lane_dev,
                    "lane_width": lane_width,
                }
            )

            if speed_kmh < 1.0:
                stationary_steps += 1
                if stationary_steps >= stationary_limit:
                    termination_reason = "stationary_5min"
                    terminated = True
            else:
                stationary_steps = 0

            if not np.isnan(goal_distance):
                if goal_distance + 0.1 < best_goal_distance:
                    best_goal_distance = goal_distance
                    moving_away_steps = 0
                elif goal_distance > best_goal_distance + 5.0:
                    moving_away_steps += 1
                    if moving_away_steps >= moving_away_limit:
                        termination_reason = "moving_away"
                        terminated = True
                else:
                    moving_away_steps = max(0, moving_away_steps - 1)

            done = terminated or truncated

        success = bool(last_info.get("success", False))
        goal_distance = float(last_info.get("goal_distance", np.nan))
        avg_lane_dev = float(
            np.nanmean([row["lane_deviation"] for row in trajectory])
        ) if trajectory else float("nan")
        if success:
            termination_reason = "goal_reached"
        print(
            f"Route {route_idx+1}/{len(routes)} | start={current_route['start']} -> goal={current_route['goal']} | "
            f"steps={step_count} | reward={total_reward:.2f} | success={success} | goal_dist={goal_distance:.2f} m | "
            f"reason={termination_reason}"
        )

        metrics.append(
            {
                "route_index": route_idx,
                "start": current_route["start"],
                "goal": current_route["goal"],
                "steps": step_count,
                "reward": total_reward,
                "success": float(success),
                "goal_distance": goal_distance,
                "lane_deviation": avg_lane_dev,
                "termination_reason": termination_reason,
            }
        )

        if traj_dir and trajectory:
            import csv

            traj_path = os.path.join(traj_dir, f"route_{route_idx:03d}.csv")
            with open(traj_path, "w", newline="") as traj_file:
                writer = csv.DictWriter(
                    traj_file,
                    fieldnames=[
                        "step",
                        "x",
                        "y",
                        "z",
                        "pitch",
                        "yaw",
                        "roll",
                        "speed_kmh",
                        "goal_distance",
                        "lane_deviation",
                        "lane_width",
                    ],
                )
                writer.writeheader()
                for row in trajectory:
                    writer.writerow(row)
            print(f"[INFO] Saved trajectory to {traj_path}")

    env.close()

    if args.out:
        import csv

        out_exists = os.path.exists(args.out)
        with open(args.out, "a", newline="") as csv_file:
            writer = csv.DictWriter(
                csv_file,
                fieldnames=[
                    "route_index",
                    "start",
                    "goal",
                    "steps",
                    "reward",
                    "success",
                    "goal_distance",
                    "lane_deviation",
                    "termination_reason",
                ],
            )
            if not out_exists:
                writer.writeheader()
            for row in metrics:
                writer.writerow(row)
        print(f"[INFO] Appended metrics to {args.out}")


if __name__ == "__main__":
    main()
