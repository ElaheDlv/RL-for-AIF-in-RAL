#!/usr/bin/env python3
"""Route-based evaluation script for the imitation learning policy."""

import argparse
import csv
import math
import time
from pathlib import Path
from typing import Dict, List

import carla
import cv2
import numpy as np
import tensorflow as tf

from route_utils import combine_route_sources, validate_route_indices


def kmh(velocity: carla.Vector3D) -> float:
    return 3.6 * math.sqrt(velocity.x ** 2 + velocity.y ** 2 + velocity.z ** 2)


def preprocess_rgb(image: np.ndarray) -> np.ndarray:
    rgb = image[:, :, :3]
    crop = rgb[350:719, 0:1279]
    resized = cv2.resize(crop, (160, 160))
    norm = resized.astype("float32") / 255.0
    norm = np.expand_dims(norm, axis=0)
    return norm


def ensure_image(sensor_dict: Dict[str, np.ndarray], key: str, timeout: float = 5.0):
    start = time.time()
    while sensor_dict.get(key) is None:
        time.sleep(0.01)
        if time.time() - start > timeout:
            raise RuntimeError(f"Timed out waiting for sensor '{key}'")


def main():
    parser = argparse.ArgumentParser(description="Evaluate imitation learning model on preset routes")
    parser.add_argument("--model", default="Imitation_learning_trained_model.h5", help="Keras model path")
    parser.add_argument("--routes", nargs="*", help="Route list as start:goal entries")
    parser.add_argument("--routes-json", help="JSON file holding routes")
    parser.add_argument("--host", default="localhost")
    parser.add_argument("--port", type=int, default=2000)
    parser.add_argument("--town", default=None)
    parser.add_argument("--goal-tol", type=float, default=5.0)
    parser.add_argument("--target-speed", type=float, default=30.0)
    parser.add_argument("--throttle", type=float, default=0.6)
    parser.add_argument("--max-steps", type=int, default=3000)
    parser.add_argument("--csv", default=None, help="Optional CSV to append metrics")
    args = parser.parse_args()

    model = tf.keras.models.load_model(args.model)
    print(f"[INFO] Loaded imitation model from {args.model}")

    routes = combine_route_sources(args.routes, args.routes_json)

    client = carla.Client(args.host, args.port)
    client.set_timeout(10.0)
    world = client.load_world(args.town) if args.town else client.get_world()

    settings = world.get_settings()
    settings.fixed_delta_seconds = 0.01
    world.apply_settings(settings)

    world_map = world.get_map()
    spawn_points = world_map.get_spawn_points()
    validate_route_indices(routes, len(spawn_points))
    bp_lib = world.get_blueprint_library()
    vehicle_bp = bp_lib.find('vehicle.lincoln.mkz_2020')

    vehicle = world.try_spawn_actor(vehicle_bp, spawn_points[routes[0]['start']])
    if vehicle is None:
        raise RuntimeError(f"Could not spawn vehicle at index {routes[0]['start']}")

    bound_x = 0.5 + vehicle.bounding_box.extent.x
    bound_y = 0.5 + vehicle.bounding_box.extent.y
    bound_z = 0.5 + vehicle.bounding_box.extent.z
    cam_trans = carla.Transform(carla.Location(x=+0.8 * bound_x, y=0.0 * bound_y, z=1.3 * bound_z))

    rgb_bp = bp_lib.find('sensor.camera.rgb')
    rgb_bp.set_attribute('image_size_x', '1280')
    rgb_bp.set_attribute('image_size_y', '720')
    rgb_bp.set_attribute('fov', '110')

    sem_bp = bp_lib.find('sensor.camera.semantic_segmentation')
    sem_bp.set_attribute('image_size_x', '1280')
    sem_bp.set_attribute('image_size_y', '720')
    sem_bp.set_attribute('fov', '110')

    sensor_data: Dict[str, np.ndarray] = {"rgb": None, "sem": None}

    def rgb_callback(image: carla.SensorData):
        array = np.frombuffer(image.raw_data, dtype=np.uint8)
        sensor_data['rgb'] = array.reshape((image.height, image.width, 4))

    def sem_callback(image: carla.SensorData):
        image.convert(carla.ColorConverter.CityScapesPalette)
        array = np.frombuffer(image.raw_data, dtype=np.uint8)
        sensor_data['sem'] = array.reshape((image.height, image.width, 4))

    rgb_cam = world.spawn_actor(rgb_bp, cam_trans, attach_to=vehicle)
    sem_cam = world.spawn_actor(sem_bp, cam_trans, attach_to=vehicle)
    rgb_cam.listen(rgb_callback)
    sem_cam.listen(sem_callback)

    window_name = 'Imitation-Preview'
    cv2.namedWindow(window_name, cv2.WINDOW_AUTOSIZE)

    metrics: List[Dict[str, float]] = []

    try:
        ensure_image(sensor_data, 'rgb')
        ensure_image(sensor_data, 'sem')

        delta_seconds = getattr(world.get_settings(), "fixed_delta_seconds", 0.01)
        stationary_limit = int(300.0 / delta_seconds)
        moving_away_limit = int(5.0 / delta_seconds)

        for idx, route in enumerate(routes):
            print(f"[INFO] Starting route {idx+1}/{len(routes)}: {route['start']} -> {route['goal']}")
            start_transform = spawn_points[route['start']]
            goal_transform = spawn_points[route['goal']]
            vehicle.set_transform(start_transform)
            if hasattr(vehicle, "set_target_velocity"):
                vehicle.set_target_velocity(carla.Vector3D(0, 0, 0))
            if hasattr(vehicle, "set_target_angular_velocity"):
                vehicle.set_target_angular_velocity(carla.Vector3D(0, 0, 0))
            world.tick()

            step = 0
            success = False
            stationary_steps = 0
            moving_away_steps = 0
            best_goal_distance = float('inf')
            termination_reason = "max_steps"
            route_log: List[Dict[str, float]] = []

            while step < args.max_steps:
                world.tick()
                rgb_image = sensor_data['rgb']
                sem_image = sensor_data['sem']
                if rgb_image is None or sem_image is None:
                    continue

                top_row = np.concatenate((rgb_image, sem_image), axis=1)
                cv2.putText(
                    top_row,
                    f"route {idx+1} | step {step}",
                    (30, 40),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    1.0,
                    (255, 255, 255),
                    2,
                )
                cv2.imshow(window_name, top_row)
                if cv2.waitKey(1) & 0xFF == ord('q'):
                    raise KeyboardInterrupt

                input_tensor = preprocess_rgb(rgb_image)
                output = model.predict(input_tensor, verbose=0)
                steering = float(output.squeeze())

                speed = kmh(vehicle.get_velocity())
                throttle = args.throttle if speed < args.target_speed else 0.0
                vehicle.apply_control(carla.VehicleControl(throttle=throttle, steer=steering))

                transform = vehicle.get_transform()
                goal_distance = transform.location.distance(goal_transform.location)
                waypoint = world_map.get_waypoint(transform.location, project_to_road=True)
                lane_width = waypoint.lane_width if waypoint else float('nan')
                lane_deviation = transform.location.distance(waypoint.transform.location) if waypoint else float('nan')
                route_log.append(
                    {
                        "step": step,
                        "x": transform.location.x,
                        "y": transform.location.y,
                        "z": transform.location.z,
                        "pitch": transform.rotation.pitch,
                        "yaw": transform.rotation.yaw,
                        "roll": transform.rotation.roll,
                        "speed_kmh": speed,
                        "goal_distance": goal_distance,
                        "lane_deviation": lane_deviation,
                        "lane_width": lane_width,
                        "steer": steering,
                        "throttle": throttle,
                    }
                )

                if speed < 1.0:
                    stationary_steps += 1
                    if stationary_steps >= stationary_limit:
                        termination_reason = "stationary_5min"
                        break
                else:
                    stationary_steps = 0

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

                if goal_distance <= args.goal_tol:
                    success = True
                    termination_reason = "goal_reached"
                    break

                step += 1

            metrics.append(
                {
                    "route_index": idx,
                    "start": route['start'],
                    "goal": route['goal'],
                    "steps": step,
                    "success": float(success),
                    "termination_reason": termination_reason,
                }
            )

            print(f"[INFO] Route {idx+1}: success={success} | steps={step} | reason={termination_reason}")

            if traj_dir and route_log:
                import csv

                traj_path = traj_dir / f"route_{idx:03d}.csv"
                with traj_path.open("w", newline="") as f_traj:
                    writer = csv.DictWriter(
                        f_traj,
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
                            "steer",
                            "throttle",
                        ],
                    )
                    writer.writeheader()
                    for row in route_log:
                        writer.writerow(row)
                print(f"[INFO] Saved trajectory to {traj_path}")

        if args.csv:
            write_header = not Path(args.csv).exists()
            with open(args.csv, 'a', newline='') as f:
                writer = csv.DictWriter(
                    f,
                    fieldnames=["route_index", "start", "goal", "steps", "success", "termination_reason"],
                )
                if write_header:
                    writer.writeheader()
                for row in metrics:
                    writer.writerow(row)
            print(f"[INFO] Appended metrics to {args.csv}")

    except KeyboardInterrupt:
        print("[WARN] Interrupted by user")
    finally:
        rgb_cam.stop()
        sem_cam.stop()
        rgb_cam.destroy()
        sem_cam.destroy()
        vehicle.destroy()
        cv2.destroyWindow(window_name)


if __name__ == '__main__':
    main()
