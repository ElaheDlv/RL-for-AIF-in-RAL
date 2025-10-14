import gymnasium as gym
from gymnasium import spaces
import numpy as np
import carla
import cv2
import matplotlib.pyplot as plt
import random

class CarEnv(gym.Env):
    metadata = {"render_modes": []}

    def __init__(self, cfg: dict):
        super().__init__()
        self.cfg = cfg
        self.obs_mode = cfg.get("obs_mode", "rgb")
        self.discrete = cfg.get("discrete_actions", True)
        self.steer_bins = np.array(cfg.get("steer_bins", np.linspace(-1,1,20)))

        default_hw = tuple(cfg.get("camera_resolution", (160, 160)))
        default_shape = (3, *default_hw) if self.obs_mode == "rgb" else (1, *default_hw)
        requested_shape = cfg.get("image_shape", default_shape)
        if len(requested_shape) != 3:
            raise ValueError("image_shape must be a 3-tuple")

        if requested_shape[0] in (1, 3):
            channels, height, width = requested_shape
        else:
            height, width, channels = requested_shape

        self.channels = int(channels)
        self.height = int(height)
        self.width = int(width)
        self.image_shape = (self.channels, self.height, self.width)
        self.max_steps = cfg.get("max_steps", 600)
        seed = int(cfg.get("seed", 0))
        self.rng = np.random.default_rng(seed)
        self.t = 0
        self.show_cam = bool(cfg.get("show_cam", False))
        self.window_name = cfg.get("window_name", "CarEnv Camera")
        self.route_goal_tolerance = float(cfg.get("route_goal_tolerance", 5.0))
        self.max_continuous_steer = float(cfg.get("max_continuous_steer", 0.6))
        self.spin_yaw_rate_threshold = float(cfg.get("spin_yaw_rate_threshold", 1.2))
        self.spin_penalty = float(cfg.get("spin_penalty", 6.0))
        self.spin_speed_threshold_kmh = float(cfg.get("spin_speed_threshold_kmh", 5.0))
        self.min_continuous_throttle = float(cfg.get("min_continuous_throttle", 0.05))
        self.spin_consec_limit = int(cfg.get("spin_consec_limit", 15))
        lane_dev_terminate = cfg.get("lane_deviation_terminate", 2.0)
        if lane_dev_terminate is None:
            self.lane_deviation_terminate = None
        else:
            lane_dev_terminate = float(lane_dev_terminate)
            self.lane_deviation_terminate = lane_dev_terminate if lane_dev_terminate > 0 else None

        # Action space
        if self.discrete:
            self.action_space = spaces.Discrete(len(self.steer_bins))
        else:
            self.action_space = spaces.Box(low=-1.0, high=1.0, shape=(1,), dtype=np.float32)

        # Observation space (channel-first)
        self.observation_space = spaces.Box(
            low=0.0,
            high=1.0,
            shape=self.image_shape,
            dtype=np.float32,
        )

        # Connect to CARLA
        host = cfg.get("carla_host", "localhost")
        port = int(cfg.get("carla_port", 2000))
        self.client = carla.Client(host, port)
        self.client.set_timeout(10.0)
        self.world = self.client.load_world(cfg.get("town","Town06"))
        self.bp_lib = self.world.get_blueprint_library()
        self.vehicle_bp = self.bp_lib.filter("vehicle.tesla.model3")[0]
        #spawn_point = self.world.get_map().get_spawn_points()[0]
        self.map = self.world.get_map()
        self.spawn_points = self.map.get_spawn_points()
        self.routes = self._parse_routes(cfg.get("routes"))
        self._route_cursor = 0
        self.current_route = None
        self.goal_transform = None
        self.prev_steer = 0.0

        initial_spawn = self._get_spawn_for_current_route(advance=False)
        self.vehicle = self.world.spawn_actor(self.vehicle_bp, initial_spawn)

        # Attach cameras
        bound_x = 0.5 + self.vehicle.bounding_box.extent.x
        bound_y = 0.5 + self.vehicle.bounding_box.extent.y
        bound_z = 0.5 + self.vehicle.bounding_box.extent.z
        cam_trans = carla.Transform(carla.Location(x=+0.8*bound_x, y=+0.0*bound_y, z=1.3*bound_z))

        # Always create RGB cam
        rgb_bp = self.bp_lib.find("sensor.camera.rgb")
        rgb_bp.set_attribute("image_size_x", str(self.width))
        rgb_bp.set_attribute("image_size_y", str(self.height))
        rgb_bp.set_attribute("fov", "100")
        self.rgb_camera = self.world.spawn_actor(rgb_bp, cam_trans, attach_to=self.vehicle)

        # Semantic camera (for road mask)
        sem_bp = self.bp_lib.find("sensor.camera.semantic_segmentation")
        sem_bp.set_attribute("image_size_x", str(self.width))
        sem_bp.set_attribute("image_size_y", str(self.height))
        sem_bp.set_attribute("fov", "100")
        self.sem_camera = self.world.spawn_actor(sem_bp, cam_trans, attach_to=self.vehicle)

        self.rgb_image = None
        self.grayroad_image = None
        self.rgb_camera.listen(lambda data: self._rgb_callback(data))
        self.sem_camera.listen(lambda data: self._sem_callback(data))

        # Sensors need a moment to warm up before first reset
        self.reset()

    # -------- Camera Callbacks --------
    def _rgb_callback(self, image):
        array = np.frombuffer(image.raw_data, dtype=np.uint8)
        array = array.reshape((image.height, image.width, 4))[:, :, :3]
        array = array[:, :, ::-1]  # BGRA -> RGB
        self.rgb_image = (array / 255.0).astype(np.float32)

    def _sem_callback(self, image):
        # Convert the semantic segmentation frame to the CityScapes colour palette
        # so that road pixels have a deterministic BGR colour (128, 64, 128).
        image.convert(carla.ColorConverter.CityScapesPalette)
        array = np.frombuffer(image.raw_data, dtype=np.uint8)
        array = array.reshape((image.height, image.width, 4))[:, :, :3]  # drop alpha

        road_mask = (
            (array[:, :, 0] == 128) &  # Blue
            (array[:, :, 1] == 64)  &  # Green
            (array[:, :, 2] == 128)    # Red
        )

        road = np.zeros((image.height, image.width), dtype=np.uint8)
        road[road_mask] = 255

        road = cv2.resize(road, (self.width, self.height))
        road = (road / 255.0).astype(np.float32)
        road = np.expand_dims(road, axis=2)  # (H,W,1)
        self.grayroad_image = road

    # -------- Route helpers --------
    def _parse_routes(self, routes):
        if not routes:
            return None
        parsed = []
        for entry in routes:
            if isinstance(entry, dict):
                start_idx = entry.get("start")
                goal_idx = entry.get("goal")
            else:
                try:
                    start_idx, goal_idx = entry
                except (TypeError, ValueError):
                    continue
            if start_idx is None or goal_idx is None:
                continue
            if not (0 <= int(start_idx) < len(self.spawn_points)):
                raise ValueError(f"Route start index {start_idx} outside available spawn points")
            if not (0 <= int(goal_idx) < len(self.spawn_points)):
                raise ValueError(f"Route goal index {goal_idx} outside available spawn points")
            parsed.append({"start": int(start_idx), "goal": int(goal_idx)})
        return parsed if parsed else None

    def _get_spawn_for_current_route(self, advance: bool = True):
        if not self.routes:
            self.current_route = None
            self.goal_transform = None
            return self.rng.choice(self.spawn_points)
        idx = self._route_cursor % len(self.routes)
        route = self.routes[idx]
        self.current_route = route
        self.goal_transform = self.spawn_points[route["goal"]]
        if advance:
            self._route_cursor = (idx + 1) % len(self.routes)
        return self.spawn_points[route["start"]]

    def _goal_distance(self):
        if self.goal_transform is None:
            return None
        veh_loc = self.vehicle.get_transform().location
        return veh_loc.distance(self.goal_transform.location)

    # -------- Gym API --------
    def reset(self, *, seed=None, options=None):
        self.t = 0
        spawn = self._get_spawn_for_current_route(advance=True)
        self.vehicle.set_transform(spawn)

        if hasattr(self.vehicle, "set_target_velocity"):
            self.vehicle.set_target_velocity(carla.Vector3D(0,0,0))
        if hasattr(self.vehicle, "set_target_angular_velocity"):
            self.vehicle.set_target_angular_velocity(carla.Vector3D(0,0,0))
        # Clear cached frames so we do not return stale observations from the previous episode.
        self.rgb_image = None
        self.grayroad_image = None
        # Step the simulator a few frames to give sensors time to produce fresh data at the new pose.
        for _ in range(3):
            self.world.tick()
        self.prev_steer = 0.0
        self.prev_lane_dev = 0.0
        self.spin_steps = 0
        obs = self._get_obs()
        info = {}
        if self.current_route:
            info["route"] = dict(self.current_route)
        return obs, info

    def step(self, action):
        self.t += 1
        # Action decoding
        if self.discrete:
            steer = float(self.steer_bins[int(action)])
        else:
            steer = float(np.clip(action, -self.max_continuous_steer, self.max_continuous_steer)[0])
        
        # Vehicle speed (convert m/s to km/h)
        vel = self.vehicle.get_velocity()
        v2 = 3.6 * np.sqrt(vel.x**2 + vel.y**2 + vel.z**2)  # km/h
        ang_vel = self.vehicle.get_angular_velocity()
        yaw_rate = float(abs(ang_vel.z))
        if self.discrete:
            throttle = 0.3 if v2 < 30.0 else 0.0
        else:
            base_throttle = 0.32 if v2 < 25.0 else 0.18
            if v2 > self.spin_speed_threshold_kmh and yaw_rate > 0.25:
                throttle_scale = max(0.15, 1.0 - 0.5 * (yaw_rate / max(self.spin_yaw_rate_threshold, 1e-3)))
            else:
                throttle_scale = 1.0
            throttle = float(np.clip(base_throttle * throttle_scale, self.min_continuous_throttle, 0.38))
        self.vehicle.apply_control(carla.VehicleControl(throttle=throttle, steer=steer))

        # Step sim
        self.world.tick()
        obs = self._get_obs()
        # --- pose and reward components ---
        transform = self.vehicle.get_transform()
        veh_loc = transform.location
        veh_rot = transform.rotation
        lane_dev = float(self._lane_deviation())
        yaw_err = float(abs(self._heading_error()))
        steer_rate = float(abs(steer - self.prev_steer))
        self.prev_steer = steer
        waypoint = self.map.get_waypoint(
            veh_loc,
            project_to_road=True,
            lane_type=carla.LaneType.Driving,
        )
        wp_transform = waypoint.transform if waypoint is not None else None

        base_reward = 0.2
        reward = (
            base_reward
            - 1.0 * min(lane_dev, 2.0)
            - 0.3 * min(yaw_err, 0.3)
            - 0.02 * steer_rate
        )
        if not self.discrete and v2 > self.spin_speed_threshold_kmh:
            reward -= 0.5 * min(yaw_rate, 2.0)

        terminated = False
        truncated = False
        info = {
            "lane_deviation": lane_dev,
            "yaw_error": yaw_err,
            "yaw_rate": yaw_rate,
            "steer_rate": steer_rate,
            "success": False,
            "vehicle_location_x": float(veh_loc.x),
            "vehicle_location_y": float(veh_loc.y),
            "vehicle_location_z": float(veh_loc.z),
            "vehicle_rotation_pitch": float(veh_rot.pitch),
            "vehicle_rotation_yaw": float(veh_rot.yaw),
            "vehicle_rotation_roll": float(veh_rot.roll),
            "vehicle_speed_kmh": float(v2),
            "waypoint_location_x": float(wp_transform.location.x) if wp_transform else None,
            "waypoint_location_y": float(wp_transform.location.y) if wp_transform else None,
            "waypoint_location_z": float(wp_transform.location.z) if wp_transform else None,
        }

        if self.lane_deviation_terminate is not None and lane_dev > self.lane_deviation_terminate:
            reward -= 5.0
            terminated = True
            info["terminated_reason"] = "lane_deviation"

        if not self.discrete and v2 > self.spin_speed_threshold_kmh:
            if yaw_rate > self.spin_yaw_rate_threshold:
                self.spin_steps += 1
            else:
                self.spin_steps = 0

            if self.spin_steps >= self.spin_consec_limit:
                reward -= self.spin_penalty
                terminated = True
                info["terminated_reason"] = "spin"
        else:
            self.spin_steps = 0

        if self.t >= self.max_steps:
            terminated = True
            info["success"] = lane_dev < 0.5
        self.prev_lane_dev = lane_dev

        return obs, reward, terminated, truncated, info
    # -------- Helpers --------
    def _get_obs(self):
        def _ensure(obs):
            if obs is None:
                return None
            arr = np.array(obs, dtype=np.float32)
            if arr.ndim == 2:
                arr = arr[..., np.newaxis]
            if arr.shape[0] == self.channels and arr.shape[1] == self.height:
                # already channel-first
                return arr
            if arr.shape[0] == self.height and arr.shape[1] == self.width:
                return np.transpose(arr, (2, 0, 1))
            raise ValueError(
                f"Unexpected observation shape {arr.shape}; expected (H,W,C) or (C,H,W) with H={self.height}, W={self.width}, C={self.channels}"
            )

        if self.obs_mode == "grayroad":
            while self.grayroad_image is None:
                self.world.tick()
            obs = _ensure(self.grayroad_image)
            #print("[DBG] grayroad obs shape:", obs.shape)
            return obs
        else:
            while self.rgb_image is None:
                self.world.tick()
            obs = _ensure(self.rgb_image)
            #print("[DBG] rgb obs shape:", obs.shape)
            return obs

    def _lane_deviation(self):
        wp = self.map.get_waypoint(self.vehicle.get_location(), project_to_road=True)
        veh_loc = self.vehicle.get_location()
        return veh_loc.distance(wp.transform.location)
    
    
    def _heading_error(self):
        wp = self.map.get_waypoint(self.vehicle.get_location(), project_to_road=True)
        lane_yaw = wp.transform.rotation.yaw * np.pi/180.0
        veh_yaw = self.vehicle.get_transform().rotation.yaw * np.pi/180.0
        return np.arctan2(np.sin(veh_yaw - lane_yaw), np.cos(veh_yaw - lane_yaw))

    def close(self):
        for actor in [self.rgb_camera, self.sem_camera, self.vehicle]:
            if actor is not None:
                actor.destroy()
        if self.show_cam:
            try:
                cv2.destroyWindow(self.window_name)
            except Exception:
                pass


    def render(self, mode="human"):
        if not self.show_cam:
            return
        frame = None
        if self.obs_mode == "grayroad":
            frame = self.grayroad_image
        else:
            frame = self.rgb_image
        if frame is None:
            return
        frame = np.array(frame)
        frame = np.clip(frame * 255.0, 0, 255).astype(np.uint8)
        if frame.ndim == 3 and frame.shape[2] == 1:
            frame = np.repeat(frame, 3, axis=2)
        if frame.ndim == 2:
            frame = cv2.cvtColor(frame, cv2.COLOR_GRAY2BGR)
        else:
            frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
        cv2.imshow(self.window_name, frame)
        cv2.waitKey(1)

    def set_show_cam(self, flag: bool):
        flag = bool(flag)
        if self.show_cam and not flag:
            try:
                cv2.destroyWindow(self.window_name)
            except Exception:
                pass
        self.show_cam = flag
