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
        self.image_shape = cfg.get("image_shape", (160,160,3))
        self.max_steps = cfg.get("max_steps", 600)
        seed = int(cfg.get("seed", 0))
        self.rng = np.random.default_rng(seed)
        self.t = 0
        self.show_cam = bool(cfg.get("show_cam", False))
        self.window_name = cfg.get("window_name", "CarEnv Camera")

        # Action space
        if self.discrete:
            self.action_space = spaces.Discrete(len(self.steer_bins))
        else:
            self.action_space = spaces.Box(low=-1.0, high=1.0, shape=(1,), dtype=np.float32)

        # Observation space
        H,W,C = self.image_shape
        self.observation_space = spaces.Box(low=0, high=1, shape=(H,W,C), dtype=np.float32)

        # Connect to CARLA
        self.client = carla.Client("localhost", 2000)
        self.client.set_timeout(10.0)
        self.world = self.client.load_world(cfg.get("town","Town06"))
        self.bp_lib = self.world.get_blueprint_library()
        self.vehicle_bp = self.bp_lib.filter("vehicle.tesla.model3")[0]
        #spawn_point = self.world.get_map().get_spawn_points()[0]
        spawn_points = self.world.get_map().get_spawn_points()
        spawn = self.rng.choice(spawn_points)
        #self.vehicle.set_transform(spawn)
        self.vehicle = self.world.spawn_actor(self.vehicle_bp, spawn)

        # Attach cameras
        bound_x = 0.5 + self.vehicle.bounding_box.extent.x
        bound_y = 0.5 + self.vehicle.bounding_box.extent.y
        bound_z = 0.5 + self.vehicle.bounding_box.extent.z
        cam_trans = carla.Transform(carla.Location(x=+0.8*bound_x, y=+0.0*bound_y, z=1.3*bound_z))

        # Always create RGB cam
        rgb_bp = self.bp_lib.find("sensor.camera.rgb")
        rgb_bp.set_attribute("image_size_x", str(W))
        rgb_bp.set_attribute("image_size_y", str(H))
        rgb_bp.set_attribute("fov", "100")
        self.rgb_camera = self.world.spawn_actor(rgb_bp, cam_trans, attach_to=self.vehicle)

        # Semantic camera (for road mask)
        sem_bp = self.bp_lib.find("sensor.camera.semantic_segmentation")
        sem_bp.set_attribute("image_size_x", str(W))
        sem_bp.set_attribute("image_size_y", str(H))
        sem_bp.set_attribute("fov", "100")
        self.sem_camera = self.world.spawn_actor(sem_bp, cam_trans, attach_to=self.vehicle)

        self.rgb_image = None
        self.grayroad_image = None
        self.rgb_camera.listen(lambda data: self._rgb_callback(data))
        self.sem_camera.listen(lambda data: self._sem_callback(data))

        self.reset()

    # -------- Camera Callbacks --------
    def _rgb_callback(self, image):
        array = np.frombuffer(image.raw_data, dtype=np.uint8)
        array = array.reshape((image.height, image.width, 4))[:,:,:3]
        self.rgb_image = (array/255.0).astype(np.float32)

    def _sem_callback(self, image):
        array = np.frombuffer(image.raw_data, dtype=np.uint8)
        array = array.reshape((image.height, image.width, 4))[:,:,2]  # label in blue channel
        # Road = 128 in CityScapesPalette
        road = np.zeros_like(array, dtype=np.uint8)
        road[array == 128] = 255
        road = cv2.resize(road, (160,160))
        road = (road/255.0).astype(np.float32)
        road = np.expand_dims(road, axis=2)  # (160,160,1)
        self.grayroad_image = road

    # -------- Gym API --------
    def reset(self, *, seed=None, options=None):
        self.t = 0
        # Respawn vehicle at spawn point
        #self.vehicle.set_transform(self.world.get_map().get_spawn_points()[0])
        spawn_points = self.world.get_map().get_spawn_points()
        spawn = self.rng.choice(spawn_points)
        self.vehicle.set_transform(spawn)

        if hasattr(self.vehicle, "set_target_velocity"):
            self.vehicle.set_target_velocity(carla.Vector3D(0,0,0))
        if hasattr(self.vehicle, "set_target_angular_velocity"):
            self.vehicle.set_target_angular_velocity(carla.Vector3D(0,0,0))
        obs = self._get_obs()
        return obs, {}

    def step(self, action):
        self.t += 1
        # Action decoding
        if self.discrete:
            steer = float(self.steer_bins[int(action)])
        else:
            steer = float(np.clip(action, -1.0, 1.0)[0])
        
        # Vehicle speed (convert m/s to km/h)
        vel = self.vehicle.get_velocity()
        v2 = 3.6 * np.sqrt(vel.x**2 + vel.y**2 + vel.z**2)  # km/h

        # Longitudinal control: throttle if below 30 km/h
        if v2 < 30.0:
            throttle = 0.3
        else:
            throttle = 0.0
        #throttle = 0.3
        self.vehicle.apply_control(carla.VehicleControl(throttle=throttle, steer=steer))

        # Step sim
        self.world.tick()
        obs = self._get_obs()

        # Reward = -lane deviation
        lane_dev = self._lane_deviation()
        reward = -abs(lane_dev)

        # Done
        terminated = self.t >= self.max_steps
        truncated = False
        info = {}
        if terminated:
            info["success"] = bool(abs(lane_dev) < 0.5)
            info["lane_deviation"] = float(abs(lane_dev))
        return obs, reward, terminated, truncated, info

    # -------- Helpers --------
    def _get_obs(self):
        if self.obs_mode == "grayroad":
            while self.grayroad_image is None:
                self.world.tick()
            return self.grayroad_image
        else:
            while self.rgb_image is None:
                self.world.tick()
            return self.rgb_image

    def _lane_deviation(self):
        wp = self.world.get_map().get_waypoint(self.vehicle.get_location(), project_to_road=True)
        veh_loc = self.vehicle.get_location()
        return veh_loc.distance(wp.transform.location)

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
