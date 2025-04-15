import gymnasium as gym
import numpy as np
import carla

class CarlaEnv(gym.Env):
    metadata = {"render_modes": ["human"], "render_fps": 30}
    
    def __init__(self):
        super(CarlaEnv, self).__init__()
        
        # Connect to CARLA server
        self.client = carla.Client('localhost', 2000)
        self.client.set_timeout(10.0)
        self.world = self.client.get_world()

        # Removes leftover vehicles
        self.cleanup()

        # Define action space (steering, throttle, brake)
        self.action_space = gym.spaces.Box(
            low=np.array([-1.0, 0.0, 0.0]),  # steering, throttle, brake
            high=np.array([1.0, 1.0, 1.0]),
            dtype=np.float32
        )
        
        # Define observation space
        self.observation_space = gym.spaces.Box(
            low = np.array([0.0, -10.0, -np.pi, 0.0]),
            high = np.array([50.0, 10.0, np.pi, 1.0]),
            dtype = np.float32
        )
        
        # Initialize vehicle and other CARLA objects
        self.vehicle = None
        self.sensors = []

        self.collision_sensor = None
        self.collision_occured = False
        self.current_step = 0
        self.max_steps = 500
        self.target_location = None
        self.lane_departures = 0

    def _get_obs(self):
        transform = self.vehicle.get_transform()
        velocity = self.vehicle.get_velocity()

        speed = np.linalg.norm([velocity.x, velocity.y, velocity.z]) * 3.6  # m/s to km/h

        location = transform.location
        waypoint = self.world.get_map().get_waypoint(location, project_to_road=True, lane_type=carla.LaneType.Driving)

        lane_center = waypoint.transform.location
        lane_offset = location.distance(lane_center)

        vehicle_forward = transform.get_forward_vector()
        lane_forward = waypoint.transform.get_forward_vector()

        dot_value = vehicle_forward.x * lane_forward.x + vehicle_forward.y * lane_forward.y + vehicle_forward.z * lane_forward.z
        dot_product = np.clip(dot_value, -1.0, 1.0)
        angle = np.arccos(dot_product)

        collision = float(self.collision_occured)

        return np.array([speed, lane_offset, angle, collision], dtype=np.float32)

    def step(self, action):
        self.current_step += 1

        # Execute action in CARLA
        if self.vehicle is not None:
            control = carla.VehicleControl()
            control.steer = float(action[0])
            control.throttle = float(action[1])
            control.brake = float(action[2])
            self.vehicle.apply_control(control)
        
        # Get new observation
        obs = self._get_obs()
        speed, lane_offset, angle, collision = obs

        # considered lane departure if we are 2.0m away from lane center
        lane_departure = abs(lane_offset) > 2.0
        if lane_departure:
            self.lane_departures += 1

        # Calculate reward
        reward = speed * (1 - abs(lane_offset) / 5.0)
        if collision:
            reward -= 100

        # Check if episode is terminated
        distance_to_target = self.vehicle.get_location().distance(self.target_location)
        route_complete = distance_to_target < 5.0

        terminated = collision or route_complete
        truncated = self.current_step > self.max_steps
        
        # Additional info
        info = {
            "speed": speed,
            "lane_offset": lane_offset,
            "angle": angle,
            "collision": bool(collision),
            "lane_departures": self.lane_departures,
            "route_complete": route_complete
        }
        
        return obs, reward, terminated, truncated, info
        
    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        
        # Clean up previous episode
        self.cleanup()

        self.collision_occured = False
        self.current_step = 0
        self.lane_departures = 0
        self.sensors = []
        
        # Spawn new vehicle
        blueprint_library = self.world.get_blueprint_library()
        vehicle_bp = blueprint_library.filter('vehicle.*')[0]
        spawn_point = self.world.get_map().get_spawn_points()[0]
        self.vehicle = self.world.spawn_actor(vehicle_bp, spawn_point)

        # Set target as a far away waypoint
        end_waypoint = self.world.get_map().get_waypoint(spawn_point.location).next(100.0)[0]
        self.target_location = end_waypoint.transform.location
        
        # Add sensors
        self._add_collision_sensor()
        self._add_camera_sensor()
        
        # Get initial observation
        observation = self._get_obs()
        
        return observation, {}
    
    def _add_collision_sensor(self):
        blueprint_library = self.world.get_blueprint_library()
        collision_bp = blueprint_library.find('sensor.other.collision')
        sensor_transform = carla.Transform()
        self.collision_sensor = self.world.spawn_actor(
            collision_bp,
            sensor_transform,
            attach_to = self.vehicle
        )

        def _on_collision(event):
            self.collision_occured = True
        
        self.collision_sensor.listen(_on_collision)
        self.sensors.append(self.collision_sensor)

    def _add_camera_sensor(self):
        blueprint_library = self.world.get_blueprint_library()
        camera_bp = blueprint_library.find('sensor.camera.rgb')
        
        # Set camera attributes
        camera_bp.set_attribute('image_size_x', '800')
        camera_bp.set_attribute('image_size_y', '600')
        camera_bp.set_attribute('fov', '90')
        
        # Set camera location relative to vehicle
        camera_transform = carla.Transform(carla.Location(x=2.5, z=0.7))
        
        self.camera = self.world.spawn_actor(
            camera_bp,
            camera_transform,
            attach_to=self.vehicle
        )
        
        self.camera_image = None
        
        def _process_camera_image(image):
            array = np.frombuffer(image.raw_data, dtype=np.dtype("uint8"))
            array = np.reshape(array, (image.height, image.width, 4))
            array = array[:, :, :3]  # Remove alpha channel
            self.camera_image = array
        
        self.camera.listen(_process_camera_image)
        self.sensors.append(self.camera)

    def get_camera_image(self):
        return self.camera_image

    def render(self, mode="human"):
        # CARLA has its own visualization
        return None
        
    def close(self):
        if self.vehicle is not None:
            if self.vehicle.is_alive:
                self.vehicle.destroy()
            self.vehicle = None
        for sensor in self.sensors:
            if sensor.is_alive:
                sensor.destroy()
        
        self.sensors = []
    
    def cleanup(self):
        # Resets the current existing vehicles and sensors
        actors = self.world.get_actors()

        vehicles = actors.filter("vehicle.*")
        for vehicle in vehicles:
            try:
                if vehicle.is_alive:
                    vehicle.destroy()
            except:
                pass
        
        sensors = actors.filter("sensor.*")
        for sensor in sensors:
            try:
                if sensor.is_alive:
                    sensor.destroy()
            except:
                pass 