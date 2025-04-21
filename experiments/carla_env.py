import gymnasium as gym
import numpy as np
import carla
import time

class CarlaEnv(gym.Env):
    metadata = {"render_modes": ["human"], "render_fps": 30}
    
    def __init__(self, disable_boost=False, initial_throttle=0.5):
        super(CarlaEnv, self).__init__()
        
        # Settings
        self.disable_boost = disable_boost
        self.initial_throttle = initial_throttle
        
        # Connect to CARLA server
        self.client = carla.Client('localhost', 2000)
        self.client.set_timeout(10.0)
        self.world = self.client.get_world()

        # Removes leftover vehicles
        self.cleanup()

        # Define action space (steering, throttle, brake)
        self.action_space = gym.spaces.Box(
            low=np.array([-1.0, 0.0, 0.0], dtype=np.float32),  # steering, throttle, brake
            high=np.array([1.0, 1.0, 1.0], dtype=np.float32),
            dtype=np.float32
        )
        
        # Define observation space with 6 continuous values:
        # 1. Speed (0 to 50 km/h)
        # 2. Lane offset (-10 to 10 meters from center)
        # 3. Angle (-π to π radians relative to lane direction) 
        # 4. Collision indicator (0 or 1)
        # 5. Distance to target (0 to 200 meters)
        # 6. Direction to target (-π to π radians)
        self.observation_space = gym.spaces.Box(
            low = np.array([0.0, -10.0, -np.pi, 0.0, 0.0, -np.pi], dtype=np.float32),
            high = np.array([50.0, 10.0, np.pi, 1.0, 200.0, np.pi], dtype=np.float32),
            dtype = np.float32
        )
        
        # Initialize vehicle and other CARLA objects
        self.vehicle = None
        self.sensors = []

        self.collision_sensor = None
        self.collision_occured = False
        self.current_step = 0
        self.max_steps = 10_000#500
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

        # Calculate distance and direction to target
        distance_to_target = location.distance(self.target_location)
        
        # Calculate direction to target relative to vehicle's forward direction
        to_target = carla.Vector3D(
            x=self.target_location.x - location.x,
            y=self.target_location.y - location.y,
            z=0  # Ignore vertical component
        )
        
        # Normalize to_target vector
        target_length = np.sqrt(to_target.x**2 + to_target.y**2)
        if target_length > 0:
            to_target.x /= target_length
            to_target.y /= target_length
        
        # Calculate angle between vehicle's forward vector and target direction
        target_dot = vehicle_forward.x * to_target.x + vehicle_forward.y * to_target.y
        target_dot = np.clip(target_dot, -1.0, 1.0)
        target_angle = np.arccos(target_dot)
        
        # Determine sign of the angle (left or right of vehicle)
        cross_product = vehicle_forward.x * to_target.y - vehicle_forward.y * to_target.x
        if cross_product < 0:
            target_angle = -target_angle

        return np.array([speed, lane_offset, angle, collision, distance_to_target, target_angle], dtype=np.float32)

    def step(self, action):
        self.current_step += 1

        previous_distance = self.vehicle.get_location().distance(self.target_location)


        # Get new observation
        obs = self._get_obs()
        speed, lane_offset, angle, collision, distance_to_target, target_angle = obs

        # Calculate progress
        progress = previous_distance - distance_to_target  # Positive if getting closer
        
        # Reward components
        speed_reward = min(speed * 1.0, 50.0)  # Increased speed reward, max 50
        
        # Direction reward based on target angle
        direction_reward = np.cos(target_angle) * 20.0  # Max 20 when facing target
        
        # Progress reward with  multiplier
        progress_reward = progress * 100.0  
        
        # Movement penalty only if very slow and not well-aligned
        movement_penalty = -5.0 if (speed < 0.2 and self.current_step > 20) else 0.0  # Much smaller penalty, and only after 20 steps
        
        # Penalties
        collision_penalty = -50.0 if collision else 0.0
        lane_departure_penalty = -20.0 if abs(lane_offset) > 2.0 else 0.0
        
        # Total reward
        reward = speed_reward + direction_reward + progress_reward + movement_penalty + collision_penalty + lane_departure_penalty

        # Debug output every 100 steps
        if self.current_step % 100 == 0:
            print(f"Step {self.current_step}: dist={distance_to_target:.2f}m, "
                  f"angle={np.degrees(target_angle):.1f}°, "
                  f"progress={progress:.2f}m, speed={speed:.2f}km/h")

        # Check if episode is terminated
        route_complete = distance_to_target < 5.0
        stuck = self.current_step > 5_000 and speed < 0.1  # Terminate if not moving after 500 steps

        terminated = collision or route_complete or stuck
        truncated = self.current_step > self.max_steps
        
        # Additional info
        info = {
            "speed": speed,
            "lane_offset": lane_offset,
            "angle": angle,
            "collision": bool(collision),
            "lane_departures": self.lane_departures,
            "route_complete": route_complete,
            "throttle": float((action[1] + 1) / 2),
            "brake": float((action[2] + 1) / 2),
            "steer": float(action[0]),
            "distance_to_target": distance_to_target,
            "target_angle": target_angle,
            "progress": progress,
            "progress_reward": progress_reward,
            "speed_reward": speed_reward,
            "direction_reward": direction_reward,
            "movement_penalty": movement_penalty,
            "collision_penalty": collision_penalty,
            "lane_departure_penalty": lane_departure_penalty,
            "total_reward": reward
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
        vehicle_bp.set_attribute('role_name', 'hero')
        self.vehicle = self.world.spawn_actor(vehicle_bp, spawn_point)

        # Initial vehicle setup
        self.vehicle.set_simulate_physics(False)
        control = carla.VehicleControl(
            throttle=0.0,
            brake=1.0,
            steer=0.0,
            hand_brake=True,
            reverse=False,
            manual_gear_shift=False
        )
        self.vehicle.apply_control(control)
        
        time.sleep(0.1)
        self.vehicle.set_simulate_physics(True)

        # Important: Create and apply a new control command with hand brake released
        control = carla.VehicleControl(
            throttle=self.initial_throttle,  # Apply initial throttle to get moving
            brake=0.0,
            steer=0.0,
            hand_brake=False,
            reverse=False,
            manual_gear_shift=False
        )
        self.vehicle.apply_control(control)
        time.sleep(0.5)  # Wait a bit for the initial throttle to take effect
        
        # Set target waypoint
        spawn_loc = self.vehicle.get_location()
        self.target_location = carla.Location(
            x=spawn_loc.x + 100.0,  # Go 100m forward in x direction
            y=spawn_loc.y,          # Keep same y coordinate
            z=spawn_loc.z           # Keep same z coordinate
        )
        
        # Debugging information
        vehicle_transform = self.vehicle.get_transform()
        vehicle_forward = vehicle_transform.get_forward_vector()
        
        print(f"\nRESET INFO:")
        print(f"  Vehicle position: ({vehicle_transform.location.x:.2f}, {vehicle_transform.location.y:.2f}, {vehicle_transform.location.z:.2f})")
        print(f"  Vehicle forward: ({vehicle_forward.x:.2f}, {vehicle_forward.y:.2f}, {vehicle_forward.z:.2f})")
        print(f"  Target position: ({self.target_location.x:.2f}, {self.target_location.y:.2f}, {self.target_location.z:.2f})")
        print(f"  Initial distance: {vehicle_transform.location.distance(self.target_location):.2f}m")
        
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