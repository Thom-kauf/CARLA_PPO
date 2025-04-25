import gymnasium as gym
import numpy as np
import carla
import time

class CarlaEnvVanilla(gym.Env):
    metadata = {"render_modes": ["human"], "render_fps": 30}
    
    def __init__(self):
        super(CarlaEnvVanilla, self).__init__()
        
        # Settings
        
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
            low = np.array([0.0, -10.0, -np.pi, 0.0, 0.0, 0.0,-np.pi], dtype=np.float32),
            high = np.array([50.0, 10.0, np.pi, 1.0, 200.0,200.0, np.pi], dtype=np.float32),
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

        self.progress_buffer_size = 50  # Track progress over last 10 frames
        self.progress_buffer = []
        self.distance_buffer = []


        # maybe add this
        # self.previous_locations_buffer_size = 500
        # self.previous_locations_buffer = []



    def _get_obs(self):
        transform = self.vehicle.get_transform()
        velocity = self.vehicle.get_velocity()

        location = transform.location

        # Find index of the closest waypoint in the route
        closest_idx = min(range(len(self.route)), key=lambda i: self.route[i].transform.location.distance(location))

        # Default to current waypoint if no next waypoint exists
        if closest_idx + 1 < len(self.route):
            waypoint = self.route[closest_idx + 1]
        else:
            waypoint = self.route[closest_idx]

        # Now you can calculate distance to the next waypoint
        lane_center = waypoint.transform.location

        delta = location - lane_center
        right_vec = waypoint.transform.get_right_vector()

        lane_offset = delta.x * right_vec.x + delta.y * right_vec.y#location.distance(lane_center)

        vehicle_forward = transform.get_forward_vector()
        lane_forward = waypoint.transform.get_forward_vector()

        dot_value = vehicle_forward.x * lane_forward.x + vehicle_forward.y * lane_forward.y + vehicle_forward.z * lane_forward.z
        dot_product = np.clip(dot_value, -1.0, 1.0)
        angle = np.arccos(dot_product)

        collision_occured = self.collision_occured

        # Calculate distance and direction to target
        # print("\n\n\n\n", waypoint, "\n\n\n\n")
        distance_to_waypoint = location.distance(waypoint.transform.location)

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

        # Calculate speed in km/h
        speed = np.sqrt(velocity.x**2 + velocity.y**2 + velocity.z**2) * 3.6

        return np.array([speed, lane_offset, angle, collision_occured, distance_to_target, distance_to_waypoint, target_angle], dtype=np.float32)



    def step(self, action):

        self.current_step += 1

        # Execute action in CARLA
        if self.vehicle is not None:
            control = carla.VehicleControl()
            control.steer = float(action[0])
            control.throttle = float((action[1] + 1) / 2)

            # Apply brake if throttle is low, and don't apply brake if throttle is high
            control.brake = float((action[2] + 1) / 2) if control.throttle < 0.1 else 0.0

            self.vehicle.apply_control(control)



        # Get new observation
        obs = self._get_obs()
        speed, lane_offset, angle, collision_occured, distance_to_target, distance_to_waypoint, target_angle = obs


        immediate_progress = self.previous_distance - distance_to_waypoint

        self.progress_buffer.append(immediate_progress)
        self.distance_buffer.append(distance_to_waypoint)

        self.previous_distance = distance_to_waypoint

        # Keep buffer at fixed size
        if len(self.progress_buffer) > self.progress_buffer_size:
            self.progress_buffer.pop(0)
            self.distance_buffer.pop(0)
            
        # Calculate buffered progress metrics
        avg_progress = np.mean(self.progress_buffer)  # Average progress over buffer
        total_progress = self.distance_buffer[0] - self.distance_buffer[-1]  # Total progress over buffer
        
        # Calculate progress reward using both immediate and buffered progress
        if avg_progress > 0:  # Consistent progress towards goal
            progress_reward = avg_progress * 10.0 + total_progress * 5.0
        else:  # Moving away from goal
            progress_reward = avg_progress * 10.0

        if distance_to_waypoint < 0.1:
            progress_reward += 20


        # Direction reward based on target angle
        direction_reward = np.cos(target_angle) * 10#50.0  # Max 20 when facing target

        lane_departure = False


        if abs(lane_offset) > 2.0:#1:
            lane_departure = True
            self.lane_departures += 1

        reward = 0.0
        time_penalty = -0.5

        if collision_occured or lane_departure:
            reward = -50.0
        else:
            reward = progress_reward + direction_reward 

        reward += time_penalty


        # Check if episode is terminated
        route_complete = distance_to_target < 5.0
        if route_complete:
            reward += 500


        stuck = self.current_step > 5_000 and speed < 0.1  # Terminate if not moving after 500 steps

        terminated = bool(collision_occured  or route_complete)
        truncated = stuck

        info = {
            "speed": speed,
            "lane_offset": lane_offset,
            "lane_departures": self.lane_departures,
            "collision": collision_occured,
            "throttle": float((action[1] + 1) / 2),
            "brake": float((action[2] + 1) / 2),
            "steer": float(action[0]),
            "distance_to_target": distance_to_target,
            "distance to waypoint": distance_to_waypoint,
            "target_angle": target_angle,
            "immediate_progress": immediate_progress,
            "average_progress": avg_progress,
            "total_buffer_progress": total_progress,
            "progress_reward": progress_reward,
            "direction_reward": direction_reward,
            "route_complete": route_complete,
            "time_penalty": time_penalty,
            "total_reward": reward
        }
        
        if self.current_step % 1000 == 0:
            print(f"Step {self.current_step}: {info}")


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

        # Add noise to the spawn location
        noise_x = np.random.uniform(-5.0, 5.0)  # Adjust the range as needed
        # noise_y = np.random.uniform(-5.0, 5.0)  # Adjust the range as needed
        spawn_point.location.x += noise_x
        # spawn_point.location.y += noise_y


        self.vehicle = self.world.spawn_actor(vehicle_bp, spawn_point)
        self.vehicle.set_simulate_physics(True)

        # Single control application with initial throttle
        control = carla.VehicleControl(
            throttle=0,
            brake=0.0,
            steer=0.0,
            hand_brake=False,
            reverse=False,
            manual_gear_shift=False
        )
        self.vehicle.apply_control(control)
        
        # Set target waypoint
        spawn_loc = self.vehicle.get_location()
        # print("\n\nSpawn location: ", spawn_loc)

        
        self.reference_waypoint = self.world.get_map().get_waypoint(
            spawn_loc,
            project_to_road = True,
            lane_type = carla.LaneType.Driving
        )

        
        self.reference_lane = self.reference_waypoint



        # Set target manually: 100 meters forward in +X direction
        self.target_location = carla.Location(
            x=spawn_loc.x + 100.0,
            y=spawn_loc.y,
            z=spawn_loc.z
        )

        # Now find the closest waypoint to that target manually
        self.target_waypoint = self.world.get_map().get_waypoint(
            self.target_location,
            project_to_road=True,
            lane_type=carla.LaneType.Driving
        )







        self.route = []
        step_size = 2.0  # meters
        distance = self.target_location.x - spawn_loc.x  # assuming +X only
        num_steps = int(distance / step_size)

        for i in range(num_steps + 1):
            x = spawn_loc.x + i * step_size
            manual_loc = carla.Location(x=x, y=spawn_loc.y, z=spawn_loc.z)

            wp = self.world.get_map().get_waypoint(
                manual_loc,
                project_to_road=True,
                lane_type=carla.LaneType.Driving
            )

            # print(wp, "\n")

            if wp:  # in case projection fails
                self.route.append(wp)



        print(f"Spawn @ {spawn_loc}")
        print(f"Target (manual +X) @ {self.target_location}")
        print(f"Snapped Target Waypoint @ {self.target_waypoint.transform.location}")


        

        # print("\n\n target waypoint loc: ", self.target_waypoint)

        self.previous_distance = spawn_loc.distance(self.target_location)
        
        # Add sensors
        self._add_collision_sensor()
        self._add_camera_sensor()
        
        # Get initial observation
        observation = self._get_obs()
        print("done")
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




class AnxiousCarlaEnv(CarlaEnvVanilla):
    """
    A version of CarlaEnv that is 'anxious' about lane departures.
    This class inherits all methods from CarlaEnv except for the step method,
    which is modified to penalize lane offset more aggressively.
    """
    
    def __init__(self):
        # Call the parent class constructor
        super(AnxiousCarlaEnv, self).__init__()
        print("Using AnxiousCarlaEnv - with enhanced lane offset penalties")

    def step(self, action):
        """
        Override the step method to apply different reward calculations,
        specifically adding a continuous lane offset penalty.
        """
        self.current_step += 1

        # Execute action in CARLA
        if self.vehicle is not None:
            control = carla.VehicleControl()
            control.steer = float(action[0])
            control.throttle = float((action[1] + 1) / 2)

            # Apply brake if throttle is low, and don't apply brake if throttle is high
            control.brake = float((action[2] + 1) / 2) if control.throttle < 0.1 else 0.0

            self.vehicle.apply_control(control)

        # Get new observation
        obs = self._get_obs()
        speed, lane_offset, angle, collision_occured, distance_to_target, target_angle = obs

        immediate_progress = self.previous_distance - distance_to_target

        self.progress_buffer.append(immediate_progress)
        self.distance_buffer.append(distance_to_target)

        self.previous_distance = distance_to_target

        # Keep buffer at fixed size
        if len(self.progress_buffer) > self.progress_buffer_size:
            self.progress_buffer.pop(0)
            self.distance_buffer.pop(0)
            
        # Calculate buffered progress metrics
        avg_progress = np.mean(self.progress_buffer)  # Average progress over buffer
        total_progress = self.distance_buffer[0] - self.distance_buffer[-1]  # Total progress over buffer
        
        # Calculate progress reward using both immediate and buffered progress
        if avg_progress > 0:  # Consistent progress towards goal
            progress_reward = avg_progress * 100.0 + total_progress * 50.0
        else:  # Moving away from goal
            progress_reward = avg_progress * 100.0

        # Direction reward based on target angle
        direction_reward = np.cos(target_angle) * 50.0  # Max 50 when facing target

        # Track lane departures but don't terminate
        if abs(lane_offset) > 2.0:
            self.lane_departures += 1

        # Apply a continuous penalty for lane offset
        lane_offset_penalty = abs(lane_offset) * -100.0

        # Time penalty to encourage faster completion
        time_penalty = -0.5

        # Calculate final reward
        if collision_occured:
            reward = -10_000.0
        else:
            reward = progress_reward + direction_reward + lane_offset_penalty

        reward += time_penalty

        # Check if episode is terminated
        route_complete = distance_to_target < 5.0
        stuck = self.current_step > 5_000 and speed < 0.1  # Terminate if not moving after 500 steps

        terminated = bool(collision_occured or route_complete)
        truncated = stuck

        # Extended info dictionary with lane_offset_penalty
        info = {
            "speed": speed,
            "lane_offset": lane_offset,
            "lane_departures": self.lane_departures,
            "collision": collision_occured,
            "throttle": float((action[1] + 1) / 2),
            "brake": float((action[2] + 1) / 2),
            "steer": float(action[0]),
            "distance_to_target": distance_to_target,
            "target_angle": target_angle,
            "immediate_progress": immediate_progress,
            "average_progress": avg_progress,
            "total_buffer_progress": total_progress,
            "progress_reward": progress_reward,
            "direction_reward": direction_reward,
            "lane_offset_penalty": lane_offset_penalty,
            "route_complete": route_complete,
            "time_penalty": time_penalty,
            "total_reward": reward
        }
        
        if self.current_step % 1000 == 0:
            print(f"Step {self.current_step}: {info}")

        return obs, reward, terminated, truncated, info