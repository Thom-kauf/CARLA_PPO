import gymnasium as gym
import numpy as np
import carla
import time

class CarlaEnvVanilla(gym.Env):
    metadata = {"render_modes": ["human"], "render_fps": 30}
    
    def __init__(self):
        super(CarlaEnvVanilla, self).__init__()
        
        # Settings
        self.lane_departure_penalty = -0.05#-0.005
        self.collision_penalty = -100.0
        self.waypoint_completed_reward = 100.0
        self.target_progress_reward = 50
        self.lane_offset_penalty = -0.01
        self.progress_buffer_size = 2

        self.route_completed_reward = 200.0

        self.stuck_steps = 5_000


        # Connect to CARLA s
        self.client = carla.Client('localhost', 2000)
        self.client.set_timeout(10.0)
        self.world = self.client.get_world()

        # Removes leftover vehicles
        self.cleanup()

        # Define action space (steering, throttle, brake)
        self.action_space = gym.spaces.Box(
            low=np.array([-1.0, -1.0, -1.0], dtype=np.float32),  # steering, throttle, brake
            high=np.array([1.0, 1.0, 1.0], dtype=np.float32),
            dtype=np.float32
        )
        
        # Define observation space with 6 continuous values:
        # 1. Speed (0 to 50 km/h)
        # 2. Lane offset (-10 to 10 meters from center)
        # 3. Collision indicator (0 or 1)
        # 4. Distance to target (0 to 200 meters)
        # 5. Distance to waypoint (0 to 200 meters)
        # 5. Direction to target (-π to π radians)
        # 6. Direction to waypoint (-π to π radians)
        self.observation_space = gym.spaces.Box(
            low = np.array([0.0,  -10.0, 0.0, 0.0,  0.0, -np.pi, -np.pi], dtype=np.float32),
            high = np.array([50.0, 10.0, 1.0, 200.0, 200.0, np.pi, np.pi], dtype=np.float32),
            dtype = np.float32
        )
        
        # Initialize vehicle and other CARLA objects
        self.vehicle = None
        self.sensors = []

        self.collision_sensor = None
        self.collision_occured = False
        self.current_step = 0
        self.max_steps = 15_000#500


    def _get_obs(self):
        transform = self.vehicle.get_transform()
        velocity = self.vehicle.get_velocity()
        location = transform.location
        
        # Get the current waypoint target - find the closest waypoint ahead of the vehicle
        current_waypoint = None
        curr_wp_idx = 0
        min_distance = float('inf')
        for i in range(self.most_recent_waypoint_reached + 1, len(self.route)):
            wp = self.route[i]
            distance = location.distance(wp.transform.location)
            if distance < min_distance:
                min_distance = distance
                current_waypoint = wp
                curr_wp_idx = i
        if current_waypoint is None:
            current_waypoint = self.route[-1]  # Fallback to last waypoint if none found
            curr_wp_idx = len(self.route) - 1

        
        # Calculate lane information
        lane_center = current_waypoint.transform.location
        delta = location - lane_center
        right_vec = current_waypoint.transform.get_right_vector()



        
        # Calculate lateral offset from lane center
        lane_offset = delta.x * right_vec.x + delta.y * right_vec.y
        
        # Calculate distance to final target
        distance_to_target = location.distance(self.target_location)
        distance_to_waypoint = location.distance(current_waypoint.transform.location)
        
        # Calculate angle to target
        target_direction = carla.Vector3D(
            x=self.target_location.x - location.x,
            y=self.target_location.y - location.y,
            z=0
        )
        
        # Normalize target direction vector
        target_length = np.sqrt(target_direction.x**2 + target_direction.y**2)
        if target_length > 0:
            target_direction.x /= target_length
            target_direction.y /= target_length
        
        # Calculate direction to waypoint
        waypoint_direction = carla.Vector3D(
            x=current_waypoint.transform.location.x - location.x,
            y=current_waypoint.transform.location.y - location.y,
            z=0
        )
        

        
        

        # Calculate angle between vehicle direction and target direction
        vehicle_forward = transform.get_forward_vector()

        # Normalize waypoint direction vector
        waypoint_length = np.sqrt(waypoint_direction.x**2 + waypoint_direction.y**2)
        if waypoint_length > 0:
            waypoint_direction.x /= waypoint_length
            waypoint_direction.y /= waypoint_length
            
        # Calculate angle between vehicle direction and waypoint direction
        waypoint_dot = vehicle_forward.x * waypoint_direction.x + vehicle_forward.y * waypoint_direction.y
        waypoint_dot = np.clip(waypoint_dot, -1.0, 1.0)
        waypoint_angle = np.arccos(waypoint_dot)
        
        # Determine sign of the angle (left or right of vehicle)
        waypoint_cross = vehicle_forward.x * waypoint_direction.y - vehicle_forward.y * waypoint_direction.x
        if waypoint_cross < 0:
            waypoint_angle = -waypoint_angle

        # print(f"Waypoint angle: {waypoint_angle}")
        # print(f"Current waypoint: {curr_wp_idx}, distance: {distance_to_waypoint}, angle: {waypoint_angle}")

        target_dot = vehicle_forward.x * target_direction.x + vehicle_forward.y * target_direction.y
        target_dot = np.clip(target_dot, -1.0, 1.0)
        target_angle = np.arccos(target_dot)
        
        # Determine sign of the angle (left or right of vehicle)
        cross_product = vehicle_forward.x * target_direction.y - vehicle_forward.y * target_direction.x
        if cross_product < 0:
            target_angle = -target_angle
        
        # Calculate speed
        speed = np.sqrt(velocity.x**2 + velocity.y**2 + velocity.z**2) * 3.6  # km/h
        
       # 1. Speed (0 to 50 km/h)
        # 2. Lane offset (-10 to 10 meters from center)
        # 3. Collision indicator (0 or 1)
        # 4. Distance to target (0 to 200 meters)
        # 5. Distance to waypoint (0 to 200 meters)
        # 5. Direction to target (-π to π radians)
        return np.array([
            speed, 
            lane_offset, 
            self.collision_occured, 
            distance_to_target, 
            distance_to_waypoint,
            target_angle,
            waypoint_angle
        ], dtype=np.float32)

    def updated_waypoints_reached(self, reward_components):
        """
        Update the waypoint index based on the agent's position and track visited waypoints.
        The agent only receives credit for visiting each waypoint once.
        
        This method:
        1. Tracks which waypoints have been visited
        2. Updates the current target waypoint for navigation
        3. Makes sure the agent doesn't get credit for revisiting waypoints
        
        Returns:
            tuple: (new_waypoint_index, new_distance, any_waypoint_newly_reached)
        """
        vehicle_location = self.vehicle.get_location()
        
        # Check for waypoints within reach (not just the current target)
        # This allows the agent to get credit for any waypoint it passes near
        for i, wp in enumerate(self.route):
            wp_distance = vehicle_location.distance(wp.transform.location)
            # print(f"Waypoint {i} distance: {wp_distance}")
            
            # If the vehicle is close enough to this waypoint and hasn't reached it before
            if wp_distance < 1.0 and ( i not in self.reached_waypoints) and (i > max(self.reached_waypoints) if self.reached_waypoints else 0):  # 0.5 meters threshold
                # Mark this waypoint as reached
                self.reached_waypoints.add(i)
                self.most_recent_waypoint_reached = i
                self.total_waypoints_reached += 1
                print(f"Waypoint {i} reached")
                reward_components["waypoint_progress"] +=  self.waypoint_completed_reward * 2 if i < 5 else self.waypoint_completed_reward

        
    def _calculate_target_progress_reward(self, current_distance):
        """
        Calculate reward based on progress toward the final target over a buffer window.
        
        Args:
            current_distance: Current distance to target
            
        Returns:
            float: Reward for progress toward target
        """
        # Only calculate reward if the buffer is full
        if len(self.progress_buffer) >= self.progress_buffer_size:
            oldest_distance = self.progress_buffer[0]
            # Reward positively if distance decreased over the buffer window and negatively if it increased
            return self.target_progress_reward * (oldest_distance - current_distance)

        return 0.0

    def _calculate_lane_offset_penalty(self, lane_offset):
        """
        Calculate penalty based on lane offset.
        
        Args:
            lane_offset (float): Current lane offset from center
            
        Returns:
            float: Penalty value based on lane offset
        """


        if abs(lane_offset) > 0.15:
            # print(f"Lane offset: {lane_offset}")
            return abs(lane_offset) * self.lane_offset_penalty
        else:
            return 0.0


    def _calculate_total_reward(self, components):
        """
        Calculate total reward from individual components.
        
        Args:
            components: Dictionary of reward components and their values
            
        Returns:
            float: Total reward
        """
        # Calculate normal reward from all components
        reward = components["target_progress"] + components["waypoint_progress"] #+ components["lane_departure_penalty"] 
        self.cumulative_reward_components["target_progress"] += components["target_progress"]
        self.cumulative_reward_components["waypoint_progress"] += components["waypoint_progress"]
        # self.cumulative_reward_components["lane_departure_penalty"] += components["lane_departure_penalty"]
        

        if components["collision_penalty"]:
            reward += self.collision_penalty
            self.cumulative_reward_components["collision_count"] += 1
            self.cumulative_reward_components["collision_penalty"] += self.collision_penalty

        if self.lane_departure:
            reward += self.lane_departure_penalty
            self.cumulative_reward_components["lane_departure_penalty"] += self.lane_departure_penalty

        # Add route completion bonus if route is complete
        if components.get("route_complete", False):
            reward += self.route_completed_reward
            self.cumulative_reward_components["route_completion_bonus"] = self.route_completed_reward
        
        return reward

    def step(self, action):
        self.current_step += 1
        # Track cumulative reward components
        reward_components = {
            "target_progress": 0.0,
            "waypoint_progress": 0.0,
            "lane_departure_penalty": 0.0,  # Track actual penalty
            "collision_penalty": 0.0,  # Track actual penalty
            "route_complete": False,  # Track route completion
        }


        # Execute action in CARLA
        if self.vehicle is not None:
            control = carla.VehicleControl()
            control.steer    = float(action[0])
            control.throttle = float((action[1] + 1.0) / 2.0)
            control.brake = float((action[2] + 1.0) / 2.0)
            self.vehicle.apply_control(control)

        # # Visualize waypoints during training
        self.draw_waypoints()

        # Get new observation
        obs = self._get_obs()
        speed, lane_offset, collision_occured, distance_to_target, distance_to_waypoint, target_angle, waypoint_angle = obs
        
        # Update waypoint index and get new distance
        self.updated_waypoints_reached(reward_components)
                    

        # Update progress buffer
        self.progress_buffer.append(distance_to_target)
        # Maintain buffer size
        if len(self.progress_buffer) > self.progress_buffer_size:
            self.progress_buffer.pop(0)
        
        # Progress toward target reward (now uses buffer)
        reward_components["target_progress"] = self._calculate_target_progress_reward(distance_to_target)

        
        # Lane departure detection
        self.lane_departure = False
        if abs(lane_offset) > 2.0:
            self.lane_departure = True
            self.lane_departures += 1
            self.cumulative_reward_components["lane_departure_count"] += 1





        
        # reward_components["lane_departure_penalty"] = lane_departure
        # reward_components["lane_departure_penalty"] += self._calculate_lane_offset_penalty(lane_offset)
        
        reward_components["collision_penalty"] = collision_occured
        
        # Check for episode end conditions
        route_complete = distance_to_target < 1.0
        reward_components["route_complete"] = route_complete
        
        # Calculate total reward
        reward = self._calculate_total_reward(reward_components)

        self.cumulative_reward += reward
        
        stuck = (self.current_step > self.stuck_steps and speed < 1.0) or distance_to_waypoint > 10
        terminated = bool(collision_occured or route_complete or stuck)


        # Create info dictionary with all reward components and other data
        info = {
            "speed": speed,
            
            "lane_offset": lane_offset,
            "lane_departures": self.lane_departures,
            "collision": collision_occured,
            "throttle": float(action[1]),
            "brake": float(action[2]),
            "steer": float(action[0]),
            "distance_to_target": distance_to_target,
            "target_angle": target_angle,
            "route_complete": route_complete,
            "stuck": stuck,
            "total_reward": reward,
            "cumulative_reward": self.cumulative_reward,
            "current_waypoint_idx": self.waypoint_index
        }
        
        # Add all reward components to info dictionary
        for component, value in reward_components.items():
            if component not in ["collision", "lane_departure"]:  # These are already in info
                info[f"{component}_reward"] = value
        
        
        # Display episode information in 3D world when episode ends
        if terminated:
            # Reset the flag first to ensure we can show a summary for this episode
            self.episode_summary_displayed = False
            
            # Show episode text
            self.display_episode_text(info)
            self.draw_waypoints(life_time=1.0, episode_info=info)
            
        # print(f"Action: {action}, Adjusted: {control.steer}, {control.throttle}, {control.brake}, Reward: {reward}")

        return obs, reward, terminated, False, info
    




    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        
        # Reset episode summary flag
        self.episode_summary_displayed = False
        
        # Clean up previous episode
        self.cleanup()
        self.lane_departure = False

        self.collision_occured = False
        self.current_step = 0
        self.sensors = []

        self.most_recent_waypoint_reached = 0
        self.target_location = None
        self.lane_departures = 0

        self.progress_buffer = []
        self.waypoint_index = 0

        self.cumulative_reward = 0.0  # Track cumulative reward for the episode
        
        # Track cumulative reward components
        self.cumulative_reward_components = {
            "target_progress": 0.0,
            "waypoint_progress": 0.0,
            "lane_departure_count": 0,  # Track count separately
            "lane_departure_penalty": 0.0,  # Track actual penalty
            "collision_count": 0,  # Track count separately
            "collision_penalty": 0.0,  # Track actual penalty
            "route_completion_bonus": 0.0,  # Track route completion bonus
        }
        
        # Reset tracking variables
        self.previous_location = None
        self.position_history = []
        self.circle_penalty_applied = False
        
        # Reset waypoint counters
        self.total_waypoints_reached = 0
        self.reached_waypoints = set()
        self.reached_waypoints.add(0)
        
        # Spawn new vehicle
        blueprint_library = self.world.get_blueprint_library()
        vehicle_bp = blueprint_library.filter('vehicle.*')[0]
        
        spawn_point = self.world.get_map().get_spawn_points()[0]
        vehicle_bp.set_attribute('role_name', 'hero')


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
            x=spawn_loc.x + 50.0,
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
        step_size = 2.5  # meters
        distance = self.target_location.x - spawn_loc.x  # assuming +X only
        num_steps = int(distance / step_size)

        for i in range(num_steps + 1):
            x = spawn_loc.x + (i) * step_size
            manual_loc = carla.Location(x=x, y=spawn_loc.y, z=spawn_loc.z)

            wp = self.world.get_map().get_waypoint(
                manual_loc,
                project_to_road=True,
                lane_type=carla.LaneType.Driving
            )

            # print(wp, "\n")

            if wp:  # in case projection fails
                self.route.append(wp)


        # Add sensors
        self._add_collision_sensor()
        # self._add_camera_sensor()
        
        # Get initial observation
        observation = self._get_obs()
        
        # Draw waypoints after reset
        self.draw_waypoints(life_time=1.0)  # Longer lifetime on reset for better visibility
        time.sleep(1.5)
        
        # print("done")
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

    def draw_waypoints(self, life_time=0.1, episode_info=None):
        """
        Draw waypoints in the CARLA world for visualization during training.
        
        Args:
            life_time: How long the visualizations should last in seconds
            episode_info: Optional dictionary with episode information (total_reward, steps, etc.)
                          to display when an episode ends
        """
        try:
            # Draw the reference waypoint
            if hasattr(self, 'reference_waypoint'):
                self.world.debug.draw_point(
                    self.reference_waypoint.transform.location,
                    size=0.2,
                    color=carla.Color(r=0, g=255, b=0),  # green
                    life_time=life_time
                )

            
            # Draw route waypoints: color reached ones yellow, others blue
            if hasattr(self, 'route'):
                for i, wp in enumerate(self.route):
                    if i in self.reached_waypoints:
                        # Reached waypoint
                        self.world.debug.draw_point(
                            wp.transform.location,
                            size=0.2,
                            color=carla.Color(r=255, g=255, b=0),  # yellow for reached
                            life_time=life_time
                        )
                    else:
                        # Unreached waypoint
                        self.world.debug.draw_point(
                            wp.transform.location,
                            size=0.1,
                            color=carla.Color(r=0, g=0, b=255),  # blue for unreached
                            life_time=life_time
                        )
                                    
        except Exception as e:
            print(f"Error drawing waypoints: {e}")

    def display_episode_text(self, episode_info):
        """Display a single episode summary in the CARLA world."""
        # Do nothing if we've already displayed a summary for this episode
        if self.episode_summary_displayed:
            return
        
        if not self.vehicle:
            return
        
        try:
            # Mark that we've displayed the summary
            self.episode_summary_displayed = True
            
            # Get vehicle location and vectors
            location = self.vehicle.get_location()
            forward = self.vehicle.get_transform().get_forward_vector()
            
            # Position text directly above the vehicle
            text_loc = carla.Location(
                x=location.x,
                y=location.y,
                z=location.z + 4.0  # Well above the vehicle
            )
            
            # Format the episode information as a single, comprehensive text block
            episode_text = f"EPISODE SUMMARY - Steps: {self.current_step}\n"
            episode_text += f"Cumulative Reward: {episode_info['cumulative_reward']:.2f}\n"
            episode_text += f"Waypoints Reached: {self.total_waypoints_reached}/{len(self.route)}\n"
            
            # Add cumulative reward components
            episode_text += "\nREWARD COMPONENTS:\n"
            episode_text += f"Target Progress: {self.cumulative_reward_components['target_progress']:.2f}\n"
            episode_text += f"Waypoint Progress: {self.cumulative_reward_components['waypoint_progress']:.2f}\n"
            episode_text += f"Route Completion Bonus: {self.cumulative_reward_components['route_completion_bonus']:.2f}\n"
            # Show counts with penalties
            ld_count = int(self.cumulative_reward_components['lane_departure_count'])
            ld_penalty = self.cumulative_reward_components['lane_departure_penalty']
            episode_text += f"Lane Departures: {ld_count} (Penalty: {ld_penalty:.2f})\n"
            
            coll_count = int(self.cumulative_reward_components['collision_count'])
            coll_penalty = self.cumulative_reward_components['collision_penalty']
            episode_text += f"Collisions: {coll_count} (Penalty: {coll_penalty:.2f})\n"
            
            episode_text += f"Waypoints Reached: {self.total_waypoints_reached}\n"
            
            episode_text += f"Distance to target: {episode_info['distance_to_target']:.2f}\n"
                                    
            # Draw the consolidated text
            self.world.debug.draw_string(
                text_loc,
                episode_text,
                draw_shadow=True,
                color=carla.Color(r=255, g=0, b=0),  # Red for visibility
                life_time=5.0  # Show for 5 seconds (increased from 3)
            )
            
            print(episode_text)
            
        except Exception as e:
            print(f"Error displaying episode text: {e}")


