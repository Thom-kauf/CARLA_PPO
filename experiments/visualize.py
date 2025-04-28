import gymnasium as gym
import numpy as np
import carla
import pygame
import os
from stable_baselines3 import PPO
import matplotlib.pyplot as plt
import argparse
import cv2
from carla_envs import CarlaEnvVanilla#, AnxiousCarlaEnv

print("=== CARLA Visualization Script ===")
print("Checking if CARLA is running...")

# Try to connect to CARLA first
try:
    client = carla.Client('localhost', 2000)
    client.set_timeout(10.0)
    world = client.get_world()
    print("✓ Successfully connected to CARLA server")
except Exception as e:
    print(f"✗ Failed to connect to CARLA server: {e}")
    print("Please make sure CARLA is running before starting this script")
    exit(1)


def generate_visualizations(collision_list, lane_departure_list, completion_rate_list, distance_list, reward_list, progress_list, speed_list, filepath):

    # Create a figure for the reward breakdown
    plt.figure(figsize=(10, 5))
    plt.plot(reward_list)
    plt.title('Total Reward per Episode')
    plt.xlabel('Episode')
    plt.ylabel('Reward')
    plt.savefig(filepath + 'reward_plot.png')

    plt.figure(figsize=(10, 5))
    plt.plot(distance_list)
    plt.title('Final Distance to Target')
    plt.xlabel('Episode')
    plt.ylabel('Distance (m)')
    plt.savefig(filepath + 'distance_plot.png')

    plt.figure(figsize=(10, 5))
    plt.plot(progress_list)
    plt.title('Cumulative Progress per Episode')
    plt.xlabel('Episode')
    plt.ylabel('Total Progress (m)')
    plt.grid(True)
    plt.savefig(filepath + 'progress_plot.png')
    
    plt.figure(figsize=(10, 5))
    plt.plot(speed_list)
    plt.title('Average Speed per Episode')
    plt.xlabel('Episode')
    plt.ylabel('Speed (km/h)')
    plt.grid(True)
    plt.savefig(filepath + 'speed_plot.png')

    plt.figure(figsize=(10, 5))
    plt.plot(lane_departure_list)
    plt.title('Lane Departures per Episode')
    plt.xlabel('Episode')
    plt.ylabel('Count')
    plt.savefig(filepath + 'lane_departure_plot.png')

    plt.figure(figsize=(10, 5))
    plt.bar(range(len(collision_list)), [1 if c else 0 for c in collision_list])
    plt.title('Collisions per Episode')
    plt.xlabel('Episode')
    plt.ylabel('Collision (1=Yes, 0=No)')
    plt.yticks([0, 1])
    plt.savefig(filepath + 'collision_plot.png')

    plt.figure(figsize=(10, 5))
    plt.bar(range(len(completion_rate_list)), [1 if c else 0 for c in completion_rate_list])
    plt.title('Route Completion per Episode')
    plt.xlabel('Episode')
    plt.ylabel('Completed (1=Yes, 0=No)')
    plt.yticks([0, 1])
    plt.savefig(filepath + 'completion_rate_plot.png')
    
    # Summary statistics
    completion_percentage = sum([1 if c else 0 for c in completion_rate_list]) / len(completion_rate_list) * 100
    collision_percentage = sum([1 if c else 0 for c in collision_list]) / len(collision_list) * 100
    avg_distance = sum(distance_list) / len(distance_list)
    avg_reward = sum(reward_list) / len(reward_list)
    
    # Save summary to file
    with open(filepath + 'summary.txt', 'w') as f:
        f.write(f"Summary Statistics:\n")
        f.write(f"Total Episodes: {len(reward_list)}\n")
        f.write(f"Route Completion Rate: {completion_percentage:.1f}%\n")
        f.write(f"Collision Rate: {collision_percentage:.1f}%\n")
        f.write(f"Average Final Distance to Target: {avg_distance:.2f}m\n")
        f.write(f"Average Episode Reward: {avg_reward:.2f}\n")
        f.write(f"Average Lane Departures: {sum(lane_departure_list)/len(lane_departure_list):.2f}\n")




def visualize():
    print("\n=== Starting Visualization ===")
    
    # Initialize pygame for visualization
    try:
        pygame.init()
        print("✓ Pygame initialized")
    except Exception as e:
        print(f"✗ Error initializing pygame: {e}")
        return
    
    # Create a single window with space for both camera and metrics
    try:
        WINDOW_WIDTH = 1200  # 800 for camera + 400 for metrics
        WINDOW_HEIGHT = 600
        display = pygame.display.set_mode((WINDOW_WIDTH, WINDOW_HEIGHT))
        pygame.display.set_caption("CARLA Visualization - Trained Model")
        print(f"✓ Window created: {WINDOW_WIDTH}x{WINDOW_HEIGHT}")
    except Exception as e:
        print(f"✗ Error creating window: {e}")
        return

    font = pygame.font.Font(None, 36)
    print("✓ Font loaded")

    # Create environment
    try:
        print("Attempting to create CARLA environment...")
        parser = argparse.ArgumentParser()
        parser.add_argument('--anxious', action='store_true',
                            help='Use AnxiousCarlaEnv instead of CarlaEnv')
        parser.add_argument('--model_path', type=str, default="ppo_carla",
                          help='Path to load the trained model (without .zip extension)')
        parser.add_argument('--record', action='store_true',
                          help='Record the visualization to a video file')
        parser.add_argument('--subfolder', type=str, required=True,
                        help='Subfolder within ./visualizations to save outputs')
        parser.add_argument('--video_name', type=str, default='visualization.avi',
                        help='Name of the video file to save')
        args = parser.parse_args()
        env_class = AnxiousCarlaEnv if args.anxious else CarlaEnvVanilla
        env = env_class()
        print("✓ Environment created")
    except Exception as e:
        print(f"✗ Error creating environment: {e}")
        pygame.quit()
        return
    
    # Load the trained model
    try:
        print("Attempting to load trained model...")
        model = PPO.load(args.model_path, env=env, device='cpu')  # Explicitly set device to CPU
        print("✓ Model loaded successfully")
    except Exception as e:
        print(f"✗ Error loading model: {e}")
        print("Exiting visualization - trained model required")
        pygame.quit()
        env.close()
        return

    try:
        print("Resetting environment...")
        obs, info = env.reset()
        print("✓ Environment reset")
    except Exception as e:
        print(f"✗ Error resetting environment: {e}")
        pygame.quit()
        env.close()
        return

    running = True
    print("\n=== Starting Main Loop ===")
    
    # Track trial information
    current_trial = 0
    num_trials = 10
    steps_in_current_trial = 0
    max_steps_per_trial = 10_000
    
    
    collision_list = []
    lane_departure_list = []
    completion_rate_list = []
    distance_list = []
    reward_list = []
    progress_list = []
    speed_list = []

    total_episode_reward = 0
    total_episode_progress = 0
    episode_speeds = []

    # Construct the full output directory path
    output_dir = os.path.join('.\\visualizations', args.subfolder)

    # Set up video recording if --record is passed
    if args.record:
        video_file_path = os.path.join(output_dir, args.video_name)
        fourcc = cv2.VideoWriter_fourcc(*'XVID')
        video_out = cv2.VideoWriter(video_file_path, fourcc, 30.0, (WINDOW_WIDTH, WINDOW_HEIGHT))

    # Create visualizations directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)

    while running and current_trial < num_trials:
        try:
            # Process pygame events
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    running = False
                    print("Quit event received")

            # Get action from trained model
            action, _ = model.predict(obs, deterministic=True)
            # Step environment
            obs, reward, terminated, truncated, info = env.step(action)

            # Track metrics
            total_episode_reward += reward
            total_episode_progress += info.get('immediate_progress', 0)
            episode_speeds.append(info.get('speed', 0))
            steps_in_current_trial += 1
            
            # Clear the display
            display.fill((0, 0, 0))
            
            # Get and display camera image
            camera_img = env.get_camera_image()
            if camera_img is not None:
                try:
                    # Convert to pygame surface and display
                    camera_img = np.swapaxes(camera_img, 0, 1)
                    pygame_surface = pygame.surfarray.make_surface(camera_img)
                    display.blit(pygame_surface, (0, 0))  # Camera view on the left
                except Exception as e:
                    print(f"Error displaying camera image: {e}")
            
            # Render metrics (on the right side)
            metrics = [
                f"Trial: {current_trial + 1}/{num_trials}",
                f"Steps: {steps_in_current_trial}",
                f"Speed: {info['speed']:.2f} km/h",
                f"Lane Offset: {info['lane_offset']:.2f} m",
                f"Distance to Target: {info['distance_to_target']:.2f} m",
                # f"Immediate Progress: {info.get('immediate_progress', 0):.3f} m",
                # f"Average Progress: {info.get('average_progress', 0):.3f} m",
                # f"Total Buffer Progress: {info.get('total_buffer_progress', 0):.3f} m",
                # f"Total Progress: {100 - :.3f} m",
                f"Target Angle: {np.degrees(info['target_angle']):.1f}°",
                f"Waypoint Reached: {info['waypoint_reached']}",
                f"Distance to Waypoint: {info['distance to waypoint']:.2f} m",
                f"Throttle: {info['throttle']:.2f}",

                f"Brake: {round(info['brake'] if info['throttle'] < 0.1 else 0.0, 2)}",
                f"Steering: {info['steer']:.2f}",
                f"Total Episode Reward: {total_episode_reward:.2f}"
            ]
            
            for i, metric in enumerate(metrics):
                text = font.render(metric, True, (255, 255, 255))
                display.blit(text, (820, 10 + i * 30))  # Metrics start at x=820
            





            # Draw waypoints if available
            try:
                # Draw the reference waypoint
                if hasattr(env, 'reference_waypoint'):
                    world.debug.draw_point(
                        env.reference_waypoint.transform.location,
                        size=0.2,
                        color=carla.Color(r=0, g=255, b=0),  # green
                        life_time=0.1
                    )

                # Draw the target waypoint
                if hasattr(env, 'target_waypoint'):
                    world.debug.draw_point(
                        env.target_waypoint.transform.location,
                        size=0.3,
                        color=carla.Color(r=255, g=0, b=0),  # red
                        life_time=0.1
                    )
                
                # Draw intermediate route waypoints if available
                if hasattr(env, 'route'):
                    for wp in env.route:
                        world.debug.draw_point(
                            wp.transform.location,
                            size=0.1,
                            color=carla.Color(r=0, g=0, b=255),  # blue
                            life_time=0.1
                        )

            except Exception as e:
                print(f"Error drawing waypoints: {e}")




































            pygame.display.flip()

            # Capture the frame from the Pygame window if recording
            if args.record:
                frame = np.array(pygame.surfarray.array3d(display))
                frame = np.transpose(frame, (1, 0, 2))  # Transpose to match OpenCV format
                video_out.write(frame)

            # Check if we should start a new trial
            if terminated or truncated or (steps_in_current_trial >= max_steps_per_trial):
                print(f"Trial {current_trial + 1} ended. Steps: {steps_in_current_trial}")
                print(f"  Final distance: {info['distance_to_target']:.2f}m")
                print(f"  Total progress: {total_episode_progress:.2f}m")
                print(f"  Total reward: {total_episode_reward:.2f}")
                
                # append to lists 
                collision_list.append(info.get('collision', False))
                lane_departure_list.append(info.get('lane_departures', 0))
                completion_rate_list.append(info.get('route_complete', False))
                
                distance_list.append(info.get('distance_to_target', 0))
                reward_list.append(total_episode_reward)
                progress_list.append(total_episode_progress)
                speed_list.append(sum(episode_speeds) / len(episode_speeds) if episode_speeds else 0)
                
                # Reset for next episode
                current_trial += 1
                steps_in_current_trial = 0
                total_episode_reward = 0
                total_episode_progress = 0
                episode_speeds = []
                obs, info = env.reset()

        except Exception as e:
            print(f"Error in main loop: {e}")
            running = False

    # Release the video writer if recording
    if args.record:
        video_out.release()
        print(f"Video saved to: {video_file_path}")

    # Generate visualizations with proper path joining
    generate_visualizations(
        collision_list, 
        lane_departure_list, 
        completion_rate_list, 
        distance_list, 
        reward_list, 
        progress_list, 
        speed_list, 
        os.path.join(output_dir, '')
    )

    print("Visualizations saved to:", output_dir)

    print("\n=== Cleaning Up ===")
    pygame.quit()
    env.close()
    print(f"Visualization ended after {current_trial} trials")

if __name__ == "__main__":
    visualize() 