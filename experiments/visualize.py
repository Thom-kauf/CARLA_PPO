import gymnasium as gym
import numpy as np
import carla
import os
from stable_baselines3 import PPO
import matplotlib.pyplot as plt
import argparse
from carla_envs import CarlaEnvVanilla

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

    # Create environment
    try:
        print("Attempting to create CARLA environment...")
        parser = argparse.ArgumentParser()
        parser.add_argument('--model_path', type=str, required=True,
                          help='Name of the model file (e.g., ppo_checkpoint_1400000_steps)')
        parser.add_argument('--subfolder', type=str, required=True,
                        help='Subfolder within ./visualizations to save outputs')
        parser.add_argument('--num_episodes', type=int, default=10,
                        help='Number of episodes to run')
        parser.add_argument('--mode', type=str, required=True)
        args = parser.parse_args()
        env = CarlaEnvVanilla()
        print("✓ Environment created")
    except Exception as e:
        print(f"✗ Error creating environment: {e}")
        return
    
    # Load the trained model
    try:
        print("Attempting to load trained model...")
        # Define the models directory path
        script_dir = os.path.dirname(os.path.abspath(__file__))
        
        if args.mode == "vanilla":
            models_dir = os.path.join(script_dir, "training", "vanilla", "models")
        elif args.mode == "anxious":
            models_dir = os.path.join(script_dir, "training", "anxious", "models")
        
        # Construct full model path
        model_name = args.model_path
        if not model_name.endswith('.zip'):
            model_name += '.zip'
        model_path = os.path.join(models_dir, model_name)
        
        if not os.path.exists(model_path):
            print(f"Model file not found at: {model_path}")
            print("\nAvailable models:")
            try:
                for file in os.listdir(models_dir):
                    if file.endswith('.zip'):
                        print(f"  {file[:-4]}")  # Print without .zip extension
            except Exception as e:
                print(f"Could not list models: {e}")
            return
            
        model = PPO.load(model_path, env=env, device='cpu')  # Explicitly set device to CPU
        print(f"✓ Model loaded successfully from: {model_path}")
    except Exception as e:
        print(f"✗ Error loading model: {e}")
        print("Exiting visualization - trained model required")
        env.close()
        return

    try:
        print("Resetting environment...")
        obs, info = env.reset()
        print("✓ Environment reset")
    except Exception as e:
        print(f"✗ Error resetting environment: {e}")
        env.close()
        return

    running = True
    print("\n=== Starting Main Loop ===")
    
    # Track trial information
    current_trial = 0
    num_trials = args.num_episodes
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


    if args.mode == "vanilla":
        output_dir = os.path.join("visualizations", "vanilla")
    elif args.mode == "anxious":
        output_dir = os.path.join("visualizations", "anxious")
    
    # Construct full model path
    model_name = args.model_path
    if not model_name.endswith('.zip'):
        model_name += '.zip'
    model_path = os.path.join(models_dir, model_name)


    # Create visualizations directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)

    while running and current_trial < num_trials:
        try:
            # Get action from trained model
            action, _ = model.predict(obs, deterministic=True)
            # Step environment
            obs, reward, terminated, truncated, info = env.step(action)

            # Track metrics
            total_episode_reward += reward
            total_episode_progress += info.get('target_progress', 0)  # Updated to match environment
            episode_speeds.append(info.get('speed', 0))
            steps_in_current_trial += 1
            


            # Draw waypoints
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
                
                # Draw intermediate route waypoints
                if hasattr(env, 'route'):
                    for i, wp in enumerate(env.route):
                        # Color reached waypoints yellow, unreached ones blue
                        color = carla.Color(r=255, g=255, b=0) if i in env.reached_waypoints else carla.Color(r=0, g=0, b=255)
                        size = 0.2 if i in env.reached_waypoints else 0.1
                        world.debug.draw_point(
                            wp.transform.location,
                            size=size,
                            color=color,
                            life_time=0.1
                        )

            except Exception as e:
                print(f"Error drawing waypoints: {e}")

            # Check if we should start a new trial
            if terminated or truncated or (steps_in_current_trial >= max_steps_per_trial):
                print(f"\nTrial {current_trial + 1} ended. Steps: {steps_in_current_trial}")
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
    env.close()
    print(f"Visualization ended after {current_trial} trials")

if __name__ == "__main__":
    visualize() 