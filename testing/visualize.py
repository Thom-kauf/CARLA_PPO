import gymnasium as gym
import numpy as np
import carla
import pygame
import time
import os
from stable_baselines3 import PPO

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

# Import environment from the new file
from carla_env import CarlaEnv

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
        env = CarlaEnv()
        print("✓ Environment created")
    except Exception as e:
        print(f"✗ Error creating environment: {e}")
        pygame.quit()
        return
    
    # Load the trained model
    try:
        print("Attempting to load trained model...")
        model = PPO.load("ppo_carla")
        print("✓ Model loaded")
    except Exception as e:
        print(f"✗ Error loading model: {e}")
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
    
    while running:
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
                f"Speed: {info['speed']:.2f} km/h",
                f"Lane Offset: {info['lane_offset']:.2f} m",
                f"Angle: {info['angle']:.2f} rad",
                f"Collision: {info['collision']}",
                f"Action: [{action[0]:.2f}, {action[1]:.2f}, {action[2]:.2f}]",
                f"Reward: {reward:.2f}"
            ]
            
            for i, metric in enumerate(metrics):
                text = font.render(metric, True, (255, 255, 255))
                display.blit(text, (820, 10 + i * 30))  # Metrics start at x=820
            
            pygame.display.flip()
            
            if terminated or truncated:
                print("Episode ended, resetting...")
                obs, info = env.reset()
            
            time.sleep(0.1)  # Small delay to make visualization smoother
            
        except Exception as e:
            print(f"Error in main loop: {e}")
            running = False
    
    print("\n=== Cleaning Up ===")
    pygame.quit()
    env.close()
    print("Visualization ended")

if __name__ == "__main__":
    visualize() 