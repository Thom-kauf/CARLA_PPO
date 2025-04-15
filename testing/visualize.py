import gymnasium as gym
import numpy as np
import carla
import pygame
import time
import os
from stable_baselines3 import PPO

# Import your environment
from PPO import CarlaEnv

def visualize():
    print("Starting visualization...")
    
    # Initialize pygame for visualization
    try:
        pygame.init()
        print("Pygame initialized successfully")
    except Exception as e:
        print(f"Error initializing pygame: {e}")
        return
    
    # Create a single window with space for both camera and metrics
    try:
        WINDOW_WIDTH = 1200  # 800 for camera + 400 for metrics
        WINDOW_HEIGHT = 600
        display = pygame.display.set_mode((WINDOW_WIDTH, WINDOW_HEIGHT))
        pygame.display.set_caption("CARLA Visualization")
        print(f"Window created: {WINDOW_WIDTH}x{WINDOW_HEIGHT}")
    except Exception as e:
        print(f"Error creating window: {e}")
        return

    font = pygame.font.Font(None, 36)
    print("Font loaded")

    # Create environment
    try:
        env = CarlaEnv()
        print("Environment created successfully")
    except Exception as e:
        print(f"Error creating environment: {e}")
        pygame.quit()
        return
    
    # Load the trained model if it exists
    model_path = "ppo_carla"
    if os.path.exists(model_path + ".zip"):
        try:
            model = PPO.load(model_path)
            print("Model loaded successfully")
        except Exception as e:
            print(f"Error loading model: {e}")
            model = None
    else:
        model = None
        print("No trained model found, using random actions")

    try:
        obs, info = env.reset()
        print("Environment reset successfully")
    except Exception as e:
        print(f"Error resetting environment: {e}")
        pygame.quit()
        env.close()
        return

    running = True
    print("Starting main loop...")
    
    while running:
        try:
            # Process pygame events
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    running = False
                    print("Quit event received")
            
            # Get action from model or random
            if model is not None:
                action, _ = model.predict(obs, deterministic=True)
            else:
                action = env.action_space.sample()
            
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
                f"Action: [{action[0]:.2f}, {action[1]:.2f}, {action[2]:.2f}]"
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
    
    print("Cleaning up...")
    pygame.quit()
    env.close()
    print("Visualization ended")

if __name__ == "__main__":
    print("Starting visualization script...")
    visualize() 