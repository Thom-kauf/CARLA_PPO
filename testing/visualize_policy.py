import gymnasium as gym
import numpy as np
import carla
import pygame
import time
import os

# Import your environment
from PPO import CarlaEnv

def visualize_policy(model_path=None, num_episodes=2):
    # Initialize pygame for visualization
    pygame.init()
    
    # Create a single window with space for both camera and metrics
    WINDOW_WIDTH = 1200  # 800 for camera + 400 for metrics
    WINDOW_HEIGHT = 600
    display = pygame.display.set_mode((WINDOW_WIDTH, WINDOW_HEIGHT))
    pygame.display.set_caption("CARLA Visualization")
    
    font = pygame.font.Font(None, 36)

    # Create environment
    env = CarlaEnv()
    
    for episode in range(num_episodes):
        obs, info = env.reset()
        episode_reward = 0
        step = 0
        
        print(f"Episode {episode + 1}")
        
        # Wait a bit for the camera to start receiving images
        time.sleep(2)
        
        while True:
            # Process pygame events
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    pygame.quit()
                    env.close()
                    return
            
            # Just use random actions for testing visualization
            action = env.action_space.sample()
            
            # Step environment
            obs, reward, terminated, truncated, info = env.step(action)
            episode_reward += reward
            step += 1
            
            # Clear the display
            display.fill((0, 0, 0))
            
            # Get and display camera image
            camera_img = env.get_camera_image()
            if camera_img is not None:
                # Convert to pygame surface and display
                camera_img = np.swapaxes(camera_img, 0, 1)
                pygame_surface = pygame.surfarray.make_surface(camera_img)
                display.blit(pygame_surface, (0, 0))  # Camera view on the left
            
            # Render metrics (on the right side)
            metrics = [
                f"Speed: {info['speed']:.2f} km/h",
                f"Lane Offset: {info['lane_offset']:.2f} m",
                f"Angle: {info['angle']:.2f} rad",
                f"Collision: {info['collision']}",
                f"Step: {step}",
                f"Action: [{action[0]:.2f}, {action[1]:.2f}, {action[2]:.2f}]"
            ]
            
            for i, metric in enumerate(metrics):
                text = font.render(metric, True, (255, 255, 255))
                display.blit(text, (820, 10 + i * 30))  # Metrics start at x=820
            
            pygame.display.flip()
            
            if terminated or truncated:
                print(f"Episode finished after {step} steps. Total reward: {episode_reward:.2f}")
                time.sleep(1)  # Pause briefly at episode end
                break
            
            time.sleep(0.1)  # Small delay to make visualization smoother
    
    pygame.quit()
    env.close()

if __name__ == "__main__":
    visualize_policy() 