import gymnasium as gym
import numpy as np
from stable_baselines3 import PPO
import carla
import pygame
import numpy as np
import time

# Import your environment
from PPO import CarlaEnv

def visualize_policy(model_path, num_episodes=5):
    # Initialize pygame for visualization
    pygame.init()
    
    # Create two windows: one for camera view, one for metrics
    camera_display = pygame.display.set_mode((800, 600))
    metrics_display = pygame.display.set_mode((400, 300), flags=pygame.RESIZABLE, display=1)
    
    pygame.display.set_caption("CARLA Camera View", display=0)
    pygame.display.set_caption("CARLA Metrics", display=1)
    
    font = pygame.font.Font(None, 36)

    # Create environment
    env = CarlaEnv()
    
    # Load the trained model
    model = PPO.load(model_path)
    
    for episode in range(num_episodes):
        obs, info = env.reset()
        episode_reward = 0
        step = 0
        
        print(f"Episode {episode + 1}")
        
        while True:
            # Process pygame events
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    pygame.quit()
                    return
            
            # Get action from policy
            action, _ = model.predict(obs, deterministic=True)
            
            # Step environment
            obs, reward, terminated, truncated, info = env.step(action)
            episode_reward += reward
            step += 1
            
            # Get and display camera image
            camera_img = env.get_camera_image()
            if camera_img is not None:
                camera_surface = pygame.surfarray.make_surface(camera_img.swapaxes(0, 1))
                camera_display.blit(camera_surface, (0, 0))
                pygame.display.flip(display=0)
            
            # Display metrics
            metrics_display.fill((0, 0, 0))  # Clear screen
            
            # Render metrics
            metrics = [
                f"Speed: {info['speed']:.2f} km/h",
                f"Lane Offset: {info['lane_offset']:.2f} m",
                f"Angle: {info['angle']:.2f} rad",
                f"Collision: {info['collision']}",
                f"Lane Departures: {info['lane_departures']}",
                f"Step: {step}",
                f"Episode Reward: {episode_reward:.2f}",
                f"Action: [{action[0]:.2f}, {action[1]:.2f}, {action[2]:.2f}]"
            ]
            
            for i, metric in enumerate(metrics):
                text = font.render(metric, True, (255, 255, 255))
                metrics_display.blit(text, (10, 10 + i * 30))
            
            pygame.display.flip(display=1)
            
            if terminated or truncated:
                print(f"Episode finished after {step} steps. Total reward: {episode_reward:.2f}")
                break
            
            time.sleep(0.05)  # Small delay to make visualization smoother
    
    pygame.quit()
    env.close()

if __name__ == "__main__":
    visualize_policy("ppo_carla")  # Path to your saved model 