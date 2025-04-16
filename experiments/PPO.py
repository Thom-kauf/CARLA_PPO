import gymnasium as gym
import numpy as np
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv
from stable_baselines3.common.env_checker import check_env
from stable_baselines3.common.callbacks import BaseCallback
import torch

# Import environment from carla_env.py
from carla_env import CarlaEnv
import argparse

# Custom callback for debugging
class DebugCallback(BaseCallback):
    def __init__(self, verbose=0):
        super().__init__(verbose)
        self.episode_count = 0
        self.episode_rewards = []
        self.current_episode_reward = 0
        self.max_speed_seen = 0
        self.steps_without_movement = 0

    def _on_step(self):
        # Get current info
        info = self.locals['infos'][0]
        raw_action = self.locals['actions'][0]
        reward = self.locals['rewards'][0]
        
        # Get clipped action from the environment
        clipped_action = np.clip(raw_action, self.locals['env'].envs[0].action_space.low, 
                               self.locals['env'].envs[0].action_space.high)
        
        self.current_episode_reward += reward
        
        # Track if car is moving
        if info['speed'] > 0.1:  # If speed is above 0.1 km/h
            self.steps_without_movement = 0
        else:
            self.steps_without_movement += 1
            
        # Update max speed
        self.max_speed_seen = max(self.max_speed_seen, info['speed'])

        # Print debug info every 100 steps
        if self.n_calls % 500 == 0:
            print(f"\nStep {self.n_calls}:")
            print(f"Raw Action: Steer={raw_action[0]:.3f}, Throttle={raw_action[1]:.3f}, Brake={raw_action[2]:.3f}")
            print(f"Clipped Action: Steer={clipped_action[0]:.3f}, Throttle={clipped_action[1]:.3f}, Brake={clipped_action[2]:.3f}")
            print(f"Speed: {info['speed']:.2f} km/h (Max seen: {self.max_speed_seen:.2f})")
            print(f"Reward: {reward:.2f}")
            print(f"Steps without movement: {self.steps_without_movement}")

        # If episode ended
        if self.locals['dones'][0]:
            self.episode_rewards.append(self.current_episode_reward)
            self.episode_count += 1
            print(f"\nEpisode {self.episode_count} ended:")
            print(f"Total reward: {self.current_episode_reward:.2f}")
            print(f"Max speed achieved: {self.max_speed_seen:.2f} km/h")
            print(f"Average episode reward: {np.mean(self.episode_rewards):.2f}")
            self.current_episode_reward = 0
            self.max_speed_seen = 0
            self.steps_without_movement = 0

        return True

# Set up argument parser
parser = argparse.ArgumentParser(description='Train PPO agent in CARLA environment')
parser.add_argument('--timesteps', type=int, default=500_000,
                   help='Number of timesteps to train (default: 500,000)')

# Parse argument for number of timesteps
args = parser.parse_args()

# Create and wrap environment
env = CarlaEnv()

# Check if environment follows gym interface
check_env(env)

# Create vectorized environment
env = DummyVecEnv([lambda: env])

# Initialize PPO model
model = PPO(
    "MlpPolicy",
    env,
    verbose=1,
    learning_rate=0.001,
    n_steps=1024,
    batch_size=64,
    n_epochs=10,
    gamma=0.99,
    gae_lambda=0.95,
    clip_range=0.2,
    ent_coef=0.03,
    device='cpu'
)


# Create callback
debug_callback = DebugCallback()

# Train the model with debug callback
model.learn(total_timesteps=args.timesteps, callback=debug_callback)

# Save the model
model.save("ppo_carla")

# Close the environment
env.close()