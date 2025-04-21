import gymnasium as gym
import numpy as np
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv
from stable_baselines3.common.env_checker import check_env
from stable_baselines3.common.callbacks import BaseCallback
import torch
import os

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
        self.best_reward = float('-inf')
        self.best_distance = float('inf')
        self.best_alignment = float('inf')  # Best alignment with target (lower is better)
        self.total_progress = 0  # Track total progress toward target
        self.last_info = None  # Store the last info for end-of-episode reporting
        self.episode_steps = 0  # Count steps in current episode

    def _on_step(self):
        # Get current info
        info = self.locals['infos'][0]
        raw_action = self.locals['actions'][0]
        reward = self.locals['rewards'][0]
        
        # Store the last info for end-of-episode reporting
        self.last_info = info
        self.episode_steps += 1
        
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
        
        # Track best distance and alignment
        if 'distance_to_target' in info:
            self.best_distance = min(self.best_distance, info['distance_to_target'])
        if 'target_angle' in info:
            self.best_alignment = min(self.best_alignment, abs(info['target_angle']))
        
        # Track progress
        if 'progress' in info:
            self.total_progress += info['progress']

        # If episode ended
        if self.locals['dones'][0]:
            self.episode_rewards.append(self.current_episode_reward)
            self.episode_count += 1
            
            # Track best reward
            if self.current_episode_reward > self.best_reward:
                self.best_reward = self.current_episode_reward
            
            # Print episode summary with key metrics
            print(f"\n{'='*50}")
            print(f"Episode {self.episode_count} Summary (Steps: {self.episode_steps})")
            print(f"{'='*50}")
            print(f"Total reward: {self.current_episode_reward:.2f} (Best: {self.best_reward:.2f})")
            print(f"Average episode reward: {np.mean(self.episode_rewards):.2f}")
            
            # Movement metrics
            print(f"\nMovement Metrics:")
            print(f"  Max speed achieved: {self.max_speed_seen:.2f} km/h")
            print(f"  Current speed: {info['speed']:.2f} km/h")
            print(f"  Steps without movement: {self.steps_without_movement}")
            
            # Target metrics
            print(f"\nTarget Metrics:")
            print(f"  Starting distance: {self.locals['env'].envs[0].reset()[0][4]:.2f}m")
            print(f"  Final distance: {info['distance_to_target']:.2f}m")
            print(f"  Best distance: {self.best_distance:.2f}m")
            print(f"  Final target angle: {np.degrees(info['target_angle']):.1f}°")
            print(f"  Best alignment: {np.degrees(self.best_alignment):.1f}°")
            print(f"  Total progress made: {self.total_progress:.2f}m")
            
            # Last action
            print(f"\nFinal Action:")
            print(f"  Steer: {clipped_action[0]:.3f}")
            print(f"  Throttle: {clipped_action[1]:.3f}")
            print(f"  Brake: {clipped_action[2]:.3f}")
            
            # Last rewards
            print(f"\nReward Components:")
            print(f"  Progress reward: {info.get('progress_reward', 0):.2f}")
            print(f"  Speed reward: {info.get('speed_reward', 0):.2f}")
            print(f"  Direction reward: {info.get('direction_reward', 0):.2f}")
            print(f"  Movement penalty: {info.get('movement_penalty', 0):.2f}")
            print(f"  Collision penalty: {info.get('collision_penalty', 0):.2f}")
            print(f"  Lane penalty: {info.get('lane_departure_penalty', 0):.2f}")
            
            # Completion status
            termination_reason = "Unknown"
            if info.get('collision', False):
                termination_reason = "Collision"
            elif info.get('route_complete', False):
                termination_reason = "Route Complete"
            elif self.steps_without_movement > 500:
                termination_reason = "Vehicle Stuck"
            elif self.episode_steps >= 2000:  # Hardcode max_steps instead of trying to access it
                termination_reason = "Max Steps Reached"
                
            print(f"\nTermination reason: {termination_reason}")
            print(f"{'='*50}\n")
            
            # Reset episode-specific metrics
            self.current_episode_reward = 0
            self.max_speed_seen = 0
            self.steps_without_movement = 0
            self.best_distance = float('inf')
            self.best_alignment = float('inf')
            self.total_progress = 0
            self.episode_steps = 0

        return True

def main():
    # Set up argument parser
    parser = argparse.ArgumentParser(description='Train PPO agent in CARLA environment')
    parser.add_argument('--timesteps', type=int, default=500_000,
                       help='Number of timesteps to train (default: 500,000)')
    parser.add_argument('--load_model', type=str, default=None,
                       help='Path to load existing model for further training')
    parser.add_argument('--save_path', type=str, default="ppo_carla",
                       help='Path to save the trained model')
    parser.add_argument('--learning_rate', type=float, default=3e-4,
                       help='Learning rate (default: 0.0003)')
    
    # Parse arguments
    args = parser.parse_args()

    # Create and wrap environment
    env = CarlaEnv()
    check_env(env)  # Check if environment follows gym interface
    env = DummyVecEnv([lambda: env])

    # Define policy kwargs for tanh activation and optimized architecture
    policy_kwargs = dict(
        activation_fn=torch.nn.Tanh,
        net_arch=dict(
            pi=[256, 256],  # Larger policy network
            vf=[256, 256]   # Larger value network
        ),
        ortho_init=True
    )

    if args.load_model and os.path.exists(args.load_model):
        print(f"Loading existing model from {args.load_model}")
        model = PPO.load(args.load_model, env=env)
        # Update learning rate and other parameters
        model.learning_rate = args.learning_rate
    else:
        print("Creating new model with optimized hyperparameters")
        model = PPO(
            "MlpPolicy",
            env,
            verbose=1,
            learning_rate=args.learning_rate,
            n_steps=2048,          # Increased for better learning
            batch_size=64,
            n_epochs=10,
            gamma=0.99,            # Standard discount factor
            gae_lambda=0.95,       # GAE lambda parameter
            clip_range=0.2,        # Standard PPO clip range
            ent_coef=0.01,         # Reduced for more exploitation
            policy_kwargs=policy_kwargs,
            device='cpu'
        )

    # Create callback
    debug_callback = DebugCallback()

    # Train the model with debug callback
    model.learn(total_timesteps=args.timesteps, callback=debug_callback)

    # Save the model
    model.save(args.save_path)
    print(f"Model saved to {args.save_path}")

    # Close the environment
    env.close()

if __name__ == "__main__":
    main()