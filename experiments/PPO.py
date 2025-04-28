import gymnasium as gym
import numpy as np
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv
from stable_baselines3.common.env_checker import check_env
import os

from carla_envs import CarlaEnvVanilla#, AnxiousCarlaEnv
import argparse
from stable_baselines3.common.callbacks import CheckpointCallback
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.callbacks import BaseCallback, CallbackList
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import gym
from stable_baselines3.common.torch_layers import BaseFeaturesExtractor

import torch
import torch.nn as nn
from stable_baselines3.common.torch_layers import BaseFeaturesExtractor
import numpy as np
import gymnasium as gym
from PIL import Image
from datetime import datetime
import time

class CustomCombinedExtractor(BaseFeaturesExtractor):
    def __init__(self, observation_space: gym.spaces.Dict, features_dim: int = 256):
        super().__init__(observation_space, features_dim)
        
        # Correctly interpret image dimensions: (C, H, W)
        image_shape = observation_space["image"].shape
        self.image_channels = image_shape[0]
        self.image_height = image_shape[1]
        self.image_width = image_shape[2]
        
        # CNN for image processing
        self.cnn = nn.Sequential(
            nn.Conv2d(self.image_channels, 16, kernel_size=3, stride=2, padding=1),
            nn.ReLU(),
            nn.Conv2d(16, 32, kernel_size=3, stride=2, padding=1),
            nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=3, stride=2, padding=1),
            nn.ReLU(),
            nn.Flatten(),
        )
        
        # Compute CNN output size
        with torch.no_grad():
            sample_image = torch.zeros(1, self.image_channels, self.image_height, self.image_width)
            n_flatten = self.cnn(sample_image).shape[1]
        
        # Vector input processing
        self.vector_size = sum(
            int(np.prod(space.shape))
            for key, space in observation_space.items()
            if key != "image"
        )
        
        self.vector_mlp = nn.Sequential(
            nn.Linear(self.vector_size, 128),
            nn.ReLU(),
            nn.Linear(128, 128),
            nn.ReLU(),
        )
        
        # Combine features
        self.final_layer = nn.Sequential(
            nn.Linear(n_flatten + 128, features_dim),
            nn.ReLU(),
        )

    def forward(self, observations) -> torch.Tensor:

        # Process image
        image = observations["image"] / 255.0  # Normalize to [0, 1]
        cnn_features = self.cnn(image)

        # Process vector observations
        vector_input = torch.cat([
            observations[key].reshape(observations[key].shape[0], -1)
            for key in observations.keys()
            if key != "image"
        ], dim=1)
        vector_features = self.vector_mlp(vector_input)

        # Combine and output
        combined = torch.cat([cnn_features, vector_features], dim=1)
        return self.final_layer(combined)





def main():
    # Set up argument parser
    parser = argparse.ArgumentParser(description='Train PPO agent in CARLA environment')
    parser.add_argument('--timesteps', type=int, default=500_000,
                       help='Number of timesteps to train (default: 500,000)')
    parser.add_argument('--load_model', type=str, default=None,
                       help='Path to load existing model for further training')
    parser.add_argument('--learning_rate', type=float, default=5e-5,
                       help='Learning rate (default: 0.0003)')
    parser.add_argument('--anxious', action='store_true',
                       help='Use AnxiousCarlaEnv instead of CarlaEnv')
    parser.add_argument('--save_path', type=str, default=None,
                       help='Path to save the trained model')
                       
    # Parse arguments
    args = parser.parse_args()
    
    print(f"Anxious flag is set to: {args.anxious}")
    # Setup training directories
    mode = 'anxious' if args.anxious else 'vanilla'
    model_dir = os.path.join('training', mode, 'models')
    reward_dir = os.path.join('training', mode, 'rewards')
    os.makedirs(model_dir, exist_ok=True)
    os.makedirs(reward_dir, exist_ok=True)

    # Create and wrap environment with Monitor for logging
    def make_env():
        env = CarlaEnvVanilla()
        check_env(env)
        return Monitor(env)
    env = DummyVecEnv([make_env])

    # Define policy kwargs for tanh activation and optimized architecture
    policy_kwargs = dict(
        features_extractor_class=CustomCombinedExtractor,
        features_extractor_kwargs=dict(features_dim=256),
        net_arch=dict(
            pi=[64, 64],  # Larger policy network
            vf=[64, 64]   # Larger value network
        ),
        activation_fn=torch.nn.Tanh,
        ortho_init=True
    )

    if args.load_model and os.path.exists(args.load_model):
        print(f"Loading existing model from {args.load_model}")
        model = PPO.load(args.load_model, env=env, device='cpu')
        # Update learning rate and other parameters
        model.learning_rate = args.learning_rate

    else:
        print("creating new model")
        
        model = PPO(
            "MultiInputPolicy",
            env,
            verbose=1,
            clip_range_vf=0.2,
            vf_coef=0.5,
            learning_rate=5e-4,
            n_steps=1024,          # Increased for better learning
            batch_size=64,
            n_epochs=10,
            gamma=0.98,#0.99            # Standard discount factor
            gae_lambda=0.95,       # GAE lambda parameter
            clip_range=0.2,        # Standard PPO clip range
            ent_coef=0.05,         
            policy_kwargs=policy_kwargs,
            device='cpu'
        )

    # Create checkpoint callback saving to model_dir
    checkpoint_callback = CheckpointCallback(
        save_freq=100_000,
        save_path=model_dir,
        name_prefix="ppo_checkpoint"
    )
    # Custom callback to plot episodic reward progression
    class PlotRewardCallback(BaseCallback):
        def __init__(self, save_freq, save_dir, verbose=0):
            super(PlotRewardCallback, self).__init__(verbose)
            self.save_freq = save_freq
            self.save_dir = save_dir
            self.episode_rewards = []
        def _on_step(self) -> bool:
            # Record episodic rewards from info dicts
            infos = self.locals.get('infos', [])
            for info in infos:
                if 'episode' in info:
                    # 'episode' dict contains 'r' for reward
                    self.episode_rewards.append(info['episode']['r'])
            # On checkpoint interval, save a plot
            if self.num_timesteps % self.save_freq == 0 and len(self.episode_rewards) > 0:
                # Plot episodic rewards
                fig, ax = plt.subplots()
                ax.plot(self.episode_rewards)
                ax.set_xlabel('Episode')
                ax.set_ylabel('Reward')
                ax.set_title('Episodic Reward Progression')
                os.makedirs(self.save_dir, exist_ok=True)
                fig_path = os.path.join(self.save_dir, f"reward_progress_{self.num_timesteps}.png")
                fig.savefig(fig_path)
                plt.close(fig)
            return True
        

    plot_callback = PlotRewardCallback(save_freq=50_000, save_dir=reward_dir)
    # Create the image saving callback


    # Combine callbacks
    callback_list = CallbackList([checkpoint_callback, plot_callback])

    print("starting training")
    # Verify observation pipeline

    # Train the model with checkpoint and plotting callbacks
    model.learn(total_timesteps=args.timesteps, callback=callback_list)


    print("Done trianing")
    # Save the final model to model_dir
    final_path = os.path.join(model_dir, "ppo_final")
    model.save(final_path)
    print(f"Model saved to {final_path}")

    # Close the environment
    env.close()

if __name__ == "__main__":
    main()