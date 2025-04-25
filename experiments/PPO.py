import gymnasium as gym
import numpy as np
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv
from stable_baselines3.common.env_checker import check_env
import torch
import os

from carla_envs import CarlaEnvVanilla, AnxiousCarlaEnv
import argparse



def main():
    # Set up argument parser
    parser = argparse.ArgumentParser(description='Train PPO agent in CARLA environment')
    parser.add_argument('--timesteps', type=int, default=500_000,
                       help='Number of timesteps to train (default: 500,000)')
    parser.add_argument('--load_model', type=str, default=None,
                       help='Path to load existing model for further training')
    parser.add_argument('--save_path', type=str, default="ppo_carla",
                       help='Path to save the trained model')
    parser.add_argument('--learning_rate', type=float, default=1e-4,
                       help='Learning rate (default: 0.0003)')
    parser.add_argument('--anxious', action='store_true',
                       help='Use AnxiousCarlaEnv instead of CarlaEnv')
    
    # Parse arguments
    args = parser.parse_args()

    print(f"Anxious flag is set to: {args.anxious}")

    # Create and wrap environment
    env_class = AnxiousCarlaEnv if args.anxious else CarlaEnvVanilla
    print(f"Using environment class: {env_class.__name__}")
    env = env_class()
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
        model = PPO.load(args.load_model, env=env, device='cpu')
        # Update learning rate and other parameters
        model.learning_rate = args.learning_rate

    else:
        print("creating new model")
        model = PPO(
            "MlpPolicy",
            env,
            verbose=1,
            learning_rate=args.learning_rate,
            n_steps=4096,          # Increased for better learning
            batch_size=256,
            n_epochs=20,
            gamma=0.97,            # Standard discount factor
            gae_lambda=0.95,       # GAE lambda parameter
            clip_range=0.2,        # Standard PPO clip range
            ent_coef=0.075,         
            policy_kwargs=policy_kwargs,
            device='cpu'
        )

    # Create callback
    # debug_callback = DebugCallback()

    print("starting training")
    # Train the model with debug callback
    model.learn(total_timesteps=args.timesteps)#, callback=debug_callback)


    print("Done trianing")
    # Save the model
    model.save(args.save_path)
    print(f"Model saved to {args.save_path}")

    # Close the environment
    env.close()

if __name__ == "__main__":
    main()