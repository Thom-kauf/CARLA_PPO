import gymnasium as gym
import numpy as np
from stable_baselines3 import PPO, SAC
from stable_baselines3.common.vec_env import DummyVecEnv
from stable_baselines3.common.env_checker import check_env
import torch
import os
import argparse
from stable_baselines3.common.callbacks import CheckpointCallback, BaseCallback, CallbackList
from stable_baselines3.common.monitor import Monitor
import matplotlib.pyplot as plt

from carla_envs import CarlaEnvVanilla

def main():
    print("Starting RL_Agent")
    # Set up argument parser
    parser = argparse.ArgumentParser(description='Train RL agent in CARLA environment')
    parser.add_argument('--algorithm', type=str, default='ppo', choices=['ppo', 'sac'],
                      help='RL algorithm to use (ppo or sac)')
    parser.add_argument('--timesteps', type=int, default=500_000,
                      help='Number of timesteps to train (default: 500,000)')
    parser.add_argument('--load_model', type=str, default=None,
                      help='Path to load existing model for further training')
    parser.add_argument('--learning_rate', type=float, default=5e-5,
                      help='Learning rate (default: 0.00005)')
    parser.add_argument('--anxious', action='store_true',
                      help='Use AnxiousCarlaEnv instead of CarlaEnv')
    parser.add_argument('--save_path', type=str, default=None,
                      help='Path to save the trained model')
                      
    # Parse arguments
    args = parser.parse_args()
    
    print(f"Using algorithm: {args.algorithm}")
    print(f"Anxious flag is set to: {args.anxious}")
    
    # Setup training directories
    mode = 'anxious' if args.anxious else 'vanilla'
    algorithm = args.algorithm.lower()
    model_dir = os.path.join('training', mode, algorithm, 'models')
    reward_dir = os.path.join('training', mode, algorithm, 'rewards')
    os.makedirs(model_dir, exist_ok=True)
    os.makedirs(reward_dir, exist_ok=True)

    # Create and wrap environment with Monitor for logging
    def make_env():
        env = CarlaEnvVanilla()
        check_env(env)
        return Monitor(env)
    env = DummyVecEnv([make_env])

    # Policy network configuration
    policy_kwargs = dict(
        activation_fn=torch.nn.Tanh,
        net_arch=dict(
            pi=[256, 256],  # Policy network 
            qf=[256, 256],  # Q-value network for SAC
            vf=[256, 256]   # Value network for PPO
        )
    )

    # Check for existing model to load
    load_path = None
    if args.load_model:
        load_path = os.path.join('training', mode, algorithm, 'models', args.load_model)
        if os.path.exists(load_path):
            print(f"Loading existing model from {load_path}")
        else:
            print(f"Model not found at {load_path}, creating new model")
            load_path = None
    
    # Create or load the model based on algorithm choice
    if algorithm == 'ppo':
        if load_path:
            model = PPO.load(load_path, env=env, device='cpu')
            model.learning_rate = args.learning_rate
        else:
            print("Creating new PPO model")
            model = PPO(
                "MlpPolicy",
                env,
                verbose=1,
                learning_rate=args.learning_rate,
                n_steps=2048,
                batch_size=128,
                n_epochs=10,
                gamma=0.99,
                gae_lambda=0.95,
                clip_range=0.2,
                ent_coef=0.01,
                policy_kwargs=policy_kwargs,
                device='cpu'
            )
    elif algorithm == 'sac':
        if load_path:
            model = SAC.load(load_path, env=env, device='cpu')
            model.learning_rate = args.learning_rate
        else:
            print("Creating new SAC model")
            model = SAC(
                "MlpPolicy",
                env,
                verbose=1,
                learning_rate=args.learning_rate,
                buffer_size=100_000,
                batch_size=128,
                gamma=0.99,
                tau=0.005,
                learning_starts=10000,
                train_freq=4,
                gradient_steps=1,
                action_noise=None,
                policy_kwargs=policy_kwargs,
                device='cpu'
            )

    # Create checkpoint callback saving to model_dir
    checkpoint_callback = CheckpointCallback(
        save_freq=100_000,
        save_path=model_dir,
        name_prefix=f"{algorithm}_checkpoint"
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
                ax.set_title(f'Episodic Reward Progression - {algorithm.upper()}')
                os.makedirs(self.save_dir, exist_ok=True)
                fig_path = os.path.join(self.save_dir, f"{algorithm}_reward_progress_{self.num_timesteps}.png")
                fig.savefig(fig_path)
                plt.close(fig)
            return True
        
    plot_callback = PlotRewardCallback(save_freq=50_000, save_dir=reward_dir)
    # Combine callbacks
    callback_list = CallbackList([checkpoint_callback, plot_callback])

    print("Starting training")
    # Train the model with checkpoint and plotting callbacks
    model.learn(total_timesteps=args.timesteps, callback=callback_list)

    print("Done training")
    # Save the final model to model_dir
    final_path = os.path.join(model_dir, f"{algorithm}_final")
    model.save(final_path)
    print(f"Model saved to {final_path}")

    # Close the environment
    env.close()

if __name__ == "__main__":
    main()