from train_algorithms.ppo_trainer import PPO
# import gymnasium as gym
import wandb
import torch
from config import Config


device = torch.device('cuda:1' if torch.cuda.is_available() else 'cpu')
if __name__ == '__main__':
    wandb.login()
    # env = gym.make("ALE/Tetris-v5")
    config = Config(episodes_per_batch=128, updates_per_iteration=3, num_mini_batch_updates=32, num_sub_mini_batches=8, overall_timesteps=100000, lr=0.01)
    ppo = PPO(device, config, experiment=1)
    ppo.train(100000)

    ppo.close()


    # env.close()
