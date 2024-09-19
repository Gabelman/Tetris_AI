from PPO import PPO
import gymnasium as gym
import wandb
import torch


device = torch.device('cuda:1' if torch.cuda.is_available() else 'cpu')
if __name__ == '__main__':
    # wandb.login()
    # env = gym.make("ALE/Tetris-v5")
    ppo = PPO(device)
    ppo.init_hyperparameters(episodes_per_batch = 1, max_timesteps_per_episode = 200, updates_per_iteration = 1, num_mini_batch_updates = 16, num_sub_mini_batches = 4, gamma = 0.95, epsilon = 0.2, lam = 0.94, lr = 1e-3)
    ppo.train(100000)

    ppo.close()


    # env.close()
