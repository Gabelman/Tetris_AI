from PPO import PPO
import gymnasium as gym
import wandb
import torch

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
if __name__ == '__main__':
    # wandb.login()
    env = gym.make("ALE/Tetris-v5")
    ppo = PPO(env, device)
    ppo.init_hyperparameters(episodes_per_batch = 4, max_timesteps_per_episode = 100, updates_per_iteration = 5, gamma = 0.95, epsilon = 0.2, lam = 0.94, actor_lr = 1e-3, cricit_lr = 1e-3)
    ppo.train(10000)


    env.close()
