from PPO import PPO
# import gymnasium as gym
# import wandb
import torch
from config import Config


device = torch.device('cuda:1' if torch.cuda.is_available() else 'cpu')
if __name__ == '__main__':
    # wandb.login()
    # env = gym.make("ALE/Tetris-v5")
    config = Config()
    ppo = PPO(device, config)
    ppo.train(100000)

    ppo.close()


    # env.close()
