import torch
from torch import nn
import torch.nn.functional as F
from torch.distributions import Categorical
# import numpy as np
import gymnasium as gym
from torch import Tensor

# env = gym.make("ALE/Tetris-v5")
# observation, _ = env.reset()
# board_width, board_height, img_depth = observation.shape
# action_space = 5


class InitialImageEmbed(nn.Module):
    def __init__(self, channels, W, H, img_channels) -> None:
        super().__init__()
        # Layers are inspired by: https://github.com/adrien1018/betatetris-tablebase
        self.conv_layer = nn.Conv2d(img_channels, channels, 5, padding=2) # Keeps the size of each embedding, as kernel_size = 5 and padding=2
        self.conv_vertical = nn.Conv2d(img_channels, channels, (H, 1)) # Conv layer that goes through all vertical pixel-lines -> shape: (board_width, 1)
        self.conv_horizontal = nn.Conv2d(img_channels, channels, (1, W)) # Conv layer that goes through all horizontal pixel-lines -> shape: (1, board_height)
        self.batch_norm = nn.BatchNorm2d(channels)
        
    def forward(self, obs):
        # Embed Image
        conv_embed = self.conv_layer(obs)
        vertical_embed = self.conv_vertical(obs)
        horizontal_embed = self.conv_horizontal(obs)
        x = conv_embed + vertical_embed + horizontal_embed # This is allowed and applies addition of shape (1, x) on all values of dimension y in (y, x)

        # Regularization
        x = self.batch_norm(x)
        x = F.relu(x)
        return x

class TetrisAgent(nn.Module):
    def __init__(self, C, H, W, output_dim):
        super().__init__()
        kernel_depth = 8  # 8 chosen arbitratily at this point
        kernel_depth2 = 4
        self.board_embed = InitialImageEmbed(kernel_depth, W, H, C)

        self.network_head = nn.Sequential(
            nn.Conv2d(kernel_depth, kernel_depth2, 1), # Sum over Embeddings. This may as well be a fully connected linear layer. However, in that case BatchNorm2d couldn't be applied the same way.
            nn.BatchNorm2d(kernel_depth2),
            nn.Flatten(), # Flatten shape (board_width, board_height, 1)
            nn.ReLU(),
            nn.Linear(kernel_depth2 * H * W, output_dim)
        )
        self.value_head = nn.Sequential(
            nn.Conv2d(kernel_depth, kernel_depth2, 1), # Sum over Embeddings
            nn.BatchNorm2d(kernel_depth2),
            nn.Flatten(), # Flatten shape (board_width, board_height, 1)
            nn.ReLU(),
            nn.Linear(kernel_depth2 * H * W, 1)
        )

    def forward(self, obs):
        obs = self.add_batch_dimension(obs)
        initial_embed = self.board_embed(obs)
        pi = self.network_head(initial_embed)
        v = self.value_head(initial_embed)
        return pi, v
    
    def get_pis(self, obs):
        obs = self.add_batch_dimension(obs)
        initial_embed = self.board_embed(obs)
        pi = self.network_head(initial_embed)
        return pi
    
    @staticmethod
    def add_batch_dimension(obs: Tensor):
        if len(obs.shape) == 3:
            return obs.unsqueeze(0)
        return obs