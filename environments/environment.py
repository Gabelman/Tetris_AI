from abc import ABC, abstractmethod
# import numpy as np

# from enum import Enum


# class EnvironmentType(Enum):
#     Pygame: 0
#     Gymnasium: 1

class Env(ABC):
    def __init__(self):
        pass

    @abstractmethod
    def step(self, action):
        """Does the next step in the environment."""

    @abstractmethod
    def close(self):
        """Cleans up the environment."""

    @abstractmethod
    def reset(self):
        """
        Resets the environment.
        Returns:
            obs: The first observation of the environment.
        """
    @property
    @abstractmethod
    def action_space(self) -> int:
        """Returns the amount of possible actions."""

    @property
    @abstractmethod
    def observation_space(self):
        """Returns the amount of possible actions."""

    @staticmethod
    @abstractmethod
    def get_environment(seed=0, discrete_obs=False, render=False, scale=1):
        """Returns a new instance of the environment."""

    @staticmethod
    def reshape_obs(obs): # obs are give as (W, H, C), but should be (C, H, W)
        return obs.transpose(2, 1, 0)

