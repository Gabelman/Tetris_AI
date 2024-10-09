from abc import ABC, abstractmethod
from numpy import ndarray

from typing import SupportsFloat

class Env(ABC):
    def __init__(self, direct_placement):
        self.direct_placement = direct_placement

    @abstractmethod
    def step(self, action) -> tuple[ndarray, SupportsFloat, bool, dict[str, object]]:
        """
        Does the next step in the environment.
        Returns:
            obs(ndarray): The observation after taking the step in the environment.
            reward(SupportsFloat): Reward given for the action.
            terminated(bool): True if the environment has terminated. For now this only happens at a game-over state.
            info(dict[str, object]): Information on the step taken.
        """

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
    def get_game_length(self) -> int:
        """Returns the current length of the game."""

    @property
    @abstractmethod
    def observation_space(self):
        """Returns the shape of the observation: (C, H, W) for indescrete models, (H * W + tetronimo.shape.flatten()) for discrete models.\n
        The shape of tetronimo is taken to be a padded shape with (2, 4), hence the flattened shape is (8,)"""

    @staticmethod
    @abstractmethod
    def get_environment(seed=0, discrete_obs=False, render=False, scale=1):
        """Returns a new instance of the environment."""

    @staticmethod
    def reshape_obs(obs): # obs are give as (W, H, C), but should be (C, H, W)
        return obs.transpose(2, 1, 0)

