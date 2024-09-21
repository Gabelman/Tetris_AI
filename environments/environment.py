from abc import ABC, abstractmethod

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
        """Returns the shape of the observation: (C, H, W) for indescrete models, (H * W + tetronimo.shape.flatten()) for discrete models.\n
        The shape of tetronimo is taken to be a padded shape with (2, 4), hence the flattened shape is (8,)"""

    @staticmethod
    @abstractmethod
    def get_environment(seed=0, discrete_obs=False, render=False, scale=1):
        """Returns a new instance of the environment."""

    @staticmethod
    def reshape_obs(obs): # obs are give as (W, H, C), but should be (C, H, W)
        return obs.transpose(2, 1, 0)

