from abc import abstractmethod
import numpy as np

class Env():
    def __init__(self):
        pass

    @abstractmethod
    def step(self, action):
        """Does the next step in the environment."""

    

