from environments.environment import Env
import gymnasium as gym
import warnings

class GymnasiumTetris(Env):
    def __init__(self, discrete_obs, render):
        if discrete_obs:
            raise NotImplementedError("Discrete obs cannot be take from gymnasium environment.")
        self.render = render
        if render:
            self.env = gym.make("ALE/Tetris-v5", render_mode="human")
        else:
            self.env = gym.make("ALE/Tetris-v5")

    def step(self, action):
        obs, reward, terminated, truncated, info = self.env.step(action)
        done = terminated or truncated
        obs = self.reshape_obs(obs)
        return obs, reward, done

    def close(self):
        self.env.close()

    def reset(self):
        return self.env.reset()[0]
    
    @property
    def action_space(self) -> int:
        return int(self.env.action_space.n)

    @property
    def observation_space(self):
        obs_space = self.env.observation_space.shape # (W, H, C)
        obs_space = (obs_space[2], obs_space[1], obs_space[0]) # (C, H, W) for conv2d layers
        return obs_space

    @staticmethod
    def get_environment(seed=0, discrete_obs=False, render=False, scale=1):
        if seed != 0:
            warnings.warn("Gymnasium environment was seeded, but seeds have no effect.")
        return GymnasiumTetris(discrete_obs=discrete_obs, render=render)







    