import gymnasium as gym
from tqdm import tqdm
import time

env = gym.make("ALE/Tetris-v5", render_mode="human")
observation, info = env.reset()
for i in range(100):
    action = env.action_space.sample()
    observation, reward, terminated, truncated, info = env.step(action)
    time.sleep(0.1)
print(observation)


