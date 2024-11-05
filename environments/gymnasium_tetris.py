from environments.environment import Env
import gymnasium as gym
import warnings
import torch
import pygame
import numpy as np

from config import Config
from generator import Generator
from models.TetrisConvModel import TetrisAgent


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
        return obs, reward, done, info

    def close(self):
        self.env.close()

    def reset(self):
        obs: np.ndarray = self.env.reset()[0]
        obs = self.reshape_obs(obs)
        return obs
    
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

    @property
    def get_game_length(self) -> int:
        raise NotImplementedError


def let_AI_play_gymnasium(model_file, direct_placement, device, prob_actions: bool, games=1, speed=1, scale=1): # currently only works for conv2d model
    config = Config(predict_placement=direct_placement)
    env = GymnasiumTetris(False, True)

    # environment_seeds = [np.random.randint(0, 2**31) for _ in range(games)]
    # game: list[PygameTetris] = factory(environment_seeds[0])
    # display: list[PygameTetris] = factory_display(environment_seeds[0])
    # observations = [env.reset() for env in environments]
    # for env in environments_display:
    #     env.reset()
    # game.reset()
    # display.reset()

    FPS = speed

    observation_space = env.observation_space
    action_space = env.action_space

    model = TetrisAgent(*observation_space, action_space, device)
    # model = TetrisAI()
    model.to(device)
    if model_file:
        try:
            model.load_state_dict(torch.load("exports/" + model_file, map_location=device))
            print("Loaded existing model to play.")
        except FileNotFoundError:
            print(f"Invalid path to model file: {model_file}.")
            return
    else:
        print("Playing with newly initialized model.")
    clock = pygame.time.Clock()
    
    # b=True
    # while b:
    #     for event in pygame.event.get():
    #         if event.type == pygame.QUIT:
    #             pygame.quit()
    #             running = False
    #             break
    #         if event.type == pygame.KEYDOWN:
    #             if event.key == pygame.K_RETURN:
    #                 b =False
    #                 break
    #     clock.tick(30)

    clock.tick(FPS)
    for i in range(games):
        print(f"starting game: ------------{i+1}------------")
        # game: PygameTetris = factory(environment_seeds[i])
        # display: PygameTetris = factory(environment_seeds[i])
        obs = env.reset()
        # display.reset()
        game_over = False
        running = True

        while running:
            
            invalid = [False] * env.action_space
            for event in pygame.event.get():
                    if event.type == pygame.QUIT:
                        pygame.quit()
                        running = False
                        break
            # if env.direct_placement:
            #     valid = game.get_valid_placements()
            #     invalid = [not v for v in valid]
            pis = model.get_pis(obs, invalid=invalid)
            if prob_actions:
                action, _ = Generator.sample_action(pis)
            else:
                action = torch.argmax(pis).item()
            print(f"action: {action}")
            # observations[env_idx] = game.step(action) # TODO: Implement toggle for train/play in step

            obs, _, game_over, _ = env.step(action)
            # _,_,_,_ = display.step(action)
            # display.render_screen()

            if game_over:
                break

            clock.tick(FPS)





    