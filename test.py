import unittest
import numpy as np
from environments.pygame_tetris import PygameTetris, Placement
# from environments.gymnasium_tetris import GymnasiumTetris
from train_algorithms.ppo_trainer import PPO
from config import Config
from functools import partial
from generator import Generator
import random

from grids import *

import pygame
pygame.init()


class TestTetris(unittest.TestCase):
    def test_obs_space(self):
        env = PygameTetris.get_environment()
        space = env.observation_space
        print(space)

    def test_generator_env(self):
        config = Config(episodes_per_batch=40)
        trainer = PPO('cpu', config, experiment=1)
        self.assertEqual(40, len(trainer.generator.environments))

    def test_generator_env2(self):
        config = Config(episodes_per_batch=40)
        environment_factory = partial(PygameTetris.get_environment, discrete_obs=False, render=False, scale=3)
        generator = Generator(num_environments=config.episodes_per_batch, max_timesteps_per_episode=config.max_timesteps_per_episode, environment_factory=environment_factory, gamma=config.gamma, device='cpu')
        self.assertEqual((3, 300, 225), generator.observation_space)
        self.assertEqual(6, generator.action_space)

    def test_generator_seeding(self):
        config = Config(episodes_per_batch=5)
        environment_factory = partial(PygameTetris.get_environment, discrete_obs=False, render=False, scale=3)
        for i in range(10):
            generator = Generator(num_environments=config.episodes_per_batch, max_timesteps_per_episode=config.max_timesteps_per_episode, environment_factory=environment_factory, gamma=config.gamma, device='cpu')
            print(f"seed: {generator.environment_seeds}")
        

    # def test_generator_env3(self):
    #     config = Config(episodes_per_batch=40)
    #     environment_factory = partial(GymnasiumTetris.get_environment, discrete_obs=False, render=False, scale=3)
    #     generator = Generator(num_environments=config.episodes_per_batch, max_timesteps_per_episode=config.max_timesteps_per_episode, environment_factory=environment_factory, gamma=config.gamma, device='cpu')
    #     self.assertEqual((3, 160, 210), generator.observation_space)
    #     self.assertEqual(5, generator.action_space)

    def test_calc_bumpiness(self):
        tetris = PygameTetris.get_environment()
        
        tetris.static_grid = grid11
        self.assertEqual(6, tetris.calculate_bumpiness(grid11))

    def test_count_holes(self):
        tetris = PygameTetris.get_environment()
        
        tetris.static_grid = grid5
        self.assertEqual(0, tetris.calculate_line_density())
        tetris.static_grid = grid6
        self.assertEqual(1.3, tetris.calculate_line_density())
        tetris.static_grid = grid7
        self.assertEqual(2.6, tetris.calculate_line_density())

    def test_line_density(self):
        tetris = PygameTetris.get_environment()
        
        tetris.static_grid = grid1
        self.assertEqual(11, tetris.calculate_height())
        tetris.static_grid = grid2
        self.assertEqual(14, tetris.calculate_height())
        tetris.static_grid = grid3
        self.assertEqual(20, tetris.calculate_height())
        tetris.static_grid = grid4
        self.assertEqual(0, tetris.calculate_height())

    def test_tetris_holes(self):
        tetris = PygameTetris.get_environment(seed=0)
        
        tetris.static_grid = grid5
        self.assertEqual(0, tetris.count_holes())
        tetris.static_grid = grid6
        self.assertEqual(1, tetris.count_holes())
        tetris.static_grid = grid7
        self.assertEqual(5, tetris.count_holes())
        tetris.static_grid = grid8
        self.assertEqual(1, tetris.count_holes())
        tetris.static_grid = grid9
        self.assertEqual(3, tetris.count_holes())
        
    def test_clear_line(self):
        tetris = PygameTetris.get_environment(seed=0)

        tetris.grid = grid10
        tetris.clear_lines()
        self.assertEqual(grid10, grid_gold10)

    def test_discrete_placements(self):
        config = Config(predict_placement=True)
        env = PygameTetris.get_environment(seed=10, config=config) # seed 10: current_tetromino is Z-shape
        valid = env.get_valid_placements()
        indeces = [i for i in range(env.action_space) if valid[i]]
        placements = [Placement(v) for v in indeces]
        env.step(placements[5])
        print(valid)

        
    
    def test_action_space(self):
        env = PygameTetris.get_environment()
        action_space_size = env.action_space
        action = random.randint(0, action_space_size -1)
        self.assertIn(action, range(action_space_size))

if __name__ == '__main__':
    unittest.main()

