from typing import SupportsFloat
import pygame
import random
import numpy as np
from enum import Enum
import copy

import torch

from environments.environment import Env
from models.TetrisConvModel import TetrisAgent
from models.tetris_discrete_model import TetrisAI
from config import Config
from generator import Generator
from functools import partial

class Actions(Enum):
    NoAction = 0
    MoveLeft = 1
    MoveRight = 2
    RotateClock = 3
    RotateCClock = 4
    MoveDown = 5



# Colors
BLACK = (0, 0, 0)
WHITE = (255, 255, 255)
RED = (255, 0, 0)
GREEN = (0, 255, 0)
BLUE = (0, 0, 255)
GRAY = (170, 170, 170)  # Light gray color only for grid lines


# Tetromino shapes
SHAPES = [
    [[1, 1, 1, 1]],  # I
    [[1, 1], [1, 1]],  # O
    [[0, 1, 0], [1, 1, 1]],  # T
    [[0, 1, 1], [1, 1, 0]],  # S
    [[1, 1, 0], [0, 1, 1]],  # Z
    [[1, 0, 0], [1, 1, 1]],  # J
    [[0, 0, 1], [1, 1, 1]]  # L
]
def shape_to_numpy(shape):
    ret_shape = np.zeros((2,4)) # padding to maximum shape
    for row in range(len(shape)):
        for col in range(len(shape[row])):
            ret_shape[row][col] = shape[row][col]
    return ret_shape

class Tetromino:
    def __init__(self, shape, grid_shape: tuple[int, int]): # grid_size: (height, width)
        self.shape = shape
        self.row_size = grid_shape[0]
        self.col_size = grid_shape[1]
        self.x = grid_shape[1] // 2 - len(shape[0]) // 2
        self.y = 0
        self.rotation_state = 0

    def rotate(self, clockwise=True):
        if len(self.shape) == 2 and len(self.shape[0]) == 2:  # O piece doesn't rotate
            return
        
        # Perform rotation
        if clockwise:
            rotated = [list(row) for row in zip(*self.shape[::-1])]  # Clockwise rotation
        else:
            inverted_rows = [row[::-1] for row in self.shape]
            rotated = [list(row) for row in zip(*inverted_rows)]
        
        self.shape = rotated
        self.rotation_state = (self.rotation_state + 1) % 4

    def is_colliding(self, grid, offset=(0, 0)):
        off_y, off_x = offset
        for y, row in enumerate(self.shape):
            for x, cell in enumerate(row):
                if cell:
                    new_x = self.x + x + off_x
                    new_y = self.y + y + off_y
                    if new_x < 0 or new_x >= self.col_size or new_y >= self.row_size:
                        return True
                    if new_y >= 0 and grid[new_y][new_x]:
                        return True
        return False

    # def draw(self, screen, grid_size):
    #     for y, row in enumerate(self.shape):
    #         for x, cell in enumerate(row):
    #             if cell:
    #                 pygame.draw.rect(screen, RED, 
    #                                  pygame.Rect((self.x + x) * grid_size, 
    #                                              (self.y + y) * grid_size, 
    #                                              grid_size, grid_size))

    def move_down(self):
        self.y += 1





class PygameTetris(Env):
    def __init__(self, seed, discrete_obs=True, render=False, scale=1, config: Config = None):
        self._init_rewards(config)
        self.random_seed = seed #TODO   
        self.discrete_obs = discrete_obs
        # Screen dimensions with extra width for the preview
        self.SCALE=scale
        self.PREVIEW_WIDTH = 25 * self.SCALE
        self.SCREEN_WIDTH = 50 * self.SCALE + self.PREVIEW_WIDTH
        self.SCREEN_HEIGHT = 100 * self.SCALE
        self.COLUMNS = 10
        self.ROWS = 20
        self.GRID_SIZE = 5 * self.SCALE

        self.grid = self.generate_grid() # This will keep track of the tiles that are set in place and the current moving tile. Observation are generated from here.
        self.static_grid = self.generate_grid() # This will keep track of the tiles that are set in place.
        
        random.seed(seed)
        self.current_tetromino: Tetromino = Tetromino(random.choice(SHAPES), (self.ROWS, self.COLUMNS))
        self.next_tetromino: Tetromino = Tetromino(random.choice(SHAPES), (self.ROWS, self.COLUMNS))

        if render:
            self.screen = pygame.display.set_mode((self.SCREEN_WIDTH, self.SCREEN_HEIGHT))
        else:
            self.screen = pygame.Surface((self.SCREEN_WIDTH, self.SCREEN_HEIGHT))
        self.render = render

        self.step_count = 0
        info = {}

    def step(self, action: Actions) -> tuple[np.ndarray, SupportsFloat, bool, dict[str, object]]:
        if isinstance(action, int):
            action = Actions(action)

        info = {
                "tetromino_placed": False,
                "lines_cleared": 0,
                "current_height": 0,
                "bumpiness": 0,
                "holes": 0,
                "step_reward": 0,
                "height_place_reward": 0,
                "line_clear_reward": 0,
                "height_penalty": 0,
                "bumpiness_penalty": 0,
                "hole_penalty": 0,
                "game_over_penalty": 0,
                "line_density_reward": 0,
                }
        # Return values
        obs = np.zeros(self.observation_space)
        reward = 0
        terminated = False
        # truncated = False

        #TODO: Action penalty?
        # reward -= self.action_penalty
        # elif action == Actions.NoAction:
        #     reward += self.action_penalty
        
        # if not self.apply_action(action):
        #     reward -= 50
        if self.apply_action(action) and action != Actions.NoAction:
            reward += self.step_reward
            info["step_reward"] = self.step_reward
        # else:
        #     info["step_reward"] = -self.step_reward
        #     reward -= self.step_reward

        if self.current_tetromino.is_colliding(self.static_grid, (1, 0)):
            self.update_grid(self.static_grid, self.current_tetromino, None)
            lines_cleared = self.clear_lines()
            height_placed = self.current_tetromino.y

            info["tetromino_placed"] = True
            info["lines_cleared"] = lines_cleared
            
            
            self.current_tetromino = self.next_tetromino
            self.next_tetromino = Tetromino(random.choice(SHAPES), (self.ROWS, self.COLUMNS))

            self.update_grid(self.grid, self.current_tetromino, None)

            if self.check_game_over(self.current_tetromino):
                print("Game Over!")
                terminated = True
                reward -= self.game_over_penalty
                info["game_over_penalty"] = -self.game_over_penalty
                return obs, reward, terminated, info
            reward += self.calculate_placement_reward(lines_cleared, height_placed, info)
            reward += self._static_reward(info)
        else:
            tetromino_to_remove = copy.deepcopy(self.current_tetromino)
            self.current_tetromino.move_down()
            self.update_grid(self.grid, self.current_tetromino, tetromino_to_remove)

        info["current_height"] = self.calculate_height()
        info["bumpiness"] = self.calculate_bumpiness(self.static_grid)
        info["holes"] = self.count_holes()

        
        # reward -= self._static_reward() # penalty
        self.step_count += 1


        self.render_screen()
        obs = self._get_observation()

        return obs, reward, terminated, info

    def reset(self):
        self.step_count = 0

        self.grid = self.generate_grid()
        self.static_grid = self.generate_grid()
        self.current_tetromino: Tetromino = Tetromino(random.choice(SHAPES), (self.ROWS, self.COLUMNS))
        self.next_tetromino: Tetromino = Tetromino(random.choice(SHAPES), (self.ROWS, self.COLUMNS))

        self.update_grid(self.grid, self.current_tetromino, None)
        self.render_screen()
        obs = self._get_observation()
        

        return obs
        
    def apply_action(self, action: Actions) -> bool:
        """Apply an action. Returns whether the action could be applied or not."""
        backup_tetromino = copy.deepcopy(self.current_tetromino)
        if action == Actions.MoveLeft:  # Move left
            self.current_tetromino.x -= 1
            if self.current_tetromino.is_colliding(self.static_grid):
                self.current_tetromino.x += 1
                return False
        elif action == Actions.MoveRight:  # Move right
            self.current_tetromino.x += 1
            if self.current_tetromino.is_colliding(self.static_grid):
                self.current_tetromino.x -= 1
                return False
        elif action == Actions.MoveDown:  # Move right
            self.current_tetromino.y += 1
            if self.current_tetromino.is_colliding(self.static_grid):
                self.current_tetromino.y -= 1
                return False
        elif action == Actions.RotateClock:  # Rotate clockwise
            self.current_tetromino.rotate()
            if self.current_tetromino.is_colliding(self.static_grid):
                self.current_tetromino.rotate(clockwise=False)
                return False
        elif action == Actions.RotateCClock: # counterclockwise
            self.current_tetromino.rotate(clockwise=False)
            if self.current_tetromino.is_colliding(self.static_grid):
                self.current_tetromino.rotate()
                return False
            
        self.update_grid(self.grid, self.current_tetromino, backup_tetromino)
        return True
    
    def update_grid(self, grid, new_tetromino: Tetromino, remove_tetromino: Tetromino):
        """Update the information in each cell. Removes the content of remove_tetronimo, before the content of new_tetronimo is put in the grid."""
        if remove_tetromino:
            for y, row in enumerate(remove_tetromino.shape):
                for x, cell in enumerate(row):
                    if cell:
                        grid[remove_tetromino.y + y][remove_tetromino.x + x] = 0
        for y, row in enumerate(new_tetromino.shape):
            for x, cell in enumerate(row):
                if cell:
                    grid[new_tetromino.y + y][new_tetromino.x + x] = 1



    # Reward stuff
    def calculate_placement_reward(self, lines_cleared, height_placed, info):
        line_clear_reward = lines_cleared ** 2 * self.line_clear_reward
        height_place_reward = (self.ROWS - height_placed) * self.height_place_reward
        info["height_place_reward"] = height_place_reward
        info["line_clear_reward"] = line_clear_reward

        return height_place_reward + line_clear_reward
    

    def _static_reward(self, info):
        """Reward based on the current tetrominos on the grid. Return value is positive. As penalty it should be negated!"""
        hole_penalty = self.count_holes() * self.hole_penalty
        # TODO: the issue here is that landing a tile incidentally gives negative reward. Maybe bumpiness and height need to be adjusted with a certain "gold standard".
        bumpiness_penalty = self.calculate_bumpiness(self.static_grid) * self.bumpiness_penalty
        height_penalty = self.calculate_height() * self.height_penalty
        line_density_reward = self.calculate_line_density() * self.line_density_reward
        '''# Penalty for height differences
        for i in range(COLUMNS - 1):
            height_diff = abs(sum(grid[j][i] for j in range(ROWS)) - sum(grid[j][i+1] for j in range(ROWS)))
            reward -= height_diff * 0.1
        '''

        info["height_penalty"] = height_penalty
        info["hole_penalty"] = hole_penalty
        info["bumpiness_penalty"] = bumpiness_penalty
        info["line_density_reward"] = line_density_reward
        return -height_penalty - hole_penalty - bumpiness_penalty + line_density_reward

    def clear_lines(self):
        """
        Goes through all lines in static_grid and grid and clears them if they are full.
        """
        full_rows = [i for i, row in enumerate(self.grid) if all(row)]
        for i in full_rows:
            del self.grid[i]
            del self.static_grid[i]
            self.grid.insert(0, [0 for _ in range(self.COLUMNS)])
            self.static_grid.insert(0, [0 for _ in range(self.COLUMNS)])
        return len(full_rows)

    def check_game_over(self, tetromino: Tetromino):
        """**This method must be called immediately after a new tile has spawned.** It checks on the static_grid whether there is a collision immediately after spawning."""
        return tetromino.is_colliding(self.static_grid, offset=(0, 0))


    def calculate_bumpiness(self, grid) -> int:
        """Returns the sum of height-difference over columns, starting from a difference of 1."""
        heights = [0] * self.COLUMNS
        for col in range(self.COLUMNS):
            for row in range(self.ROWS):
                if grid[row][col]:
                    heights[col] = self.ROWS - row
                    break
        
        bumpiness = 0
        for i in range(self.COLUMNS - 1):
            bumpiness += max(0, abs(heights[i] - heights[i+1]) - 1)
        
        return bumpiness

    def calculate_line_density(self) -> int:
        line_density = 0
        for row in self.static_grid:
            density = 0
            for col in row:
                density += col
            line_density += (density ** 2) / self.COLUMNS
        return line_density


    def count_holes(self):
        holes = 0
        for col in range(self.COLUMNS):
            block_found = False
            for row in range(self.ROWS):
                if self.static_grid[row][col]:
                    block_found = True
                elif block_found:
                    holes += 1
        return holes
    # def count_holes(self) -> int:
    #     """
    #     Returns the amount of holes. Holes are defined as follows:\n
    #     - A free cell that is surrounded (except on its bottom) by filled spots or another quasi-surrounded cell
    #     - A quasi-surrounded cell is a cell that is surrounded by filled spots, however there may be a distance between the cell and these filled spots > 1. In that distance there may only be other quasi-surrounded cells.

    #     **Examples:**\n
    #     [0,0,0,0,0,0]    [0,0,0,0,0,0]\n
    #     [0,0,0,0,0,0]    [0,0,0,0,0,0]\n
    #     [0,0,0,1,0,0]    [0,0,0,1,0,0]\n
    #     [0,0,1,0,1,1]    [0,0,1,0,1,0]\n
    #     [0,0,1,0,0,0]    [0,0,1,0,0,0]\n
    #     The left example has 4 holes, whereas the right example only 1. In the right example, the cell at (5, 5) is not quasi-surrounded and hence (5, 4) and (5, 3) are also not quasi-surrounded. Therefore, only (4, 3) is a hole.
    #     """
    #     holes_table = self.generate_grid()
    #     hole_count = 0
    #     for row in range(self.ROWS):
    #         # block_found = False
    #         for col in range(self.COLUMNS):
    #             if self.static_grid[row][col]:
    #                 continue
    #             left_open = True
    #             up_open = True
    #             if col > 0:
    #                 if self.static_grid[row][col - 1]:
    #                     left_open = False
    #                 elif holes_table[row][col - 1] > 0:
    #                     left_open = False
    #                     holes_table[row][col] = holes_table[row][col - 1]
    #             if row > 0:
    #                 if self.static_grid[row - 1][col] or holes_table[row - 1][col] > 0:
    #                     up_open = False
    #             if up_open: #opening on top found
    #                 holes_table[row][col] = 0
    #             if not up_open and not left_open:
    #                 holes_table[row][col] += 1
    #                 if col + 1 >= self.COLUMNS or self.static_grid[row][col + 1]:
    #                     hole_count += holes_table[row][col]
                    
    #     return hole_count

    def calculate_height(self):
        for row in range(self.ROWS):
            if any(self.static_grid[row]):
                return self.ROWS - row
        return 0

    def count_clearable_lines(grid):
        return sum(1 for row in grid if all(row))
    

    # Rendering
    def _get_observation(self):
        if self.discrete_obs:
            grid_flattened = np.array(self.grid).flatten()
            tetromino_flattened = shape_to_numpy(self.next_tetromino.shape).flatten()
            obs = np.hstack((grid_flattened, tetromino_flattened))
        else:
            obs = pygame.surfarray.array3d(self.screen)
            obs = self.reshape_obs(obs)
            obs = obs.astype(np.float32) / 255.0
        return obs
    
    def _reset_screen(self):
        self.screen.fill(BLACK)

        # Draw gray grid lines
        for x in range(self.COLUMNS + 1):
            pygame.draw.line(self.screen, GRAY, (x * self.GRID_SIZE, 0), (x * self.GRID_SIZE, self.SCREEN_HEIGHT))
        for y in range(self.ROWS + 1):
            pygame.draw.line(self.screen, GRAY, (0, y * self.GRID_SIZE), (self.SCREEN_WIDTH - self.PREVIEW_WIDTH, y * self.GRID_SIZE))

    def render_screen(self):
        """Resets the screen. Then draws the grid, including the current running tile. TODO: Color current running tile differently. Then draws the next tile."""
        self._reset_screen()
        # self.current_tetromino.draw(self.screen)
        for y in range(self.ROWS):
            for x in range(self.COLUMNS):
                if self.grid[y][x]:
                    pygame.draw.rect(self.screen, BLUE, 
                                     pygame.Rect(x * self.GRID_SIZE, 
                                                 y * self.GRID_SIZE, 
                                                 self.GRID_SIZE, self.GRID_SIZE))
        self.draw_next_tetromino(self.next_tetromino.shape, self.COLUMNS + 1, 2, self.screen, GREEN)

        if self.render:
            pygame.display.flip()

    def draw_next_tetromino(self, shape, x, y, screen, color):
        for row_idx, row in enumerate(shape):
            for col_idx, cell in enumerate(row):
                if cell:
                    pygame.draw.rect(
                        screen,
                        color,
                        pygame.Rect(
                            (x + col_idx) * self.GRID_SIZE,
                            (y + row_idx) * self.GRID_SIZE,
                            self.GRID_SIZE,
                            self.GRID_SIZE
                        )
                    )


    # Helper
    def generate_grid(self):
        return [[0 for _ in range(self.COLUMNS)] for _ in range(self.ROWS)]
    
    def _init_rewards(self, config: Config):
        if config:
            self.line_clear_reward = config.line_clear_reward
            self.height_place_reward = config.height_place_reward
            self.height_penalty = config.height_penalty
            self.bumpiness_penalty = config.bumpiness_penalty
            self.hole_penalty = config.hole_penalty
            self.line_density_reward = config.line_density_reward
            self.step_reward = config.step_reward
            self.game_over_penalty = config.game_over_penalty
        # else:
        #     self.line_clear_reward = 50
        #     self.height_place_reward = 0.3
        #     self.height_penalty = 0.2
        #     self.bumpiness_penalty = 0.5
        #     self.hole_penalty = 2
        #     self.game_over_penalty = 500
        #     # self.action_penalty = 3e-4
        #     self.step_reward = 1e-3

    def close(self):
        pygame.quit()

    def __str__(self) -> str:
        ret = ""
        for row in self.grid:
            for cell in row:
                if cell:
                    ret += "X"
                else:
                    ret += "O"
            ret += "\n"
        ret += f"next shape: {self.next_tetromino.shape}"
        return ret
    @property
    def action_space(self):
        return len(Actions)
    
    @property
    def get_game_length(self):
        return self.step_count

    @property
    def observation_space(self):
        if self.discrete_obs:
            grid_flattened_size = np.array(self.grid).flatten().shape
            tetromino_flattened_size = shape_to_numpy(self.next_tetromino.shape).flatten().shape
            return grid_flattened_size[0] + tetromino_flattened_size[0]
        else:
            obs_space = pygame.surfarray.array3d(self.screen).shape # (W, H, C)
            obs_space = (obs_space[2], obs_space[1], obs_space[0]) # (C, H, W) for conv2d layer
            return obs_space
        


    @staticmethod
    def get_environment(seed=0, discrete_obs=False, render=False, scale=1, config: Config = None):
        return PygameTetris(seed, discrete_obs, render=render, scale=scale, config=config)



# potential buggy things so far:
# penalty for making moves
# holes are defined too losely
# rotation only works in one direction
# No illegal move for rotation
# No penalty for illegal moves
def let_AI_play_pygame(model_file, device, prob_actions: bool, games=1, speed=1): # currently only works for conv2d model
    factory = partial(PygameTetris.get_environment, render = True)

    environment_seeds = [np.random.randint(0, 2**31) for _ in range(games)]
    environments: list[PygameTetris] = [factory(seed) for seed in environment_seeds]
    observations = [env.reset() for env in environments]

    FPS = 2

    observation_space = environments[0].observation_space
    action_space = environments[0].action_space

    model = TetrisAgent(*observation_space, action_space, device)
    # model = TetrisAI()
    model.to(device)
    try:
        model.load_state_dict(torch.load("exports/" + model_file, map_location=device))
        print("Loaded existing model to play.")
    except FileNotFoundError:
        print(f"Invalid path to model file: {model_file}.")
        return
    
    clock = pygame.time.Clock()


    game_over = False
    running = True
    env_idx = 0

    print(f"starting game: ------------{env_idx + 1}------------")
    clock.tick(FPS)
    while running:
        game: PygameTetris = environments[env_idx]
        for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    pygame.quit()
                    running = False
                    break
        pis = model.get_pis(observations[env_idx])
        if prob_actions:
            action, _ = Generator.sample_action(pis)
        else:
            action = torch.argmax(pis).item()
        print(f"action: {action}")
        # observations[env_idx] = game.step(action) # TODO: Implement toggle for train/play in step

        observations[env_idx], _, game_over, _ = game.step(action)
        game.render_screen()

        if game_over:
            env_idx += 1
            if env_idx >= games:
                running = False 
            print(f"starting game: ------------{env_idx + 1}------------")

        clock.tick(FPS)


def play_pygame(speed=1, scale=1): 
    game = PygameTetris(0, discrete_obs=False, render=True, scale=scale)
    FPS = 64

    clock = pygame.time.Clock()

    game_over = False
    running = True
    
    clock.tick(FPS)
    frame_count = 0

    while running:
        frame_count += 1
        action = Actions.NoAction
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                pygame.quit()
                running = False
                break
            if event.type == pygame.KEYDOWN:
                if event.key == pygame.K_LEFT:
                    action = Actions.MoveLeft
                elif event.key == pygame.K_RIGHT:
                    action = Actions.MoveRight
                elif event.key == pygame.K_e:
                    action = Actions.RotateClock
                elif event.key == pygame.K_q:
                    action = Actions.RotateCClock
                elif event.key == pygame.K_DOWN:
                    action = Actions.MoveDown
                game.apply_action(action)
                game.render_screen()
        if frame_count % (FPS // speed) == 0:
            _, _, game_over, _ = game.step(action)
            
        if game_over:
            running = False


        clock.tick(FPS)

    pygame.quit()
