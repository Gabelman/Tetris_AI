import pygame
import random
import numpy as np
from enum import Enum
import copy

from environments.environment import Env

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
    def __init__(self, seed, discrete_obs=True, render=False, scale=1):
        self._init_rewards()
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

        self.grid = self.generate_grid()
        self.static_grid = self.generate_grid()
        self.current_tetromino: Tetromino = Tetromino(random.choice(SHAPES), (self.ROWS, self.COLUMNS))
        self.next_tetromino: Tetromino = Tetromino(random.choice(SHAPES), (self.ROWS, self.COLUMNS))

        if render:
            self.screen = pygame.display.set_mode((self.SCREEN_WIDTH, self.SCREEN_HEIGHT))
        else:
            self.screen = pygame.Surface((self.SCREEN_WIDTH, self.SCREEN_HEIGHT))
        self.render = render

    
    def step(self, action: Actions):
        # self.reset_screen()
        
        # Return values
        obs = np.zeros(self.observation_space)
        reward = 0
        terminated = False
        # truncated = False


        #TODO: Action penalty?
        # reward -= self.action_penalty
        # elif action == Actions.NoAction:
        #     reward += self.action_penalty
        
        if not self.apply_action(action):
            reward -= 50

        if self.current_tetromino.is_colliding(self.static_grid, (1, 0)):
            self.update_grid(self.static_grid, self.current_tetromino, None)
            lines_cleared = self.clear_lines()
            height_placed = self.current_tetromino.y
            
            self.current_tetromino = self.next_tetromino
            self.next_tetromino = Tetromino(random.choice(SHAPES), (self.ROWS, self.COLUMNS))

            self.update_grid(self.grid, self.current_tetromino, None)

            if self.check_game_over(self.current_tetromino):
                print("Game Over!")
                terminated = True
                reward -= 500
                return obs, reward, terminated
            reward += self.calculate_reward(lines_cleared, height_placed)
        else:
            tetromino_to_remove = copy.deepcopy(self.current_tetromino)
            self.current_tetromino.move_down()
            self.update_grid(self.grid, self.current_tetromino, tetromino_to_remove)


        
        reward += self.step_reward


        self.render_screen()
        obs = self._get_observation()

        return obs, reward, terminated

    def reset(self):
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
    def calculate_reward(self, lines_cleared, height_placed):
        reward = 0
        reward += lines_cleared ** 2 * 50
        reward += (self.ROWS - height_placed) / 10
        reward -= self.count_holes(self.static_grid) * 2
        # TODO: the issue here is that landing a tile incidentally gives negative reward. Maybe bumpiness and height need to be adjusted with a certain "gold standard".
        reward -= self.calculate_bumpiness(self.static_grid) * 0.5
        reward -= self.calculate_height() * 0.2
        '''# Penalty for height differences
        for i in range(COLUMNS - 1):
            height_diff = abs(sum(grid[j][i] for j in range(ROWS)) - sum(grid[j][i+1] for j in range(ROWS)))
            reward -= height_diff * 0.1
        '''
        return reward

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
        """Returns the sum of height-difference over columns."""
        heights = [0] * self.COLUMNS
        for col in range(self.COLUMNS):
            for row in range(self.ROWS):
                if grid[row][col]:
                    heights[col] = self.ROWS - row
                    break
        
        bumpiness = 0
        for i in range(self.COLUMNS - 1):
            bumpiness += abs(heights[i] - heights[i+1])
        
        return bumpiness

    def count_holes(self, grid) -> int:
        """
        Returns the amount of holes. Holes are defined as follows:\n
        - A free cell that is surrounded (except on its bottom) by filled spots or another quasi-surrounded cell
        - A quasi-surrounded cell is a cell that is surrounded by filled spots, however there may be a distance between the cell and these filled spots > 1. In that distance there may only be other quasi-surrounded cells.

        **Examples:**\n
        [0,0,0,0,0,0]    [0,0,0,0,0,0]\n
        [0,0,0,0,0,0]    [0,0,0,0,0,0]\n
        [0,0,0,1,0,0]    [0,0,0,1,0,0]\n
        [0,0,1,0,1,1]    [0,0,1,0,1,0]\n
        [0,0,1,0,0,0]    [0,0,1,0,0,0]\n
        The left example has 4 holes, whereas the right example only 1. In the right example, the cell at (5, 5) is not quasi-surrounded and hence (5, 4) and (5, 3) are also not quasi-surrounded. Therefore, only (4, 3) is a hole.
        """
        holes_table = self.generate_grid()
        hole_count = 0
        for row in range(self.ROWS):
            # block_found = False
            for col in range(self.COLUMNS):
                if grid[row][col]:
                    continue
                left_open = True
                up_open = True
                if col > 0:
                    if grid[row][col - 1]:
                        left_open = False
                    elif holes_table[row][col - 1] > 0:
                        left_open = False
                        holes_table[row][col] = holes_table[row][col - 1]
                if row > 0:
                    if grid[row - 1][col] or holes_table[row - 1][col] > 0:
                        up_open = False
                if up_open: #opening on top found
                    holes_table[row][col] = 0
                if not up_open and not left_open:
                    holes_table[row][col] += 1
                    if col + 1 >= self.COLUMNS or grid[row][col + 1]:
                        hole_count += holes_table[row][col]
                    
        return hole_count

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
    
    def _init_rewards(self):
        self.action_penalty = 3e-4
        self.step_reward = 1e-3

    @property
    def action_space(self):
        return len(Actions)
    
    def close(self):
        pygame.quit()

    @property
    def observation_space(self):
        if self.discrete_obs:
            grid_flattened_size = np.array(self.grid).flatten().shape
            tetromino_flattened_size = shape_to_numpy(self.next_tetromino.shape).flatten().shape
            return (grid_flattened_size[0] + tetromino_flattened_size[0], )
        else:
            obs_space = pygame.surfarray.array3d(self.screen).shape # (W, H, C)
            obs_space = (obs_space[2], obs_space[1], obs_space[0]) # (C, H, W) for conv2d layer
            return obs_space

    @staticmethod
    def get_environment(seed=0, discrete_obs=False, render=False, scale=1):
        return PygameTetris(seed, discrete_obs, render=render, scale=scale)



# potential buggy things so far:
# penalty for making moves
# holes are defined too losely
# rotation only works in one direction
# No illegal move for rotation
# No penalty for illegal moves