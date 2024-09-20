import pygame
import random
import numpy as np
from enum import Enum

from environment import Env



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

class Tetromino:
    def __init__(self, shape, grid_shape: tuple[int, int]): # grid_size: (height, width)
        self.shape = shape
        self.row_size = grid_shape[0]
        self.col_size = grid_shape[1]
        self.x = grid_shape[1] // 2 - len(shape[0]) // 2
        self.y = 0
        self.rotation_state = 0

    def rotate(self, grid, clockwise=True):
        if len(self.shape) == 2 and len(self.shape[0]) == 2:  # O piece doesn't rotate
            return

        original_x = self.x
        original_y = self.y
        original_shape = self.shape

        # Perform rotation
        if clockwise:
            rotated = [list(row) for row in zip(*self.shape[::-1])]  # Clockwise rotation
        else:
            inverted_rows = [row[::-1] for row in self.shape]
            rotated = [list(row) for row in zip(*inverted_rows)]
        
        # Check if rotation is possible
        self.shape = rotated
        if self.collision(grid):
            # If collision occurs, revert the rotation
            self.shape = original_shape
            self.x = original_x
            self.y = original_y
        else:
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

    def draw(self, screen, grid_size):
        for y, row in enumerate(self.shape):
            for x, cell in enumerate(row):
                if cell:
                    pygame.draw.rect(screen, RED, 
                                     pygame.Rect((self.x + x) * grid_size, 
                                                 (self.y + y) * grid_size, 
                                                 grid_size, grid_size))

    def lock_on_grid(self, grid):
        for y, row in enumerate(self.shape):
            for x, cell in enumerate(row):
                if cell:
                    grid[self.y + y][self.x + x] = 1

    def move_down(self):
        self.y += 1


# class Grid():
#     def __init__(self, rows, cols):
#         # self.rows = rows
#         # self.cols = cols
#         self.grid = np.zeros((cols, rows))

#     @property
#     def num_rows(self):
#         return self.grid.shape[1]
#     @property
#     def num_cols(self):
#         return self.grid.shape[0]
#     def get_val_at_pos()
class Actions(Enum):
    NoAction = 0
    MoveLeft = 1
    MoveRight = 2
    RotateClock = 3
    RotateCClock = 4


# def get_action():
#     # random.seed()
#     grid = [[0 for _ in range(COLUMNS)] for _ in range(ROWS)]
#     current_tetromino = Tetromino(random.choice(SHAPES))
#     next_tetromino = Tetromino(random.choice(SHAPES))
#     running = True

#     while running:
#         screen.fill(BLACK)

#         # Draw gray grid lines
#         for x in range(COLUMNS + 1):
#             pygame.draw.line(screen, GRAY, (x * GRID_SIZE, 0), (x * GRID_SIZE, SCREEN_HEIGHT))
#         for y in range(ROWS + 1):
#             pygame.draw.line(screen, GRAY, (0, y * GRID_SIZE), (SCREEN_WIDTH - PREVIEW_WIDTH, y * GRID_SIZE))
        
#         state = agent.get_state(grid, current_tetromino)
#         action = agent.act(state)

#         if action == 0:  # Move left
#             current_tetromino.x -= 1
#             if current_tetromino.collision(grid):
#                 current_tetromino.x += 1
#         elif action == 1:  # Move right
#             current_tetromino.x += 1
#             if current_tetromino.collision(grid):
#                 current_tetromino.x -= 1
#         elif action == 2:  # Rotate
#             current_tetromino.rotate(grid)

#         current_tetromino.y += 1
#         if current_tetromino.collision(grid):
#             current_tetromino.y -= 1
#             current_tetromino.lock(grid)
#             lines_cleared = clear_lines(grid)
            
#             current_tetromino = next_tetromino
#             next_tetromino = Tetromino(random.choice(SHAPES))

#             if check_game_over(grid, current_tetromino):
#                 print("Game Over!")
#                 running = False

#         current_tetromino.draw(screen)
#         for y in range(ROWS):
#             for x in range(COLUMNS):
#                 if grid[y][x]:
#                     pygame.draw.rect(screen, BLUE, 
#                                      pygame.Rect(x * GRID_SIZE, 
#                                                  y * GRID_SIZE, 
#                                                  GRID_SIZE, GRID_SIZE))
#         draw_tetromino(next_tetromino.shape, COLUMNS + 1, 2, screen, GREEN)

#         pygame.display.flip()
#         clock.tick(10)  # Slower speed to observe AI's moves

#         for event in pygame.event.get():
#             if event.type == pygame.QUIT:
#                 running = False

class PygameTetris(Env):
    def __init__(self, seed):
        self._init_rewards()
        self.random_seed = seed #TODO   
        # Screen dimensions with extra width for the preview
        self.SCALE=1
        self.PREVIEW_WIDTH = 25 * self.SCALE
        self.SCREEN_WIDTH = 50 * self.SCALE + self.PREVIEW_WIDTH
        self.SCREEN_HEIGHT = 100 * self.SCALE
        self.COLUMNS = 10
        self.ROWS = 20
        self.GRID_SIZE = 5 * self.SCALE

        self.grid = [[0 for _ in range(self.COLUMNS)] for _ in range(self.ROWS)]
        self.current_tetromino: Tetromino = Tetromino(random.choice(SHAPES), (self.ROWS, self.COLUMNS))
        self.next_tetromino: Tetromino = Tetromino(random.choice(SHAPES), (self.ROWS, self.COLUMNS))

        self.screen = pygame.display.set_mode((self.SCREEN_WIDTH, self.SCREEN_HEIGHT))

    
    def step(self, action: Actions):
        self.reset_screen()

        # Return values
        obs = pygame.surfarray.array3d(self.screen)
        reward = 0
        terminated = False
        # truncated = False

        #TODO: Action penalty?
        # reward -= self.action_penalty
        # elif action == Actions.NoAction:
        #     reward += self.action_penalty

        if not self.apply_action(action):
            reward -= 100

        if self.current_tetromino.is_colliding(self.grid, (1, 0)):
            self.current_tetromino.lock_on_grid(self.grid)
            lines_cleared = self.clear_lines(self.grid)
            height_placed = self.current_tetromino.y
            
            current_tetromino = next_tetromino
            next_tetromino = Tetromino(random.choice(SHAPES))

            if self.check_game_over(self.grid, current_tetromino):
                print("Game Over!")
                terminated = True
                reward -= 500
                return obs, reward, terminated
        else:
            self.current_tetromino.move_down()

        reward += self.step_reward
        reward += self.calculate_reward(lines_cleared, height_placed)


        self.render_screen()
        obs = pygame.surfarray.array3d(self.screen)
        

        # pygame.display.flip()
        # clock.tick(10)  # Slower speed to observe AI's moves

        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False

    def apply_action(self, action: Actions) -> bool:
        """Apply an action. Returns whether the action could be applied or not."""
        if action == Actions.MoveLeft:  # Move left
            self.current_tetromino.x -= 1
            if self.current_tetromino.is_colliding(self.grid):
                self.current_tetromino.x += 1
                return False
        elif action == Actions.MoveRight:  # Move right
            self.current_tetromino.x += 1
            if self.current_tetromino.is_colliding(self.grid):
                self.current_tetromino.x -= 1
                return False
        elif action == Actions.RotateClock:  # Rotate clockwise
            self.current_tetromino.rotate(self.grid)
            if self.current_tetromino.is_colliding(self.grid):
                self.current_tetromino.rotate(self.grid, clockwise=False)
                return False
        elif action == Actions.RotateCClock: # counterclockwise
            self.current_tetromino.rotate(self.grid, clockwise=False)
            if self.current_tetromino.is_colliding(self.grid):
                self.current_tetromino.rotate(self.grid)
                return False
        return True
    
    def calculate_reward(self, lines_cleared, height_placed):
        reward = 0
        reward += lines_cleared ** 2 * 50
        reward += (self.ROWS - height_placed) / 10
        reward -= self.count_holes() * 2
        reward -= self.calculate_bumpiness() * 0.5
        reward -= self.calculate_height() * 0.2
        '''# Penalty for height differences
        for i in range(COLUMNS - 1):
            height_diff = abs(sum(grid[j][i] for j in range(ROWS)) - sum(grid[j][i+1] for j in range(ROWS)))
            reward -= height_diff * 0.1
        '''
        return reward

    def clear_lines(self, grid):
        full_rows = [i for i, row in enumerate(grid) if all(row)]
        for i in full_rows:
            del grid[i]
            grid.insert(0, [0 for _ in range(self.COLUMNS)])
        return len(full_rows)

    def check_game_over(self, grid, tetromino):
        return tetromino.collision(grid, offset=(0, 0))


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

    def count_holes(self) -> int:
        """Returns the amount of holes."""
        holes_table = [[0 for _ in range(self.COLUMNS)] for _ in range(self.ROWS)]
        hole_count = 0
        for row in range(self.ROWS):
            # block_found = False
            for col in range(self.COLUMNS):
                if self.grid[row][col]:
                    continue
                left_open = True
                up_open = True
                if col > 0:
                    if self.grid[row][col - 1]:
                        left_open = False
                    elif holes_table[row][col - 1] > 0:
                        left_open = False
                        holes_table[row][col] = holes_table[row][col - 1]
                if row > 0:
                    if self.grid[row - 1][col] or holes_table[row - 1][col] > 0:
                        up_open = False
                if up_open: #opening on top found
                    holes_table[row][col] = 0
                if not up_open and not left_open:
                    holes_table[row][col] += 1
                    if col + 1 >= self.COLUMNS or self.grid[row][col + 1]:
                        hole_count += holes_table[row][col]
                    # elif :
                    #     hole_count += holes_table[row][col]
                    



                # if left_open and holes_table[col]:
                    
                # up_open = row < 0 # open over the screen
                # if not up_open:
                #     up_open = holes_table[col][row - 1] > 0 or self.grid[col][row - 1]
                
                # if self.grid[row][col]:
                #     block_found = True
                # elif block_found:
                #     if (row + 1 < len(self.grid) and self.grid[row + 1][col]) and (row - 1 > 0 and self.grid[row - 1][col]):
                #         # The check here (row + 1 < len()) is evaluated first, such that no wrong indexing takes place.
                #         holes += 1
        return hole_count

    def calculate_height(self):
        for row in range(self.ROWS):
            if any(self.grid[row]):
                return self.ROWS - row
        return 0

    def count_clearable_lines(grid):
        return sum(1 for row in grid if all(row))
    
    def reset_screen(self):
        self.screen.fill(BLACK)

        # Draw gray grid lines
        for x in range(self.COLUMNS + 1):
            pygame.draw.line(self.screen, GRAY, (x * self.GRID_SIZE, 0), (x * self.GRID_SIZE, self.SCREEN_HEIGHT))
        for y in range(self.ROWS + 1):
            pygame.draw.line(self.screen, GRAY, (0, y * self.GRID_SIZE), (self.SCREEN_WIDTH - self.PREVIEW_WIDTH, y * self.GRID_SIZE))

    def render_screen(self):
        self.current_tetromino.draw(self.screen)
        for y in range(self.ROWS):
            for x in range(self.COLUMNS):
                if self.grid[y][x]:
                    pygame.draw.rect(self.screen, BLUE, 
                                     pygame.Rect(x * self.GRID_SIZE, 
                                                 y * self.GRID_SIZE, 
                                                 self.GRID_SIZE, self.GRID_SIZE))
        self.draw_tetromino(self.next_tetromino.shape, self.COLUMNS + 1, 2, self.screen, GREEN)

    def draw_tetromino(self, shape, x, y, screen, color):
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

    def _init_rewards(self):
        self.action_penalty = 3e-4
        self.step_reward = 1e-3


# potential buggy things so far:
# penalty for making moves
# holes are defined too losely
# rotation only works in one direction
# No illegal move for rotation
# No penalty for illegal moves