import pygame
import random
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from collections import deque
import pickle

from tetris import PygameTetris, Actions

# Initialize Pygame
pygame.init()

# Screen dimensions with extra width for the preview
SCALE=5
PREVIEW_WIDTH = 25 * SCALE
SCREEN_WIDTH = 50 * SCALE + PREVIEW_WIDTH
SCREEN_HEIGHT = 100 * SCALE
COLUMNS = 10
ROWS = 20
GRID_SIZE = 5 * SCALE

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
    def __init__(self, shape):
        self.shape = shape
        self.x = COLUMNS // 2 - len(shape[0]) // 2
        self.y = 0
        self.rotation_state = 0

    def rotate(self, grid):
        if len(self.shape) == 2 and len(self.shape[0]) == 2:  # O piece doesn't rotate
            return

        original_x = self.x
        original_y = self.y
        original_shape = self.shape

        # Perform rotation
        rotated = [list(row) for row in zip(*self.shape[::-1])]  # Clockwise rotation
        
        # Check if rotation is possible
        self.shape = rotated
        if self.collision(grid):
            # If collision occurs, revert the rotation
            self.shape = original_shape
            self.x = original_x
            self.y = original_y
        else:
            self.rotation_state = (self.rotation_state + 1) % 4

    def collision(self, grid, offset=(0, 0)):
        off_x, off_y = offset
        for y, row in enumerate(self.shape):
            for x, cell in enumerate(row):
                if cell:
                    new_x = self.x + x + off_x
                    new_y = self.y + y + off_y
                    if new_x < 0 or new_x >= COLUMNS or new_y >= ROWS:
                        return True
                    if new_y >= 0 and grid[new_y][new_x]:
                        return True
        return False

    def draw(self, screen):
        for y, row in enumerate(self.shape):
            for x, cell in enumerate(row):
                if cell:
                    pygame.draw.rect(screen, RED, 
                                     pygame.Rect((self.x + x) * GRID_SIZE, 
                                                 (self.y + y) * GRID_SIZE, 
                                                 GRID_SIZE, GRID_SIZE))

    def lock(self, grid):
        for y, row in enumerate(self.shape):
            for x, cell in enumerate(row):
                if cell:
                    grid[self.y + y][self.x + x] = 1

def clear_lines(grid):
    full_rows = [i for i, row in enumerate(grid) if all(row)]
    for i in full_rows:
        del grid[i]
        grid.insert(0, [0 for _ in range(COLUMNS)])
    return len(full_rows)


def check_game_over(grid, tetromino):
    return tetromino.collision(grid, offset=(0, 0))

def draw_tetromino(shape, x, y, screen, color):
    for row_idx, row in enumerate(shape):
        for col_idx, cell in enumerate(row):
            if cell:
                pygame.draw.rect(
                    screen,
                    color,
                    pygame.Rect(
                        (x + col_idx) * GRID_SIZE,
                        (y + row_idx) * GRID_SIZE,
                        GRID_SIZE,
                        GRID_SIZE
                    )
                )

def calculate_bumpiness(grid):
        heights = [0] * COLUMNS
        for col in range(COLUMNS):
            for row in range(ROWS):
                if grid[row][col]:
                    heights[col] = ROWS - row
                    break
        
        bumpiness = 0
        for i in range(COLUMNS - 1):
            bumpiness += abs(heights[i] - heights[i+1])
        
        return bumpiness

def count_holes(grid):
    holes = 0
    for col in range(COLUMNS):
        block_found = False
        for row in range(ROWS):
            if grid[row][col]:
                block_found = True
            elif block_found:
                holes += 1
    return holes

def calculate_height(grid):
    for row in range(ROWS):
        if any(grid[row]):
            return ROWS - row
    return 0

def count_clearable_lines(grid):
    return sum(1 for row in grid if all(row))

# AI implementation
class TetrisAI(nn.Module):
    def __init__(self):
        super(TetrisAI, self).__init__()
        input_size = COLUMNS * ROWS + 8  # +8 for x, y, shape dimensions, bumpiness, holes, height, clearable_lines
        self.fc1 = nn.Linear(input_size, 128)
        self.fc2 = nn.Linear(128, 64)
        self.fc3 = nn.Linear(64, 32)
        self.fc4 = nn.Linear(32, 4) # 4 possible actions: left, right, rotate, do nothing

    def forward(self, x):
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        x = torch.relu(self.fc3(x))
        return self.fc4(x)

class DQNAgent:
    def __init__(self):
        self.model = TetrisAI()
        self.target_model = TetrisAI()
        self.target_model.load_state_dict(self.model.state_dict())
        self.optimizer = optim.Adam(self.model.parameters())
        self.loss_fn = nn.MSELoss()
        self.memory = deque(maxlen=10000)
        self.batch_size = 32
        self.gamma = 0.99
        self.epsilon = 1.0
        self.epsilon_min = 0.01
        self.epsilon_decay = 0.995


    def get_state(self, grid, tetromino):
        state = np.array(grid).flatten()
        holes = count_holes(grid)
        bumpiness = calculate_bumpiness(grid)
        height = calculate_height(grid)
        lines_clearable = count_clearable_lines(grid)
        state = np.append(state, [tetromino.x, tetromino.y, len(tetromino.shape), len(tetromino.shape[0]), holes, bumpiness, height, lines_clearable]) 
        return torch.FloatTensor(state)

    def act(self, state):
        if random.random() <= self.epsilon:
            return random.randint(0, 3)
        with torch.no_grad():
            q_values = self.model(state)
            return torch.argmax(q_values).item()

    def remember(self, state, action, reward, next_state, done):
        self.memory.append((state, action, reward, next_state, done))

    def replay(self):
        if len(self.memory) < self.batch_size:
            return

        batch = random.sample(self.memory, self.batch_size)
        states, actions, rewards, next_states, dones = zip(*batch)

        states = torch.stack(states)
        next_states = torch.stack(next_states)
        actions = torch.LongTensor(actions)
        rewards = torch.FloatTensor(rewards)
        dones = torch.FloatTensor(dones)

        current_q = self.model(states).gather(1, actions.unsqueeze(1))
        next_q = self.target_model(next_states).max(1)[0].detach()
        target_q = rewards + (1 - dones) * self.gamma * next_q

        loss = self.loss_fn(current_q.squeeze(), target_q)
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        if self.epsilon > self.epsilon_min:
            self.epsilon *= self.epsilon_decay

    def update_target_model(self):
        self.target_model.load_state_dict(self.model.state_dict())

    def save_memory(self, filename="replay_memory.pkl"):
        with open(filename, "wb") as f:
            pickle.dump(list(self.memory), f)

    def load_memory(self, filename="replay_memory.pkl"):
        try:
            with open(filename, "rb") as f:
                loaded_memory = pickle.load(f)
                self.memory = deque(loaded_memory, maxlen=self.memory.maxlen)
            print(f"Loaded {len(self.memory)} experiences from memory.")
        except FileNotFoundError:
            print("No existing memory file found.")

def train_ai(continue_training=False):
    agent = DQNAgent()
    
    if continue_training:
        try:
            agent.model.load_state_dict(torch.load("tetris_ai_model.pth"))
            agent.target_model.load_state_dict(torch.load("tetris_ai_model.pth"))
            agent.load_memory()
            print("Loaded existing model and memory for continued training.")
        except FileNotFoundError:
            print("No existing model or memory found. Starting training from scratch.")
    
    num_episodes = 1000
    update_target_every = 100

    for episode in range(num_episodes):
        grid = [[0 for _ in range(COLUMNS)] for _ in range(ROWS)]
        current_tetromino = Tetromino(random.choice(SHAPES))
        next_tetromino = Tetromino(random.choice(SHAPES))
        game_over = False
        total_reward = 0
        lines_cleared = 0  # Initialize lines cleared for this episode

        while not game_over:
            state = agent.get_state(grid, current_tetromino)
            action = agent.act(state)

            # Apply action
            if action == 0:  # Move left
                current_tetromino.x -= 1
                if current_tetromino.collision(grid):
                    current_tetromino.x += 1
            elif action == 1:  # Move right
                current_tetromino.x += 1
                if current_tetromino.collision(grid):
                    current_tetromino.x -= 1
            elif action == 2:  # Rotate
                current_tetromino.rotate(grid)

            # Move tetromino down
            current_tetromino.y += 1
            if current_tetromino.collision(grid):
                current_tetromino.y -= 1
                height_placed = current_tetromino.y
                current_tetromino.lock(grid)
                lines_cleared += clear_lines(grid)  # Count lines cleared
                reward = calculate_reward(grid, lines_cleared, height_placed)
                total_reward += reward

                current_tetromino = next_tetromino
                next_tetromino = Tetromino(random.choice(SHAPES))

                if check_game_over(grid, current_tetromino):
                    game_over = True
                    reward -= 500  # Larger penalty for game over
            else:
                reward = -0.01  # Small penalty for each move

            next_state = agent.get_state(grid, current_tetromino)
            agent.remember(state, action, reward, next_state, game_over)
            agent.replay()

        if episode % update_target_every == 0:
            agent.update_target_model()

        if episode % 1 == 0:  # Save every 1 episodes
            torch.save(agent.model.state_dict(), "tetris_ai_model.pth")
            agent.save_memory()
            print(f"Episode: {episode}, Total Reward: {round(total_reward)}, Lines Cleared: {lines_cleared}, Epsilon: {agent.epsilon:.2f}")

    # Final save
    torch.save(agent.model.state_dict(), "tetris_ai_model.pth")
    agent.save_memory()

def play_ai(human_player=True):
    game = PygameTetris(0, render=True, scale=6)
    FPS = 5
    # screen = pygame.display.set_mode((SCREEN_WIDTH, SCREEN_HEIGHT))
    clock = pygame.time.Clock()

    obs = game.reset()
    if not human_player:

        ai_model = TetrisAI()
        ai_model.load_state_dict(torch.load("tetris_ai_model.pth"))
        ai_model.eval()
        agent = DQNAgent()
        agent.model = ai_model

    # grid = [[0 for _ in range(COLUMNS)] for _ in range(ROWS)]
    # current_tetromino = Tetromino(random.choice(SHAPES))
    # next_tetromino = Tetromino(random.choice(SHAPES))
    running = True
    clock.tick(FPS)

    while running:
        action = Actions.NoAction
        if human_player:
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
        else:
            # state = agent.get_state(grid, current_tetromino)
            # action = agent.act(state)
            pass

        obs, reward, terminated = game.step(action)

        clock.tick(FPS)  # Slower speed to observe AI's moves

        # for event in pygame.event.get():
        #     if event.type == pygame.QUIT:
        #         running = False

    pygame.quit()

if __name__ == "__main__":
    play_ai()
    # try:
    #     choice = input("Enter 'train' to train the AI or 'play' to watch the AI play: ")
    #     if choice.lower() == 'train':
    #         train_ai(continue_training=True)
    #     elif choice.lower() == 'play':
    #         play_ai()
    #     else:
    #         print("Invalid choice. Please enter 'train' or 'play'.")
    # except EOFError:
    #     print("No input received. Defaulting to training mode.")
    #     train_ai(continue_training=True)
    # except KeyboardInterrupt:
    #     print("\nProgram interrupted by user. Exiting.")
    # except Exception as e:
    #     print(f"An unexpected error occurred: {e}")
