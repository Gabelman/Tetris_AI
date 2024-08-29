import pygame
import random
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from collections import deque
import pickle

# Initialize Pygame
pygame.init()

# Screen dimensions with extra width for the preview
PREVIEW_WIDTH = 150
SCREEN_WIDTH = 300 + PREVIEW_WIDTH
SCREEN_HEIGHT = 600
COLUMNS = 10
ROWS = 20
GRID_SIZE = 30

# Colors
BLACK = (0, 0, 0)
WHITE = (255, 255, 255)
RED = (255, 0, 0)
GREEN = (0, 255, 0)
BLUE = (0, 0, 255)

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

    def rotate(self):
        self.shape = [list(row) for row in zip(*self.shape[::-1])]

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

# AI implementation
class TetrisAI(nn.Module):
    def __init__(self):
        super(TetrisAI, self).__init__()
        self.fc1 = nn.Linear(COLUMNS * ROWS + 4, 64)
        self.fc2 = nn.Linear(64, 32)
        self.fc3 = nn.Linear(32, 4)  # 4 possible actions: left, right, rotate, do nothing

    def forward(self, x):
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        return self.fc3(x)

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
        state = np.append(state, [tetromino.x, tetromino.y, len(tetromino.shape), len(tetromino.shape[0])])
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

def calculate_reward(grid, lines_cleared, height_placed):
    reward = 0
    
    # Reward for placing a piece
    reward += 1
    
    # Reward based on lines cleared
    reward += lines_cleared ** 2 * 50
    
    # Reward for lower placements
    reward += (ROWS - height_placed) / 10
    
    # Penalty for height differences
    for i in range(COLUMNS - 1):
        height_diff = abs(sum(grid[j][i] for j in range(ROWS)) - sum(grid[j][i+1] for j in range(ROWS)))
        reward -= height_diff * 0.1
    
    return reward

def train_ai(continue_training=True):
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
                current_tetromino.rotate()
                if current_tetromino.collision(grid):
                    current_tetromino.rotate()
                    current_tetromino.rotate()
                    current_tetromino.rotate()

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
                    reward -= 50  # Larger penalty for game over
            else:
                reward = -0.01  # Small penalty for each move

            next_state = agent.get_state(grid, current_tetromino)
            agent.remember(state, action, reward, next_state, game_over)
            agent.replay()

        if episode % update_target_every == 0:
            agent.update_target_model()

        if episode % 10 == 0:  # Save every 10 episodes
            torch.save(agent.model.state_dict(), "tetris_ai_model.pth")
            agent.save_memory()
            print(f"Episode: {episode}, Total Reward: {round(total_reward)}, Lines Cleared: {lines_cleared}, Epsilon: {agent.epsilon:.2f}")

    # Final save
    torch.save(agent.model.state_dict(), "tetris_ai_model.pth")
    agent.save_memory()

def play_ai():
    screen = pygame.display.set_mode((SCREEN_WIDTH, SCREEN_HEIGHT))
    clock = pygame.time.Clock()

    ai_model = TetrisAI()
    ai_model.load_state_dict(torch.load("tetris_ai_model.pth"))
    ai_model.eval()
    agent = DQNAgent()
    agent.model = ai_model

    grid = [[0 for _ in range(COLUMNS)] for _ in range(ROWS)]
    current_tetromino = Tetromino(random.choice(SHAPES))
    next_tetromino = Tetromino(random.choice(SHAPES))
    running = True

    while running:
        screen.fill(BLACK)
        
        state = agent.get_state(grid, current_tetromino)
        action = agent.act(state)

        if action == 0:  # Move left
            current_tetromino.x -= 1
            if current_tetromino.collision(grid):
                current_tetromino.x += 1
        elif action == 1:  # Move right
            current_tetromino.x += 1
            if current_tetromino.collision(grid):
                current_tetromino.x -= 1
        elif action == 2:  # Rotate
            current_tetromino.rotate()
            if current_tetromino.collision(grid):
                current_tetromino.rotate()
                current_tetromino.rotate()
                current_tetromino.rotate()

        current_tetromino.y += 1
        if current_tetromino.collision(grid):
            current_tetromino.y -= 1
            current_tetromino.lock(grid)
            lines_cleared = clear_lines(grid)
            
            current_tetromino = next_tetromino
            next_tetromino = Tetromino(random.choice(SHAPES))

            if check_game_over(grid, current_tetromino):
                print("Game Over!")
                running = False

        current_tetromino.draw(screen)
        for y in range(ROWS):
            for x in range(COLUMNS):
                if grid[y][x]:
                    pygame.draw.rect(screen, BLUE, 
                                     pygame.Rect(x * GRID_SIZE, 
                                                 y * GRID_SIZE, 
                                                 GRID_SIZE, GRID_SIZE))
        draw_tetromino(next_tetromino.shape, COLUMNS + 1, 2, screen, GREEN)

        pygame.display.flip()
        clock.tick(10)  # Slower speed to observe AI's moves

        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False

    pygame.quit()

if __name__ == "__main__":
    try:
        choice = input("Enter 'train' to train the AI or 'play' to watch the AI play: ")
        if choice.lower() == 'train':
            train_ai(continue_training=True)
        elif choice.lower() == 'play':
            play_ai()
        else:
            print("Invalid choice. Please enter 'train' or 'play'.")
    except EOFError:
        print("No input received. Defaulting to training mode.")
        train_ai(continue_training=True)
    except KeyboardInterrupt:
        print("\nProgram interrupted by user. Exiting.")
    except Exception as e:
        print(f"An unexpected error occurred: {e}")
