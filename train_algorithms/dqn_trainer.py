import pygame
import random
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from collections import deque
import pickle

from Environments.pygame_tetris import PygameTetris, Actions
from models.tetris_discrete_model import TetrisAI

# Initialize Pygame
pygame.init()

export_path = "exports/"

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

    def save_memory(self, filename=export_path + "replay_memory.pkl"):
        with open(filename, "wb") as f:
            pickle.dump(list(self.memory), f)

    def load_memory(self, filename=export_path + "replay_memory.pkl"):
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
            agent.model.load_state_dict(torch.load(export_path + "tetris_ai_model.pth"))
            agent.target_model.load_state_dict(torch.load(export_path + "tetris_ai_model.pth"))
            agent.load_memory()
            print("Loaded existing model and memory for continued training.")
        except FileNotFoundError:
            print("No existing model or memory found. Starting training from scratch.")
    
    num_episodes = 1000
    update_target_every = 100

    game = PygameTetris()
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
            torch.save(agent.model.state_dict(), export_path + "tetris_ai_model.pth")
            agent.save_memory()
            print(f"Episode: {episode}, Total Reward: {round(total_reward)}, Lines Cleared: {lines_cleared}, Epsilon: {agent.epsilon:.2f}")

    # Final save
    torch.save(agent.model.state_dict(), "tetris_ai_model.pth")
    agent.save_memory()

def play_ai(human_player=True):
    game = PygameTetris(0, discrete_obs=False, render=True, scale=6)
    FPS = 64

    clock = pygame.time.Clock()

    obs = game.reset()
    if not human_player:

        ai_model = TetrisAI()
        ai_model.load_state_dict(torch.load("tetris_ai_model.pth"))
        ai_model.eval()
        agent = DQNAgent()
        agent.model = ai_model

    running = True
    clock.tick(FPS)
    frame_count = 0

    while running:
        frame_count += 1
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
                    elif event.key == pygame.K_DOWN:
                        action = Actions.MoveDown
                    game.apply_action(action)
                    game.render_screen()
        else:
            # state = agent.get_state(grid, current_tetromino)
            # action = agent.act(state)
            pass
        
        if frame_count % (FPS * 3) == 0:
            obs, reward, terminated = game.step(action)
            print("obs: ")
            print(obs)
            print(f"reward: {reward}")


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
