import pygame
import random
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from collections import deque
import pickle
import wandb

from environments.pygame_tetris import PygameTetris, play_pygame
from models.tetris_discrete_model import TetrisAI
from generator import Generator

from functools import partial

# Initialize Pygame
pygame.init()

export_path = "exports/"

class DQNAgent:
    def __init__(self):
        self.environment = PygameTetris(seed=0)
        self.input_size = self.environment.observation_space  # new
        self.action_space = self.environment.action_space  # new
        self.model = TetrisAI(self.input_size, self.action_space)
        self.target_model = TetrisAI(self.input_size, self.action_space)
        self.target_model.load_state_dict(self.model.state_dict())

        # new: check for gpu and set device
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        # new: move models to the specified device
        self.model.to(self.device)
        self.target_model.to(self.device)

        # wandb initialization
        wandb.init(project="Tetris_AI", name="Tetris_DQN", config={
            "batch_size": 32,
            "gamma": 0.99,
            "epsilon_decay": 0.995,
            "learning_rate": 0.001,
            "num_episodes": 1000,})
        self.model = TetrisAI(self.input_size, self.action_space)
        self.target_model = TetrisAI(self.input_size, self.action_space)
        self.target_model.load_state_dict(self.model.state_dict())

        # torch stuff
        self.optimizer = optim.Adam(self.model.parameters())
        self.loss_fn = nn.MSELoss()

        # train parameters
        self.memory = deque(maxlen=10000)
        self.batch_size = 32
        self.gamma = 0.99
        self.epsilon = 1.0
        self.epsilon_min = 0.01
        self.epsilon_decay = 0.995

        # environments
        # environment_factory = partial(PygameTetris.get_environment, discrete_obs=True)
        # self.generator = Generator(num_environments=1, max_timesteps_per_episode=100, environment_factory=environment_factory)
        # seed = 0

    def train(self, continue_training=False):
        
        if continue_training:
            try:
                self.model.load_state_dict(torch.load(export_path + "tetris_ai_model.pth"))
                self.target_model.load_state_dict(torch.load(export_path + "tetris_ai_model.pth"))
                self.load_memory()
                print("Loaded existing model and memory for continued training.")
            except FileNotFoundError:
                print("No existing model or memory found. Starting training from scratch.")
        
        num_episodes = 1000
        update_target_every = 100

        for episode in range(num_episodes):
            game_over = False
            total_reward = 0
            steps = 0
            
            obs = self.environment.reset()

            while not game_over:
                steps += 1
                action = self.sample_action(obs)
                next_obs, reward, terminated, _ = self.environment.step(action)
                
                self.remember(obs, action, reward, next_obs, terminated)
                self.replay()
                obs = next_obs

                total_reward += reward
                game_over = terminated

            if episode % update_target_every == 0:
                self.update_target_model()

            if episode % 1 == 0:  # Save every 1 episodes
                torch.save(self.model.state_dict(), export_path + "tetris_ai_model.pth")
                self.save_memory()
                # TODO: reimplement getting information from the environment. See: lines_cleared
                # print(f"Episode: {episode}, Total Reward: {round(total_reward)}, Lines Cleared: {lines_cleared}, Epsilon: {agent.epsilon:.2f}")
                print(f"Episode: {episode}, Total Reward: {round(total_reward)}, Epsilon: {self.epsilon:.2f}")

            wandb.log({
                "episode": episode,
                "total_reward": total_reward,
                "epsilon": self.epsilon,
                "game_time": steps,  # Game time based on steps
            })

            print(f"Episode: {episode}, Total Reward: {round(total_reward)}, Game Time: {steps}, Epsilon: {self.epsilon:.2f}")

        # Final save
        torch.save(self.model.state_dict(), export_path + "tetris_ai_model.pth")
        self.save_memory()

    def sample_action(self, obs):
        # new: convert obs to torch from numpy
        obs = torch.tensor(obs, dtype=torch.float32).to(self.device)
        #if torch.rand(1).item() < self.epsilon:
        #    return torch.tensor(self.environment.sample_action()) 
        #else:
        #    q_values = self.model(obs)
        #    return torch.argmax(q_values).item()

    
        if random.random() <= self.epsilon:
            return random.randint(0, 3)
        with torch.no_grad():
            q_values = self.model(obs)
            return torch.argmax(q_values).item()

    def remember(self, state, action, reward, next_state, done):
        self.memory.append((state, action, reward, next_state, done))

    def replay(self):
        if len(self.memory) < self.batch_size:
            return

        batch = random.sample(self.memory, self.batch_size)

        states, actions, rewards, next_states, dones = zip(*batch)
        states = [torch.as_tensor(state) for state in states]  # new
        next_states = [torch.as_tensor(next_state) for next_state in next_states]  # new
        states = torch.stack(states)
        next_states = torch.stack(next_states)
        actions = torch.LongTensor(actions)
        rewards = torch.FloatTensor(rewards)
        dones = torch.FloatTensor(dones)
        states = states.float()  # new: s.t. input and weights are same dtype (float)
        next_states = next_states.float()  # new
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



if __name__ == "__main__":
    play_pygame(None)
    
