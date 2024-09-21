import torch
from torch import nn
from torch.optim import Adam
from torch.amp import GradScaler, autocast
import numpy as np

from tqdm import tqdm
# import wandb

from generator import Generator
from models.TetrisConvModel import TetrisAgent
from environments.pygame_tetris import PygameTetris
from utils import get_batch_idx
from config import Config

from functools import partial

class PPO():
    def __init__(self, device, config: Config) -> None:
        self.init_hyperparameters(config)
        self.config=config

        environment_factory = partial(PygameTetris.get_environment, discrete_obs=False, render=False, scale=1)
        self.generator = Generator(num_environments=self.episodes_per_batch, max_timesteps_per_episode=self.max_timesteps_per_episode, environment_factory=environment_factory, gamma=self.gamma, device=device)
        channels, height, width = self.generator.get_observation_space()
        self.tetris_model = TetrisAgent(channels, height, width, self.generator.get_action_space()).to(device)
        
        self.optim = Adam(self.tetris_model.parameters(), lr=self.lr)
        self.scaler = GradScaler(device)

        self.iteration = 0

        self.batch_size = self.episodes_per_batch * self.max_timesteps_per_episode
        self.update_size = self.batch_size // self.num_mini_batch_updates
        self.mini_batch_size = self.update_size // self.num_sub_mini_batches
        self.device = device
        
    def train(self, total_timesteps):
        current_timesteps = 0
        self.iteration = 0
        while current_timesteps < total_timesteps:
            self.iteration += 1
            batch_obs, batch_actions, batch_log_probs, batch_rtgs, batch_rews, batch_lengths, batch_done_mask = self.generator.sample(self.tetris_model)

            step_sum = sum(batch_lengths)
            current_timesteps += step_sum

            with torch.no_grad():
                # Calculate V_{\phi,    k}(a, s)
                V, _ = self.evaluate(batch_obs, batch_actions)

                # Calculate advantage
                A_k = self._calc_advantages(batch_rews, V.detach(), batch_lengths)

                # Normalize advantage
                A_k = (A_k - A_k.mean()) / (A_k.std() + 1e-10)


            # update_size = step_sum // self.num_mini_batch_updates # floor division
            batch_idcs = np.arange(self.batch_size)

            print(f"=============\Iteration: {self.iteration}\ncurrent time steps: {current_timesteps}\n=============\n")
            print(f"episodic return: {torch.sum(batch_rews[batch_done_mask])}")
            for _ in tqdm(range(self.updates_per_iteration)):
                update_batch_idcs = np.random.choice(batch_idcs, (self.num_mini_batch_updates, self.update_size))
                for update_idcs in update_batch_idcs:
                    self.optim.zero_grad()

                    for start in range(0, self.update_size, self.mini_batch_size):
                        end = start + self.mini_batch_size
                        idcs = update_idcs[start:end]

                        mini_obs = batch_obs[idcs]
                        mini_actions = batch_actions[idcs]
                        mini_log_probs = batch_log_probs[idcs]
                        mini_rtgs = batch_rtgs[idcs]
                        mini_done_mask = batch_done_mask[idcs]
                        mini_A_k = A_k[idcs]

                        # with autocast(device_type='cuda', dtype=torch.float16):
                        V, log_probs = self.evaluate(mini_obs, mini_actions)

                        V = V[mini_done_mask]
                        log_probs = log_probs[mini_done_mask]
                        mini_log_probs = mini_log_probs[mini_done_mask]
                        mini_A_k = mini_A_k[mini_done_mask]


                        # Calculate loss for actor model
                        # \phi_{\theta}(a, s) / \phi_{\theta_{k}}(a, s)
                        prob_ratio = torch.exp(log_probs - mini_log_probs)
                        
                        # surrogate objective see https://spinningup.openai.com/en/latest/algorithms/ppo.html
                        # torch.where: takes a conditional as first param, then the result for true, false. 
                        clip = torch.where(mini_A_k >= 0, torch.min(prob_ratio, torch.tensor(1 + self.epsilon, dtype=prob_ratio.dtype, device=self.device)), torch.max(prob_ratio, torch.tensor(1 - self.epsilon, dtype=prob_ratio.dtype, device=self.device)))

                        # Calculate Losses
                        actor_loss = (-clip * mini_A_k).mean() # negative, such that advantage is maximized
                        value_loss = nn.MSELoss()(mini_rtgs[mini_done_mask], V)

                        loss = actor_loss  + value_loss # maximize actor_loss and minimize value_loss

                        loss /= self.num_sub_mini_batches

                        self.scaler.scale(loss).backward(retain_graph=False)
                    

                    self.scaler.unscale_(self.optim)
                    torch.nn.utils.clip_grad_norm_(self.tetris_model.parameters(), max_norm=0.5)
                    
                    self.scaler.step(self.optim)
                    self.scaler.update()


                    # self.actor_optim.zero_grad()
                    # actor_loss.backward(retain_graph=True)
                    # self.actor_optim.step()
                    # self.critic_optim.zero_grad()
                    # critic_loss.backward()
                    # self.critic_optim.step()

    
    def _calc_advantages(self, rewards, values, episode_lengths):
        """Calculate Generalized Advantage Estimate (GAE), see https://arxiv.org/abs/1506.02438"""
        advantages = torch.full((self.batch_size,), 0, device=self.device, dtype=torch.float)
        for ep in range(self.episodes_per_batch):
            length = episode_lengths[ep]
            last_advantage = 0
            for t in reversed(range(length)):
                idx_t = get_batch_idx(self.max_timesteps_per_episode, ep, t)
                if t+1 == length:
                    delta = rewards[idx_t] - values[idx_t]
                else:
                    delta = rewards[idx_t] + self.gamma * values[idx_t+1] - values[idx_t] #reward + TD residual, which is V(s_{t+1}) - V(s_{t}), except for the "latest" time step (the first to come, as the steps are inversed)
                advantage = delta + self.lam * self.gamma * last_advantage
                last_advantage = advantage
                advantages[idx_t] = advantage

        return advantages
    

    def evaluate(self, batch_obs, batch_acts):
        pi, v = self.tetris_model(batch_obs)
        v = v.squeeze()
        # V = self.critic(batch_obs).squeeze()
        # log_probs = self.actor(batch_obs)
        pi = pi[torch.arange(pi.size(0)), batch_acts] # select only probs of actions taken. torch.arange(...) selects all rows, batch_acts selects the appropriate actions for each row (timestep over different episodes).
        return v, pi

    
    def init_hyperparameters(self, config: Config):
        """
        Hyperparameters for the learning process.
        Parameters:
            episodes_per_batch(int): Creates that many environments to sample data from. Each environment corresponds to an episode with up to max_timesteps_per_episode steps. Environments will be reset only once they terminate.
            max_timesteps_per_episode(int): The amount of timesteps take in each environment for a batch. If an environment terminates before, the steps may be less.
            updates_per_iteration(int): Each batch is a rollout of samples from each environment. Through the PPO objective, after each rollout, multiple updates can be applied. updates_per_iteration specifies how many updates are applied.
            num_mini_batch_updates(int): The amount of mini-batches. For each mini-batch, the loss will be backpropagated through the model to update weights.
            num_sub_mini_batches(int): The number of sub-batches for each mini-batch to control memory usage. This doesn't influence the learning process but trades off speed vs. memory.

        A batch size will have the shape: (episodes_per_batch * max_timesteps_per_episode, *(obs)).
        A mini-batch will have the shape: (episodes_per_batch * max_timestpes_per_episode / num_mini_batch_updates, *(obs)).
        A sub-mini-batch (that will be passed in the model) will have the shape: (episodes_per_batch * max_timestpes_per_episode / (num_mini_batch_updates * num_sub_min_batches), *(obs)).
        Sizes are rounded down. In order to avoid index-error, (episodes_per_batch * max_timesteps_per_episode) % (num_mini_batch_updates * num_sub_min_batches) should be = 0.
        """
        assert(((config.episodes_per_batch * config.max_timesteps_per_episode) % (config.num_mini_batch_updates * config.num_sub_mini_batches)) == 0)
        self.episodes_per_batch = config.episodes_per_batch
        self.max_timesteps_per_episode = config.max_timesteps_per_episode
        self.updates_per_iteration = config.updates_per_iteration 
        self.num_mini_batch_updates = config.num_mini_batch_updates # updates per batch
        self.num_sub_mini_batches = config.num_sub_mini_batches # To control memory usage

        self.gamma = config.gamma # Used in rewards to go
        self.epsilon = config.epsilon # PPO clipping objective
        self.lam = config.lam# value is following https://arxiv.org/abs/1506.0243

        # self.actor_lr = actor_lr
        # self.critic_lr = cricit_lr
        self.lr = config.lr

    def close(self):
        self.generator.close()