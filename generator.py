import torch
from tqdm.auto import tqdm
import gymnasium as gym

from torch.distributions import Categorical
import numpy as np
import wandb

from enum import Enum

from utils import get_batch_idx
from models.TetrisConvModel import TetrisAgent

from environments.environment import Env
from typing import Callable


class Generator():
    def __init__(self, num_environments, max_timesteps_per_episode, environment_factory: Callable[[int, bool, bool, int], Env], gamma, device):
        """
        Creates a Generator class to sample observations from environments.
        Arguments:
            num_environments(int): Number of environments to create and sample from.
            max_timesteps_per_episode(int): Number of timesteps done on each environment for a sample.
            environment_factory(Callable[[int]]): Takes a partial function from environments.environment get_environment(), such that only seed must be input.
            gamma(float): Value to control weight of accumulated reward (reward to go).
            device(str): Device onto which tensors are loaded.
        """
        self.max_timesteps_per_episode = max_timesteps_per_episode
        self.num_environments = num_environments

        self.environment_seeds = [np.random.randint(0, 2**31) for _ in range(num_environments)]
        self.environments: list[Env] = [environment_factory(seed) for seed in self.environment_seeds]
        self.last_observations = [env.reset() for env in self.environments]
        self.action_space = self.environments[0].action_space
        self.observation_space = self.environments[0].observation_space
        
        self.environments_done = [False for _ in range(num_environments)]


        self.device = device
        self.gamma = gamma # reward calculation


    def sample(self, model: TetrisAgent):
        batch_obs = torch.full((self.num_environments * self.max_timesteps_per_episode, *self.observation_space), -1, dtype=torch.float, device=self.device, requires_grad=False) # Batch Observations. (num_episodes * episode_length, observation_shape)
        batch_log_probs = torch.full((self.num_environments * self.max_timesteps_per_episode,), 0, dtype=torch.float, device=self.device, requires_grad=False) # (num_episodes * episode_length)
        batch_actions = torch.full((self.num_environments * self.max_timesteps_per_episode,), 0, dtype=torch.int, device=self.device, requires_grad=False) # (num_episodes * episode_length, action_space). 0 action is doing nothing
        batch_rewards = torch.full((self.num_environments * self.max_timesteps_per_episode,), 0, dtype=torch.float, device=self.device, requires_grad=False) # (num episodes, episode_length)
        # batch_values = torch.full(self.num_environments, self.max_timesteps_per_episode, 0, dtype=torch.float)
        batch_rewards_to_go = torch.full((self.num_environments * self.max_timesteps_per_episode,), 0, dtype=torch.float, device=self.device, requires_grad=False) # (num_episodes * episode_length)
        batch_done_mask = torch.full((self.num_environments * self.max_timesteps_per_episode,), False, device=self.device, requires_grad=False)
        batch_episode_lengths = []

        game_done_lengths = []
        
        model.eval()

        # print(f"--------------------\nSampling for iteration {self.iteration}\n--------------------\n")
        for i in tqdm(range(self.num_environments)):
            current_env: Env = self.environments[i]

            if self.environments_done[i]:
                obs = current_env.reset()
            else:
                obs = self.last_observations[i]

            # iterator = tqdm(range(self.max_timesteps_per_episode))
            for t_ep in tqdm(range(self.max_timesteps_per_episode), leave=False):
                idx = get_batch_idx(self.max_timesteps_per_episode, i, t_ep) # In order to insert values into "flattened" tensors immediately
                with torch.no_grad():
                    batch_done_mask[idx] = True

                    obs = self.obs_to_tensor(obs)
                    
                    pi = model.get_pis(obs)
                    action, log_prob = self.sample_action(pi)

                    batch_obs[idx] = obs
                    batch_actions[idx] = action
                    batch_log_probs[idx] = log_prob

                    obs, reward, done, _ = current_env.step(action)
                    self.environments_done[i] = done

                    batch_rewards[idx] = reward

                self.last_observations[i] = obs
                if self.environments_done[i]:
                    game_done_lengths.append(self.environments[i].get_game_length)
                    # iterator.close()
                    break
            batch_episode_lengths.append(t_ep + 1)

        if len(game_done_lengths) > 0:
            wandb.log({"average_game_lengths": sum(game_done_lengths)/len(game_done_lengths)})

        batch_rewards_to_go = self._calc_rewards_to_go(batch_rewards, batch_episode_lengths)
        # batch_advantages = self._calc_advantages(batch_rewards, batch_values, batch_episode_lengths)
        return batch_obs, batch_actions, batch_log_probs, batch_rewards_to_go, batch_rewards, batch_episode_lengths, batch_done_mask
    
    
    def _calc_rewards_to_go(self, batch_rewards, episode_lengths):
        batch_rtgs = torch.full((self.num_environments * self.max_timesteps_per_episode,), -float("inf"), device=self.device, dtype=torch.float)
        for ep_idx in range(self.num_environments): # Go through last episode first, such that the order in "flattened" batch_rtgs is consistent with episode order
            current_rtg = 0

            for t_ep in reversed(range(episode_lengths[ep_idx])): # Go back in time: approximation for Q(a, s), through Bellman equation, which is r(a, s) + \gamma Q(a', s')
                current_idx = get_batch_idx(self.max_timesteps_per_episode, ep_idx, t_ep)
                rtg = batch_rewards[current_idx] + self.gamma * current_rtg
                batch_rtgs[current_idx] = rtg

        return batch_rtgs
    
    def get_action_space(self):
        return self.action_space
    
    def get_observation_space(self):
        return self.observation_space

    def obs_to_tensor(self, obs):
        return torch.tensor(obs).to(self.device, dtype=torch.float)

    @staticmethod
    def sample_action(logits):
        pi = Categorical(logits=logits)
        # print(f"cat logits: {pi.logits}")
        a = pi.sample()
        log_prob = pi.log_prob(a)
        return a.item(), log_prob.item()

    def close(self):
        for e in self.environments:
            e.close()