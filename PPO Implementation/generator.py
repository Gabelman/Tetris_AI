import torch
from tqdm import tqdm
import gymnasium as gym

from torch.distributions import Categorical

class Generator():
    def __init__(self, num_environments, max_timesteps_per_episode, observation_space, gamma):
        self.observation_space = observation_space
        self.max_timesteps_per_episode = max_timesteps_per_episode
        self.num_environments = num_environments
        self.environments = [gym.make("ALE/Tetris-v5") for _ in range(num_environments)]
        self.environments_done = [True for _ in range(num_environments)]
        self.last_observations = []

        self.gamma = gamma # reward calculation

    def sample(self, model):
        batch_obs = torch.full((self.num_environments * self.max_timesteps_per_episode, *self.observation_space), -1, dtype=torch.float, device=self.device, requires_grad=False) # Batch Observations. (num_episodes * episode_length, observation_shape)
        batch_log_probs = torch.full((self.num_environments * self.max_timesteps_per_episode,), 0, dtype=torch.float, device=self.device, requires_grad=False) # (num_episodes * episode_length)
        batch_actions = torch.full((self.num_environments * self.max_timesteps_per_episode,), -1, dtype=torch.int, device=self.device, requires_grad=False) # (num_episodes * episode_length, action_space)
        batch_rewards = torch.full((self.num_environments * self.max_timesteps_per_episode,), 0, dtype=torch.float, device=self.device, requires_grad=False) # (num episodes, episode_length)
        # batch_values = torch.full(self.num_environments, self.max_timesteps_per_episode, 0, dtype=torch.float)
        batch_rewards_to_go = torch.full((self.num_environments * self.max_timesteps_per_episode,), 0, dtype=torch.float, device=self.device, requires_grad=False) # (num_episodes * episode_length)
        batch_done_mask = torch.full((self.num_environments * self.max_timesteps_per_episode,), False, device=self.device, requires_grad=False)
        batch_episode_lengths = []

        #reshape obs-space from (widht, height, channels) to (channels, height, width) (keeping batch size the same)
        batch_obs = torch.einsum("nwhc->nchw", batch_obs)
        # print(f"--------------------\nSampling for iteration {self.iteration}\n--------------------\n")
        for i in tqdm(range(self.num_environments)):
            current_env = self.environments[i]

            if self.environments_done[i]:
                obs, _ = current_env.reset()
            else:
                obs = self.last_observations[i]

            for t_ep in tqdm(range(self.max_timesteps_per_episode)):
                self.last_observations[i] = obs
                idx = self.get_batch_idx(i, t_ep) # In order to insert values into "flattened" tensors immediately
                with torch.no_grad():
                    batch_done_mask[idx] = True

                    obs = torch.tensor(obs).to(self.device, dtype=torch.float)
                    obs = torch.einsum("ijk->kji", obs) # Change to shape: (channels, H, W)
                    batch_obs[idx,:,:,:] = obs
                    pi = model(obs.unsqueeze(0))
                    action, log_prob = self.sample_action(pi)

                    # ep_v = self.critic(obs)

                    batch_actions[idx] = action
                    batch_log_probs[idx] = log_prob
                    # batch_values[idx] = ep_v

                    obs, reward, terminated, truncated, _ = self.env.step(action)
                    self.environments_done[i] = terminated or truncated

                    batch_rewards[idx] = reward

                if self.environments_done[i]:
                    break
            batch_episode_lengths.append(t_ep + 1)


        batch_rewards_to_go = self._calc_rewards_to_go(batch_rewards, batch_episode_lengths)
        # batch_advantages = self._calc_advantages(batch_rewards, batch_values, batch_episode_lengths)
        return batch_obs, batch_actions, batch_log_probs, batch_rewards_to_go, batch_rewards, batch_episode_lengths, batch_done_mask
    
    
    def _calc_rewards_to_go(self, batch_rewards, episode_lengths):
        batch_rtgs = torch.full((self.num_environments * self.max_timesteps_per_episode,), -float("inf"), device=self.device, dtype=torch.float)
        for ep_idx in range(self.num_environments): # Go through last episode first, such that the order in "flattened" batch_rtgs is consistent with episode order
            current_rtg = 0

            for t_ep in reversed(range(episode_lengths[ep_idx])): # Go back in time: approximation for Q(a, s), through Bellman equation, which is r(a, s) + \gamma Q(a', s')
                current_idx = self.get_batch_idx(ep_idx, t_ep)
                rtg = batch_rewards[current_idx] + self.gamma * current_rtg
                batch_rtgs[current_idx] = rtg

        return batch_rtgs
    
    def get_batch_idx(self, episode, episode_timestep): # timestep starting from 0
        """Find index on flattened batch tensors. One batch has {episodes_per_batch} episodes and a total of {episodes_per_batch} * {max_timesteps_per_batch} timesteps."""
        return self.max_timesteps_per_episode * episode + episode_timestep
    
    @staticmethod
    def sample_action(logits):
        pi = Categorical(logits=logits)
        a = pi.sample()
        log_prob = pi.logits.squeeze()[a]
        return a.item(), log_prob.item()