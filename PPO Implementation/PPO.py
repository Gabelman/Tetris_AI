import torch
from torch import nn
from generator import Generator
from TetrisConvModel import TetrisAgent
from torch.distributions import Categorical
from torch.optim import Adam
import numpy as np

from gymnasium import Env

from tqdm import tqdm
import wandb

class PPO():
    def __init__(self, env: Env, device) -> None:
        self.init_hyperparameters()

        self.env = env
        self.action_space = env.action_space.n # this is specifically for openAI gymnasium. Might as well be = 5
        self.observation_space = env.observation_space.shape
        self.actor = TetrisAgent(self.observation_space, self.action_space).to(device)
        self.critic = TetrisAgent(self.observation_space, 1).to(device)
        # self.tetris_model = TetrisAgent(self.observation_space, self.action_space).to(device)
        
        self.generator = Generator(observation_space=self.observation_space, max_timesteps_per_episode=self.max_timesteps_per_episode,num_environments=self.episodes_per_batch, gamma=self.gamma)
        self.actor_optim = Adam(self.actor.parameters(), lr=self.actor_lr)
        self.critic_optim = Adam(self.critic.parameters(), lr=self.cricit_lr)

        self.iteration = 0

        self.device = device
        
    def train(self, total_timesteps):
        current_timesteps = 0
        self.iteration = 0
        while current_timesteps < total_timesteps:
            self.iteration += 1
            batch_obs, batch_actions, batch_log_probs, batch_rtgs, batch_rews, batch_lengths, batch_done_mask = self.generator.sample(self.actor)

            step_sum = sum(batch_lengths)
            current_timesteps += step_sum

            # Calculate V_{\phi, k}(a, s)
            V, _ = self.evaluate(batch_obs, batch_actions, batch_done_mask)

            # Calculate advantage
            A_k = self._calc_advantages(batch_rews, V.detach(), batch_lengths, batch_done_mask)

            # Normalize advantage
            A_k = (A_k - A_k.mean()) / (A_k.std() + 1e-10)


            minibatch_size = step_sum // self.num_minibatches # floor division
            batch_idcs = np.arange(step_sum)

            print(f"=============\Iteration: {self.iteration}\ncurrent time steps: {current_timesteps}\n=============\n")
            print(f"episodic return: {torch.sum(batch_rews[batch_done_mask])}")
            for i in tqdm(range(self.updates_per_iteration)):
                min_batch_idcs = np.random.choice(batch_idcs, (self.num_minibatches, minibatch_size))
                for idcs in min_batch_idcs:
                    mini_obs = batch_obs[idcs]
                    mini_actions = batch_actions[idcs]
                    mini_log_probs = batch_log_probs[idcs]
                    mini_rtgs = batch_rtgs[idcs]
                    mini_done_mask = batch_done_mask[idcs]
                    mini_A_k = A_k[idcs]

                    V, log_probs = self.evaluate(mini_obs, mini_actions, mini_done_mask)

                    # Calculate loss for actor model
                    # \phi_{\theta}(a, s) / \phi_{\theta_{k}}(a, s)
                    prob_ratio = torch.exp(log_probs - mini_log_probs)
                    
                    # surrogate objective see https://spinningup.openai.com/en/latest/algorithms/ppo.html
                    # torch.where: takes a conditional as first param, then the result for true, false. 
                    clip = torch.where(mini_A_k >= 0, torch.min(prob_ratio, torch.tensor(1 + self.epsilon, dtype=prob_ratio.dtype, device=self.device)), torch.max(prob_ratio, torch.tensor(1 - self.epsilon, dtype=prob_ratio.dtype, device=self.device)))

                    # Calculate Losses
                    actor_loss = (-clip * mini_A_k).mean()
                    critic_loss = nn.MSELoss()(mini_rtgs[mini_done_mask], V)


                    self.actor_optim.zero_grad()
                    actor_loss.backward(retain_graph=True)
                    self.actor_optim.step()
                    self.critic_optim.zero_grad()
                    critic_loss.backward()
                    self.critic_optim.step()



    # def sample(self):
    #     batch_obs = torch.full((self.episodes_per_batch * self.max_timesteps_per_episode, *self.observation_space), -1, dtype=torch.float, device=self.device, requires_grad=False) # Batch Observations. (num_episodes * episode_length, observation_shape)
    #     batch_log_probs = torch.full((self.episodes_per_batch * self.max_timesteps_per_episode,), 0, dtype=torch.float, device=self.device, requires_grad=False) # (num_episodes * episode_length)
    #     batch_actions = torch.full((self.episodes_per_batch * self.max_timesteps_per_episode,), -1, dtype=torch.int, device=self.device, requires_grad=False) # (num_episodes * episode_length, action_space)
    #     batch_rewards = torch.full((self.episodes_per_batch * self.max_timesteps_per_episode,), 0, dtype=torch.float, device=self.device, requires_grad=False) # (num episodes, episode_length)
    #     # batch_values = torch.full(self.episodes_per_batch, self.max_timesteps_per_episode, 0, dtype=torch.float)
    #     batch_rewards_to_go = torch.full((self.episodes_per_batch * self.max_timesteps_per_episode,), 0, dtype=torch.float, device=self.device, requires_grad=False) # (num_episodes * episode_length)
    #     batch_done_mask = torch.full((self.episodes_per_batch * self.max_timesteps_per_episode,), False, device=self.device, requires_grad=False)
    #     batch_episode_lengths = []

    #     #reshape obs-space from (widht, height, channels) to (channels, height, width) (keeping batch size the same)
    #     batch_obs = torch.einsum("nwhc->nchw", batch_obs)
    #     print(f"--------------------\nSampling for iteration {self.iteration}\n--------------------\n")
    #     for e in tqdm(range(self.episodes_per_batch)):
    #         done = False
    #         obs, _ = self.env.reset()

    #         for t_ep in tqdm(range(self.max_timesteps_per_episode)):
    #             idx = self.get_batch_idx(e, t_ep) # In order to insert values into "flattened" tensors immediately
    #             batch_done_mask[idx] = True

    #             obs = torch.tensor(obs).to(self.device, dtype=torch.float)
    #             obs = torch.einsum("ijk->kji", obs) # Change to shape: (channels, H, W)
    #             batch_obs[idx,:,:,:] = obs
    #             pi = self.actor(obs.unsqueeze(0))
    #             action, log_prob = self.sample_action(pi)

    #             # ep_v = self.critic(obs)

    #             batch_actions[idx] = action
    #             batch_log_probs[idx] = log_prob
    #             # batch_values[idx] = ep_v

    #             obs, reward, terminated, truncated, _ = self.env.step(action)
    #             done = terminated or truncated

    #             batch_rewards[idx] = reward

    #             if done:
    #                 break
    #         batch_episode_lengths.append(t_ep + 1)


    #     batch_rewards_to_go = self._calc_rewards_to_go(batch_rewards, batch_episode_lengths)
    #     # batch_advantages = self._calc_advantages(batch_rewards, batch_values, batch_episode_lengths)
    #     return batch_obs, batch_actions, batch_log_probs, batch_rewards_to_go, batch_rewards, batch_episode_lengths, batch_done_mask

    # @staticmethod
    # def sample_action(logits):
    #     pi = Categorical(logits=logits)
    #     a = pi.sample()
    #     log_prob = pi.logits.squeeze()[a]
    #     return a.item(), log_prob.item()
    
    # def _calc_rewards_to_go(self, batch_rewards, episode_lengths):
    #     batch_rtgs = torch.full((self.episodes_per_batch * self.max_timesteps_per_episode,), -float("inf"), device=self.device, dtype=torch.float)
    #     for ep_idx in range(self.episodes_per_batch): # Go through last episode first, such that the order in "flattened" batch_rtgs is consistent with episode order
    #         current_rtg = 0

    #         for t_ep in reversed(range(episode_lengths[ep_idx])): # Go back in time: approximation for Q(a, s), through Bellman equation, which is r(a, s) + \gamma Q(a', s')
    #             current_idx = self.get_batch_idx(ep_idx, t_ep)
    #             rtg = batch_rewards[current_idx] + self.gamma * current_rtg
    #             batch_rtgs[current_idx] = rtg

    #     return batch_rtgs
    
    def _calc_advantages(self, rewards, values, episode_lengths, done_mask):
        """Calculate Generalized Advantage Estimate (GAE), see https://arxiv.org/abs/1506.02438"""
        advantages = torch.full((self.episodes_per_batch * self.max_timesteps_per_episode,), 0, device=self.device, dtype=torch.float)
        for ep in range(self.episodes_per_batch):
            length = episode_lengths[ep]
            last_advantage = 0
            for t in reversed(range(length)):
                idx_t = self.get_batch_idx(ep, t)
                if t+1 == length:
                    delta = rewards[idx_t] - values[idx_t]
                else:
                    delta = rewards[idx_t] + self.gamma * values[idx_t+1] - values[idx_t] #reward + TD residual, which is V(s_{t+1}) - V(s_{t}), except for the "latest" time step (the first to come, as the steps are inversed)
                advantage = delta + self.lam * self.gamma * last_advantage
                last_advantage = advantage
                advantages[idx_t] = advantage

        return advantages[done_mask]
    

    def evaluate(self, batch_obs, batch_acts, valid):
        V = self.critic(batch_obs).squeeze()
        log_probs = self.actor(batch_obs)
        log_probs = log_probs[torch.arange(log_probs.size(0)), batch_acts] # select only probs of actions taken. torch.arange(...) selects all rows, batch_acts selects the appropriate actions for each row (timestep over different episodes).
        return V[valid], log_probs[valid]

    # def get_batch_idx(self, episode, episode_timestep): # timestep starting from 0
    #     """Find index on flattened batch tensors. One batch has {episodes_per_batch} episodes and a total of {episodes_per_batch} * {max_timesteps_per_batch} timesteps."""
    #     return self.max_timesteps_per_episode * episode + episode_timestep
    
    def init_hyperparameters(self, episodes_per_batch = 4, max_timesteps_per_episode = 10000, updates_per_iteration = 5, num_minibatches = 1, gamma = 0.95, epsilon = 0.2, lam = 0.94, actor_lr = 1e-3, cricit_lr = 1e-3):
        self.episodes_per_batch = episodes_per_batch
        self.max_timesteps_per_episode = max_timesteps_per_episode
        self.updates_per_iteration = updates_per_iteration
        self.num_minibatches = num_minibatches

        self.gamma = gamma # Used in rewards to go
        self.epsilon = epsilon # PPO clipping objective
        self.lam = lam# value is following https://arxiv.org/abs/1506.0243

        self.actor_lr = actor_lr
        self.cricit_lr = cricit_lr