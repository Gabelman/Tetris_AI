import torch
from torch import nn
from TetrisConvModel import TetrisAgent
from torch.distributions import Categorical
from torch.optim import Adam

from gymnasium import Env

class PPO():
    def __init__(self, env: Env) -> None:
        self._init_hyperparameters()

        self.env = env
        self.action_space = env.action_space.n # this is specifically for openAI gymnasium. Might as well be = 5
        self.observation_space = env.observation_space
        self.actor = TetrisAgent(self.observation_space, self.action_space)
        self.cricit = TetrisAgent(self.observation_space, 1)
        

        self.actor_optim = Adam(self.actor.parameters(), lr=self.actor_lr)
        self.critic_optim = Adam(self.cricit.parameters(), lr=self.cricit_lr)

    def learn(self, total_timesteps):
        current_timesteps = 0

        while current_timesteps < total_timesteps:
            batch_obs, batch_actions, batch_log_probs, batch_rtgs, batch_lengths, batch_done_mask = self.sample()
            current_timesteps += torch.sum(batch_lengths)

            # Calculate V_{\phi, k}(a, s)
            V, log_probs_k = self.evaluate(batch_obs, batch_done_mask)

            # Calculate advantage
            A_k = batch_rtgs - V.detach()

            # Normalize advantage
            A_k = (A_k - A_k.mean()) / (A_k.std() + 1e-10)

            for _ in range(self.updates_per_iteration):
                V, log_probs = self.evaluate(batch_obs, batch_done_mask)

                # Calculate loss for actor model
                # \phi_{\theta}(a, s) / \phi_{\theta_{k}}(a, s)
                prob_ratio = torch.exp(log_probs - log_probs_k)
                
                # surrogate objective see https://spinningup.openai.com/en/latest/algorithms/ppo.html
                # torch.where: takes a conditional as first param, then the result for true, false. 
                clip = torch.where(A_k >= 0, torch.min(prob_ratio, 1 + self.epsilon), torch.max(prob_ratio, 1 - self.epsilon))

                # Calculate Losses
                actor_loss = (-clip * A_k).mean()
                critic_loss = nn.MSELoss()(batch_rtgs, V)


                self.actor_optim.zero_grad()
                actor_loss.backward(retain_graph=True)
                self.actor_optim().step()
                self.critic_optim.zero_grad()
                critic_loss.backward()
                self.critic_optim.step()



    def sample(self):
        batch_obs = torch.full((self.episodes_per_batch * self.max_timesteps_per_episode, self.observation_space), -1, dtype=torch.float) # Batch Observations. (num_episodes * episode_length, observation_shape)
        batch_log_probs = torch.full(self.episodes_per_batch * self.max_timesteps_per_episode, 0, dtype=torch.float) # (num_episodes * episode_length)
        batch_actions = torch.full((self.episodes_per_batch * self.max_timesteps_per_episode, self.action_space), -1, dtype=torch.float) # (num_episodes * episode_length, action_space)
        batch_rewards = torch.full(self.episodes_per_batch, self.max_timesteps_per_episode, -float("inf"), dtype=torch.float) # (num episodes, episode_length)
        batch_rewards_to_go = torch.full(self.episodes_per_batch * self.max_timesteps_per_episode, -float("inf"), dtype=torch.float) # (num_episodes * episode_length)
        batch_done_mask = torch.full(self.episodes_per_batch * self.max_timesteps_per_episode, False)
        batch_episode_lengths = torch.full(self.episodes_per_batch, 0)


        for e in range(self.episodes_per_batch):
            done = False
            obs, _ = self.env.reset()

            for t_ep in range(self.max_timesteps_per_episode):
                idx = self.get_batch_idx(e, t_ep) # In order to insert values into flattened tensors immediately
                batch_done_mask[idx] = True

                batch_obs[idx] = obs
                pi = self.actor(obs)
                action, log_prob = self.sample_action(pi)

                batch_actions[idx] = action
                batch_log_probs[idx] = log_prob

                obs, reward, terminated, truncated, _ = self.env.step(action)
                done = terminated or truncated
                batch_rewards[e][t_ep] = reward

                if done:
                    break
            batch_episode_lengths[e] = t_ep + 1


        batch_rewards_to_go = self.calc_rewards_to_go(batch_rewards, batch_episode_lengths)
        return batch_obs, batch_actions, batch_log_probs, batch_rewards_to_go, batch_episode_lengths, batch_done_mask


    def sample_action(logits):
        pi = Categorical(logits=logits)
        a = pi.sample()
        log_prob = pi[a]
        return pi.item(), log_prob.item()
    
    def calc_rewards_to_go(self, batch_rewards, episode_lengths):
        batch_rtgs = torch.full(self.episodes_per_batch * self.max_timesteps_per_episode, -float("inf"), dtype=torch.float)
        for ep_idx in range(self.episodes_per_batch): # Go through last episode first, such that the order in "flattened" batch_rtgs is consistent with episode order
            current_rtg = 0

            for t_ep in reversed(range(episode_lengths[ep_idx])): # Go back in time: approximation for Q(a, s), through Bellman equation, which is r(a, s) + \gamma Q(a', s')
                rtg = batch_rewards[ep_idx][t_ep] + self.gamma * current_rtg
                current_idx = self.get_batch_idx(ep_idx, t_ep)
                batch_rtgs[current_idx] = rtg

        return batch_rtgs
    
    def evaluate(self, batch_obs, valid):
        V = self.cricit(batch_obs).squeeze()
        log_probs = self.actor(batch_obs)
        return V[valid], log_probs[valid]

    def get_batch_idx(self, episode, episode_timestep): # timestep starting from 0
        return self.max_timesteps_per_episode * episode + episode_timestep
    
    def _init_hyperparameters(self):
        self.episodes_per_batch = 4
        self.max_timesteps_per_episode = 10000
        self.updates_per_iteration = 5
        self.gamma = 0.95 # Used in rewards to go
        self.epsilon = 0.2 # PPO clipping objective
        
        self.actor_lr = 1e-3
        self.cricit_lr = 1e-3