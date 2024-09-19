

def get_batch_idx(self, episode, episode_timestep): # timestep starting from 0
        """Find index on flattened batch tensors. One batch has {episodes_per_batch} episodes and a total of {episodes_per_batch} * {max_timesteps_per_batch} timesteps."""
        return self.max_timesteps_per_episode * episode + episode_timestep