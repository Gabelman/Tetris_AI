class Config():
    def __init__(self):
        # learning params
        self.episodes_per_batch = 32
        self.max_timesteps_per_episode = 200
        self.updates_per_iteration = 1
        self.num_mini_batch_updates = 8
        self.num_sub_mini_batches = 4

        self.overall_timesteps = 10000
        # train
        self.gamma = 0.95
        self.epsilon = 0.2
        self.lam = 0.94
        self.lr = 1e-3

    def as_dict(self):
        dictionary = {
            "episodes_per_batch": self.episodes_per_batch,
            "max_timesteps_per_episode": self.max_timesteps_per_episode,
            "updates_per_iteration": self.updates_per_iteration,
            "num_mini_batch_updates": self.num_mini_batch_updates,
            "num_sub_mini_batches": self.num_sub_mini_batches,
            "overall_timesteps": self.overall_timesteps,
            "gamma": self.gamma,
            "epsilon": self.epsilon,
            "lam": self.lam,
            "lr": self.lr,
        }
        return dictionary
