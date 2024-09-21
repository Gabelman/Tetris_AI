class Config():
    def __init__(self, episodes_per_batch = 32, max_timesteps_per_episode = 200, updates_per_iteration = 1, num_mini_batch_updates = 8, num_sub_mini_batches = 4, overall_timesteps = 10000, gamma = 0.95, epsilon = 0.2, lam = 0.94, lr = 1e-3, ):
        # learning params
        self.episodes_per_batch = episodes_per_batch
        self.max_timesteps_per_episode = max_timesteps_per_episode
        self.updates_per_iteration = updates_per_iteration
        self.num_mini_batch_updates = num_mini_batch_updates
        self.num_sub_mini_batches = num_sub_mini_batches

        self.overall_timesteps = overall_timesteps
        # train
        self.gamma = gamma
        self.epsilon = epsilon
        self.lam = lam
        self.lr = lr

        self.current_commit = "a57665626ba57f127bee842e3221b549cb823f82"


    def to_dict(self):
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
            "current_commit": self.current_commit,
        }
        return dictionary
