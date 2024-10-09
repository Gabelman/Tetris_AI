class Config():
    def __init__(self,
                episodes_per_batch = 32, max_timesteps_per_episode = 200,
                updates_per_iteration = 1, num_mini_batch_updates = 8,
                num_sub_mini_batches = 4, overall_timesteps = 10000,
                gamma = 0.95, epsilon = 0.2,
                lam = 0.94, lr = 1e-3,
                anneal_factor = 1, min_lr = 1e-7,
                entropy_coef=0.01, vf_weight=0.1,
                step_reward = 1e-3, line_clear_reward = 50,
                line_density_reward = 0.8,
                height_place_reward = 0.1, height_penalty = 0.2,
                bumpiness_penalty = 0.5, hole_penalty = 2, game_over_penalty = 500, info = "", predict_placement=False, load_model = -1):
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
        self.entropy_coef = entropy_coef
        self.vf_weigth = vf_weight
        self.lr = lr
        self.min_lr = min_lr
        self.anneal_factor = anneal_factor

        # self.current_commit = "a57665626ba57f127bee842e3221b549cb823f82"

        self.line_clear_reward = line_clear_reward
        self.height_place_reward = height_place_reward
        self.height_penalty = height_penalty
        self.bumpiness_penalty = bumpiness_penalty
        self.hole_penalty = hole_penalty
        self.line_density_reward = line_density_reward
        self.game_over_penalty = game_over_penalty
        self.step_reward = step_reward
        self.info = info

        # model
        self.load_model = load_model
        self.predict_placement = predict_placement # Action space will be placements if True, button-actions if False


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
            "entropy_coef": self.entropy_coef,
            "lr": self.lr,
            "anneal_factor": self.anneal_factor,
            "min_lr": self.min_lr,
            # "current_commit": self.current_commit,
            "line_clear_reward": self.line_clear_reward,
            "line_density_reward": self.line_density_reward,
            "height_place_reward": self.height_place_reward,
            "height_penalty": self.height_penalty,
            "bumpiness_penalty": self.bumpiness_penalty,
            "hole_penalty": self.hole_penalty,
            "step_reward": self.step_reward,
            "game_over_penalty": self.game_over_penalty,
            "info": self.info,
            "load_model": self.load_model,
            "predict_placement": self.predict_placement
        }
        return dictionary
