from train_algorithms.ppo_trainer import PPO
from environments.pygame_tetris import play_pygame

import wandb
import torch
from config import Config


device = torch.device('cuda:1' if torch.cuda.is_available() else 'cpu')
if __name__ == '__main__':
    # play_pygame("ppo_conv_model_exp_1.pth", device)
    wandb.login()
    config = Config(episodes_per_batch=50, updates_per_iteration=3,
                    num_mini_batch_updates=5, num_sub_mini_batches=1,
                    max_timesteps_per_episode=200, overall_timesteps=20000, lr=0.01,
                    game_over_penalty=2000)
    ppo = PPO(device, config, experiment=4)
    ppo.train(config.overall_timesteps)

    ppo.close()
    # try:
    #     choice = input("Enter 'train' to train the AI or 'play' to watch the AI play: ")
    #     if choice.lower() == 'train':
    #         # train_ai(continue_training=True)
    #         wandb.login()
    #         config = Config(episodes_per_batch=128, updates_per_iteration=3,
    #                         num_mini_batch_updates=32, num_sub_mini_batches=8,
    #                         overall_timesteps=100000, lr=0.01,
    #                         game_over_penalty=2000)
    #         ppo = PPO(device, config, experiment=1)
    #         ppo.train(100000)

    #         ppo.close()

    #     elif choice.lower() == 'play':
    #         player = input("Enter the filename {model.pth} to load the model from. Press ENTER to (empty string) to play as Human.")
    #         play_pygame(player, device, speed=3, scale=6)
    #     else:
    #         print("Invalid choice. Please enter 'train' or 'play'.")
    # except EOFError:
    #     print("No input received. Defaulting to training mode.")
    #     # train_ai(continue_training=True)
    # except KeyboardInterrupt:
    #     print("\nProgram interrupted by user. Exiting.")
    # except Exception as e:
    #     print(f"An unexpected error occurred: {e}")
    # env.close()
