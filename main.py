from train_algorithms.ppo_trainer import PPO
from environments.pygame_tetris import play_pygame, let_AI_play_pygame
from environments.gymnasium_tetris import let_AI_play_gymnasium
from train_algorithms.dqn_trainer import DQNAgent

import wandb
import torch
from config import Config

from models.tetris_discrete_model import TetrisAI



device = torch.device('cuda:1' if torch.cuda.is_available() else 'cpu')
if __name__ == '__main__':
    
    
    info = "Reward gives penalty for every step based on the static board state. Otherwise there is normal step reward and placement reward."
    experiment = 1
    config = Config(episodes_per_batch=1, updates_per_iteration=2,
                  num_mini_batch_updates=5, num_sub_mini_batches=1,
                  max_timesteps_per_episode=150, overall_timesteps=50000, lr=0.01,
                  game_over_penalty=2000, step_reward=0.1, height_place_reward=2, info=info, predict_placement=False)
       
    try:
        choice = input("Enter 'train' to train the AI or 'play' to watch the AI play: ")
        if choice.lower() == 'train':
            choice = input("Log results?[n/Y]: ")
            if choice.lower() == 'n':
                wandb.init(mode="disabled")
            else:
                wandb.init()
            wandb.login()
            ppo = PPO(device, config, experiment=experiment)
            ppo.train(100000)

            ppo.close()

        elif choice.lower() == 'play':
            player = input("Enter the filename {model.pth} to load the model from. Press ENTER (empty string) to play as Human.")
            if player == "gym":
                let_AI_play_gymnasium("", False, device, True, speed=100)
            elif player:
                let_AI_play_pygame(player, False, device,games=1,prob_actions=False, speed=50, scale=6)
            else:
                play_pygame(speed=3, scale=6)
        else:
            print("Invalid choice. Please enter 'train' or 'play'.")
    except EOFError:
        print("No input received. Defaulting to training mode.")
        # train_ai(continue_training=True)
    except KeyboardInterrupt:
        print("\nProgram interrupted by user. Exiting.")
    except Exception as e:
        print(f"An unexpected error occurred: {e}")
    # env.close()
