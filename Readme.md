## Tetris AI Project
### Run the Code
In order to run all functionalities, these packages must be installed:
```shell
pip install gymnasium[atari] torch numpy tqdm wandb pygame
```
To train an AI, let the AI play the game or play the game yourself, run main.py:

    python main.py

You will be prompted to either train or play the game.
### Playing the game
After a model has been trained, it can be tested or visualized. The default mode will display the AI on the pygame-tetris.
You will be prompted to specify a `{model_file}` to choose for visualization. `model_file` must be the complete file name. The file has to reside in `exports/`.\
After the game has loaded, press [Enter] to start the game.

Currently, no model has been trained on the gymnasium-tetris environment. It is still possible to see a newly initialized model play on the gymnasium environment - this is just for proof of concept. In order to do so, give "gym" as `model_file`.

In order to play the game yourselve, just enter no `model_file` (an empty one). The tetrominoes can be controlled as follows:
- [q], [e]: rotate right, rotate left
- $\uparrow$, $\downarrow$, $\leftarrow$, $\rightarrow$: move tetromino


### Training the AI

Training is handled by the two classes `DQNAgent()` and `PPO()` in `train_algorithms/`. After initializing an object of them, training can be started by simpling calling `train()`.
#### PPO
The PPO-class takes a few arguments to handle different experiements better:
- config: A config-object from `config.py`. Parameters must be set directly in the main-function of ```main.py```. A better description of these parameters is given in the config-class.
- experiment: To handle multiple training runs, every run is marked with an experiment number. Experiments are saved under `exports` with their respective number.

##### Changing Environment
The PPO training approach uses the pygame-tetris implementation. The sampling process is handled by the generator-class in `generator.py`, which is initialized in the PPO-class through an environment-factory, a `partial` function. Therefore, in order to change environments, this factory or partial function would have to be adapted.

##### Changing Model
Similarly, the model to train could be changed, by simply changing its initialization in the PPO-class (`tetris_model`). If the discrete model was used, which takes discrete information of the game grid, the `environment_factory` would have to be adjusted accordingly, by passing `discrete_obs=True`.
