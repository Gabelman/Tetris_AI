import torch
import torch.nn as nn

# AI implementation
class TetrisAI(nn.Module):
    #def __init__(self, grid_size: tuple[int, int], tetronimo_size: int, action_space: int):
    def __init__(self, input_size, action_space: int):
        super(TetrisAI, self).__init__()
        #input_size = grid_size[0] * grid_size[1] + tetronimo_size  # Deprecated: +8 for x, y, shape dimensions, bumpiness, holes, height, clearable_lines

        # Now the input size is just the grid and the new_tile information.
        self.fc1 = nn.Linear(input_size, 128)
        self.fc2 = nn.Linear(128, 64)
        self.fc3 = nn.Linear(64, 32)
        self.fc4 = nn.Linear(32, action_space) # 4 possible actions: left, right, rotate, do nothing

    def forward(self, x):
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        x = torch.relu(self.fc3(x))
        return self.fc4(x)