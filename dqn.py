import torch.nn as nn

class DQN(nn.Module):

    def __init__(self, num_observations, num_actions):
        super(DQN, self).__init__()

        self.layer1 = nn.Linear(num_observations, 128)
        self.layer2 = nn.Linear(128, 64)
        self.layer3 = nn.Linear(64, num_actions)

    def forward(self, x):
        x = nn.functional.relu(self.layer1(x))
        x = nn.functional.relu(self.layer2(x))
        x = self.layer3(x)
        return x