import torch.nn as nn
import torch.nn.functional as F


class CatModel(nn.Module):
    def __init__(self, obs_dim: int, act_dim: int):
        super(CatModel, self).__init__()

        hid1_size = act_dim * 16
        hid2_size = act_dim * 4
        self.fc1 = nn.Linear(obs_dim, hid1_size)
        self.fc2 = nn.Linear(hid1_size, hid2_size)
        self.fc3 = nn.Linear(hid2_size, act_dim)

    def forward(self, obs):
        h1 = F.relu(self.fc1(obs))
        h2 = F.relu(self.fc2(h1))
        q = self.fc3(h2)
        return q
