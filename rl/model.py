from math import sqrt
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.modules.conv import Conv1d


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


class CatCNNModel(nn.Module):
    def __init__(self, obs_dim: int, act_dim: int):
        super(CatCNNModel, self).__init__()

        self.n = int(sqrt(obs_dim))
        assert self.n ** 2 == obs_dim

        hid1_size = act_dim * 16
        hid2_size = act_dim * 4
        self.conv1 = nn.Conv2d(1, 3, 2)
        self.fc1 = nn.Linear(self.fc1_in_size(), hid1_size)
        self.fc2 = nn.Linear(hid1_size, hid2_size)
        self.fc3 = nn.Linear(hid2_size, act_dim)

    def forward(self, obs):
        obs = torch.reshape(obs, (-1, 1, self.n, self.n))
        c1 = F.relu(self.conv1(obs))

        c1 = torch.reshape(c1, (-1, self.fc1_in_size()))
        h1 = F.relu(self.fc1(c1))
        h2 = F.relu(self.fc2(h1))
        q = self.fc3(h2)
        return q

    def fc1_in_size(self) -> int:
        return 3 * ((self.n - 2 + 1) ** 2)
