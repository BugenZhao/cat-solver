from typing import Union
from rl.model import CatModel
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from copy import deepcopy


class DQN:
    def __init__(self, model: CatModel, gamma: float = 1.0, lr: float = 0.0):
        self.model = model
        self.target_model = deepcopy(model)
        self.device = torch.device(
            "cuda" if torch.cuda.is_available() else "cpu")
        self.model.to(self.device)
        self.target_model.to(self.device)

        self.gamma = gamma
        self.lr = lr

        self.loss = nn.MSELoss()
        self.optimizer = optim.Adam(self.model.parameters(), lr=self.lr)

    def predict(self, obs):
        with torch.no_grad():
            q = self.model(obs)
        return q

    def learn(self, obs, action, reward, next_obs, terminal):
        pred_value = self.model(obs).gather(1, action)
        with torch.no_grad():
            max_v = self.target_model(next_obs).max(1, keepdim=True)[0]
            target = reward + (1 - terminal) * self.gamma * max_v
        self.optimizer.zero_grad()
        loss = self.loss(pred_value, target)
        loss.backward()
        self.optimizer.step()
        return loss.item()

    def sync_target(self):
        state = self.model.state_dict()
        self.target_model.load_state_dict(state)
