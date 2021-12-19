from .utils import DEVICE
from .algorithm import DQN
import torch
import numpy as np


class CatAgent:
    def __init__(self, algorithm: DQN, act_dim: int):
        self.alg = algorithm
        self.act_dim = act_dim
        self.global_step = 0
        self.update_target_steps = 200

        self.e_greed = 0.2
        self.e_greed_decr = 0.001
        self.rng = np.random.default_rng()
        self.train_mode = True

    def sample(self, obs: np.ndarray) -> int:
        if self.train_mode:
            if self.rng.uniform() < self.e_greed:
                # exploration
                act = self.rng.integers(self.act_dim)
            else:
                # exploitation
                act = self.predict(obs)
            self.e_greed = max(0.001, self.e_greed - self.e_greed_decr)
            return act
        else:
            # exploitation
            return self.predict(obs)

    def predict(self, obs: np.ndarray) -> int:
        obs = torch.tensor(obs, dtype=torch.float32).to(DEVICE)
        q = self.alg.predict(obs)
        act = q.argmax().cpu().numpy()
        return act

    def learn(self, obs, act, reward, next_obs, terminal) -> float:
        if self.global_step % self.update_target_steps == 0:
            self.alg.sync_target()
        self.global_step += 1

        act = np.expand_dims(act, axis=-1)
        reward = np.expand_dims(reward, axis=-1)
        terminal = np.expand_dims(terminal, axis=-1)

        obs = torch.tensor(obs, dtype=torch.float32).to(DEVICE)
        act = torch.tensor(act, dtype=torch.int64).to(DEVICE)
        reward = torch.tensor(reward, dtype=torch.float32).to(DEVICE)
        next_obs = torch.tensor(next_obs, dtype=torch.float32).to(DEVICE)
        terminal = torch.tensor(terminal, dtype=torch.float32).to(DEVICE)

        loss = self.alg.learn(obs, act, reward, next_obs, terminal)
        return loss

    def save(self, save_path: str):
        torch.save(self.alg.model.state_dict(), save_path)

    def restore(self, save_path: str):
        checkpoint = torch.load(save_path, map_location=DEVICE)
        self.alg.model.load_state_dict(checkpoint)
