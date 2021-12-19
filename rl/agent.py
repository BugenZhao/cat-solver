from .algorithm import DQN
import torch
import numpy as np


class CatAgent:
    def __init__(self, algorithm: DQN, act_dim: int):
        self.alg = algorithm
        self.act_dim = act_dim
        self.global_step = 0
        self.update_target_steps = 200

        self.e_greed = 0.1
        self.e_greed_decr = 0.001
        self.rng = np.random.default_rng()

    def sample(self, obs: np.ndarray) -> int:
        if self.rng.uniform() < self.e_greed:
            # exploration
            act = self.rng.integers(self.act_dim)
        else:
            act = self.predict(obs)
        self.e_greed = max(0.001, self.e_greed - self.e_greed_decr)
        return act

    def predict(self, obs: np.ndarray) -> int:
        obs = torch.tensor(obs, dtype=torch.float32)
        q = self.alg.predict(obs)
        act = q.argmax().numpy()
        return act

    def learn(self, obs, act, reward, next_obs, terminal) -> float:
        if self.global_step % self.update_target_steps == 0:
            self.alg.sync_target()
        self.global_step += 1

        act = np.expand_dims(act, axis=-1)
        reward = np.expand_dims(reward, axis=-1)
        terminal = np.expand_dims(terminal, axis=-1)

        obs = torch.tensor(obs, dtype=torch.float32)
        act = torch.tensor(act, dtype=torch.int64)
        reward = torch.tensor(reward, dtype=torch.float32)
        next_obs = torch.tensor(next_obs, dtype=torch.float32)
        terminal = torch.tensor(terminal, dtype=torch.float32)

        loss = self.alg.learn(obs, act, reward, next_obs, terminal)
        return loss
