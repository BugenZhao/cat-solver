import paddle
import paddle.nn as nn
import paddle.nn.functional as F
import parl
import numpy as np


class CatModel(parl.Model):
    def __init__(self, obs_dim: int, act_dim: int):
        super().__init__()

        hid1_size = act_dim * 10
        hid2_size = act_dim * 10
        self.fc1 = nn.Linear(obs_dim, hid1_size)
        self.fc2 = nn.Linear(hid1_size, hid2_size)
        self.fc3 = nn.Linear(hid2_size, act_dim)

    def forward(self, obs):
        h1 = F.relu(self.fc1(obs))
        h2 = F.relu(self.fc2(h1))
        q = self.fc3(h2)
        return q


class CatAgent(parl.Agent):
    def __init__(self, algorithm: parl.Algorithm, act_dim: int):
        super().__init__(algorithm)

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
        obs = paddle.to_tensor(obs, dtype='float32')
        q = self.alg.predict(obs)
        act = q.argmax().numpy()[0]
        return act

    def learn(self, obs, act, reward, next_obs, terminal) -> float:
        if self.global_step % self.update_target_steps == 0:
            self.alg.sync_target()
        self.global_step += 1

        act = np.expand_dims(act, axis=-1)
        reward = np.expand_dims(reward, axis=-1)
        terminal = np.expand_dims(terminal, axis=-1)

        obs = paddle.to_tensor(obs, dtype='float32')
        act = paddle.to_tensor(act, dtype='int32')
        reward = paddle.to_tensor(reward, dtype='float32')
        next_obs = paddle.to_tensor(next_obs, dtype='float32')
        terminal = paddle.to_tensor(terminal, dtype='float32')
        loss = self.alg.learn(obs, act, reward, next_obs, terminal)
        return loss.numpy()[0]
