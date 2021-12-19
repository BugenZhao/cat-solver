from typing import Tuple
from .core import GameCore, Result
import numpy as np

Obs = np.ndarray
Act = int
Reward = float


class TrainableGame:
    def __init__(self, n=11) -> None:
        self.n = n
        self.reset()

    def obs_dim(self) -> int:
        return self.n ** 2

    def act_dim(self) -> int:
        return self.n ** 2

    def get_obs(self) -> np.ndarray:
        return np.array(self.game.state, dtype='int8')

    def reset(self):
        self.game = GameCore(self.n, self.n, self.n - 3)

    def step(self, action: Act) -> Tuple[Reward, Obs, bool]:
        i = action / self.n
        j = action % self.n
        reward = 0
        terminal = True

        put_success = self.game.put_wall((i, j))
        if not put_success:
            reward -= 5
        self.game.step_cat()
        next_obs = self.get_obs()
        if self.game.result == Result.GAMING:
            terminal = False
        elif self.game.result == Result.WIN:
            reward += 100

        return reward, next_obs, terminal
