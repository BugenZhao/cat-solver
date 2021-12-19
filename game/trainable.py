import logging
from typing import Tuple
from .core import GameCore, Result
import numpy as np

Obs = np.ndarray
Act = int
Reward = float


class TrainableGame:
    def __init__(self, n: int = 11) -> None:
        self.n = n
        self.extra_walls = 10
        self.reset()

    def increase_difficulty(self):
        self.extra_walls = max(self.extra_walls - 1, 0)
        logging.info(f"increased difficulty, extra walls: {self.extra_walls}")

    def difficulty_hard(self):
        self.extra_walls = 0

    def obs_dim(self) -> int:
        return self.n ** 2

    def act_dim(self) -> int:
        return self.n ** 2

    def get_obs(self) -> np.ndarray:
        vals = []
        for row in self.game.state:
            for block in row:
                vals.append(block.value)
        return np.array(vals)

    def reset(self):
        wall_count = self.n - 3 + self.extra_walls
        self.game = GameCore(self.n, self.n, wall_count)

    def win_reward(self) -> Reward:
        return 100

    def step(self, action: Act, display: bool = False) -> Tuple[Reward, Obs, bool]:
        action = Act(action)
        i = action // self.n
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
            reward += 0
        elif self.game.result == Result.WIN:
            reward += self.win_reward()

        if display:
            print(self.game.result.name)

        return reward, next_obs, terminal

    def print(self):
        self.game.print_state()
