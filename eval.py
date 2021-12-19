import sys
from typing import Tuple, Union
from game.trainable import Reward, TrainableGame
import numpy as np
from numpy.lib.function_base import disp
from rl import *


def eval_episode(agent: CatAgent, env: TrainableGame, episodes: int = 10, display: bool = False) -> Reward:
    agent.train_mode = False
    rewards = []
    for _ in range(episodes):
        env.reset()
        episode_reward = 0
        step = 0

        if display:
            env.print()

        while True:
            obs = env.get_obs()
            action = agent.predict(obs)
            reward, _, terminal = env.step(action)
            step += 1

            if display:
                print(f"> STEP {step}")
                env.print()

            episode_reward += reward
            if terminal:
                break

        rewards.append(episode_reward)
    return np.mean(rewards)


def load_model(path: Union[str, None], gamma: float = 1.0, lr: float = 0.0) -> Tuple[CatAgent, TrainableGame]:
    env = TrainableGame()
    obs_dim = env.obs_dim()
    act_dim = env.act_dim()

    model = CatModel(obs_dim, act_dim)
    alg = DQN(model, gamma=gamma, lr=lr)
    agent = CatAgent(alg, act_dim)

    if path is not None:
        agent.restore(path)
        agent.e_greed = 0.0

    return agent, env


def eval():
    if len(sys.argv) != 2:
        print(f'usage: {sys.argv[0]} <model-path>')
        exit(1)
    path = sys.argv[1]
    agent, env = load_model(path)

    print("evaluating...")
    reward = eval_episode(agent, env, episodes=1000)
    print(f"average reward: {reward}")

    input('> interactive?')
    while True:
        eval_episode(agent, env, episodes=1, display=True)
        input('> next?')


if __name__ == '__main__':
    eval()
