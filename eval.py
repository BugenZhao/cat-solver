import sys
from game.trainable import Reward, TrainableGame
import numpy as np
from numpy.lib.function_base import disp
from rl import *


def eval_episode(agent: CatAgent, env: TrainableGame, episodes: int = 5, display: bool = False) -> Reward:
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


def eval():
    if len(sys.argv) != 2:
        print(f'usage: {sys.argv[0]} <model-path>')
        exit(1)
    path = sys.argv[1]

    env = TrainableGame()
    obs_dim = env.obs_dim()
    act_dim = env.act_dim()

    model = CatModel(obs_dim, act_dim)
    alg = DQN(model)
    agent = CatAgent(alg, act_dim)
    agent.restore(path)

    print("evaluating...")
    reward = eval_episode(agent, env, episodes=1000)
    print(f"average reward: {reward}")

    input('> interactive?')
    while True:
        eval_episode(agent, env, episodes=1, display=True)
        input('> next?')
