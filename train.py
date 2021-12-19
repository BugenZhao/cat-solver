import logging
import os
from eval import eval_episode
from game.trainable import TrainableGame
import numpy as np
from logging import info
from rl import *

logging.basicConfig(level="INFO")

LEARN_FREQ = 5  # training frequency
MEMORY_SIZE = 200000
MEMORY_WARMUP_SIZE = 200
BATCH_SIZE = 64
LEARNING_RATE = 0.0005
GAMMA = 0.99
TRAIN_EPISODE = 100000
EVAL_PERIOD = 50
SAVE_PERIOD = 500
MODEL_DIR = "models"


def train_episode(agent: CatAgent, env: TrainableGame, rpm: ReplayMemory):
    env.reset()
    total_reward = 0
    step = 0

    while True:
        step += 1
        obs = env.get_obs()
        action = agent.sample(obs)
        reward, next_obs, terminal = env.step(action)
        total_reward += reward

        rpm.append(obs, action, reward, next_obs, terminal)
        if len(rpm) > MEMORY_WARMUP_SIZE and step % LEARN_FREQ == 0:
            b_obs, b_act, b_reward, b_next_obs, b_terminal = rpm.sample_batch(
                BATCH_SIZE)
            _loss = agent.learn(b_obs, b_act, b_reward, b_next_obs, b_terminal)

        if terminal:
            break

    return total_reward


def train():
    os.makedirs(MODEL_DIR)
    env = TrainableGame()
    obs_dim = env.obs_dim()
    act_dim = env.act_dim()

    rpm = ReplayMemory(MEMORY_SIZE, obs_dim, 0)

    model = CatModel(obs_dim, act_dim)
    alg = DQN(model, gamma=GAMMA, lr=LEARNING_RATE)
    agent = CatAgent(alg, act_dim)

    while len(rpm) < MEMORY_WARMUP_SIZE:
        train_episode(agent, env, rpm)

    for episode in range(TRAIN_EPISODE + 1):
        _reward = train_episode(agent, env, rpm)
        if episode % EVAL_PERIOD == 0:
            eval_reward = eval_episode(agent, env)
            info(f'episode {episode}, eval_reward {eval_reward}')
        if episode % SAVE_PERIOD == 0:
            agent.save(os.sep.join(
                [MODEL_DIR, f'e_{episode}_r_{eval_reward}.model']))
