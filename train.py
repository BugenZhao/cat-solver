import logging
import os
import sys
from eval import eval_episode, load_model
from game.trainable import TrainableGame
import numpy as np
from logging import info
from rl import *

LEARN_FREQ = 5  # training frequency
MEMORY_SIZE = 200000
MEMORY_WARMUP_SIZE = 256
BATCH_SIZE = 64
LEARNING_RATE = 0.0005
GAMMA = 0.99
TRAIN_EPISODE = 1000000
EVAL_PERIOD = 100
SAVE_PERIOD = 1000
MODEL_DIR = "models"


def train_episode(agent: CatAgent, env: TrainableGame, rpm: ReplayMemory):
    agent.train_mode = True
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

    if total_reward >= env.win_reward() * 0.9:
        env.increase_difficulty()

    return total_reward


def train():
    logging.basicConfig(level="INFO")
    os.makedirs(MODEL_DIR, exist_ok=True)

    checkpoint_path = sys.argv[1] if len(sys.argv) == 2 else None
    agent, env = load_model(checkpoint_path, gamma=GAMMA, lr=LEARNING_RATE)
    rpm = ReplayMemory(MEMORY_SIZE, env.obs_dim(), 0)

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


if __name__ == '__main__':
    train()
