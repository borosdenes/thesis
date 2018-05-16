"""
Script to test environment observation shape, reward scale, etc.
"""

import gym
from environment_utils import prepro
from PIL import Image
import os
import numpy as np


env = gym.make('Skiing-v0')
observation_folder = 'observer/{}/diff'.format(env)

env.reset()
previous_observation, _, _, _ = env.step(env.action_space.sample())
previous_observation = np.array(Image.fromarray(previous_observation).convert(mode='L')).astype(np.uint8)

for idx in range(1000):
    # env.render()
    observation, reward, done, info = env.step(env.action_space.sample())

    observation = np.array(Image.fromarray(observation).convert(mode='L')).astype(np.uint8)
    diff = observation - previous_observation

    if not os.path.exists(observation_folder):
        os.makedirs(observation_folder)
    Image.fromarray(diff).save(os.path.join(observation_folder, '{}.png'.format(idx)))

    previous_observation = observation
