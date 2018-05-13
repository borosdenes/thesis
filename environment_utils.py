"""
Environment utility functions.
OpenCV is avoided due to bad installation.
"""

import numpy as np
from PIL import Image


def rgb2gray(rgb):
    return np.dot(rgb[...,:3], [0.299, 0.587, 0.114])


def prepro(I, flatten=True, color='bin', downsample='simple'):
    """ prepro 210x160x3 uint8 frame into 6400 (80x80) 1D float vector """
    I = I[35:195]  # crop

    if downsample == 'simple':
        I = I[::2, ::2, 0]  # downsample by factor of 2

    if (downsample == 'pil') & (color == 'gray'):
        I = np.array(Image.fromarray(I).convert(mode='L').resize((80, 80)))

    elif downsample == 'cv2':
        raise NotImplementedError

    if color == 'bin':
        I[I == 144] = 0  # erase background (background type 1)
        I[I == 109] = 0  # erase background (background type 2)
        I[I != 0] = 1  # everything else (paddles, ball) just set to 1
    if flatten:
        return I.astype(np.float).ravel()
    else:
        return I.astype(np.float)


def discount_rewards(rewards, discount_factor):
    discounted_rewards = np.zeros_like(rewards)
    for t in range(len(rewards)):
        discounted_reward_sum = 0
        discount = 1
        for k in range(t, len(rewards)):
            discounted_reward_sum += rewards[k] * discount
            discount *= discount_factor
            if rewards[k] != 0:
                # Don't count rewards from subsequent rounds
                break
        discounted_rewards[t] = discounted_reward_sum
    return discounted_rewards
