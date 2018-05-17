"""
Environment utility functions.
OpenCV is avoided due to bad installation.
"""

import numpy as np
from PIL import Image
import cv2


def rgb2gray(rgb):
    return np.dot(rgb[...,:3], [0.299, 0.587, 0.114])


def bin_prepro(I):
    """ prepro 210x160x3 uint8 frame into 6400 (80x80) 1D float vector """
    I = I[35:195]  # crop
    I = I[::2, ::2, 0]  # downsample by factor of 2
    I[I == 144] = 0  # erase background (background type 1)
    I[I == 109] = 0  # erase background (background type 2)
    I[I != 0] = 1  # everything else (paddles, ball) just set to 1
    return I.astype(np.float).ravel()


def prepro(image, crop='Pong-v0', grayscale=True, resize_to=80):
    """
    Args:
        image: input image with shape (W,H,Ch)
        crop: [environment..., None]
        color: ['bin', 'gray', None]
        resize_to: int

    Returns:
        preprocessed image
    """

    if crop == 'Pong-v0':
        image = image[35:195]  # crop
    elif crop is not None:
        raise NotImplementedError

    if grayscale:
        image = cv2.cvtColor(image, code=cv2.COLOR_RGB2GRAY)

    if resize_to:
        image = cv2.resize(image, (resize_to, resize_to))

    return image.astype(np.float)


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


def write_video(image_list, destination_path):
    fourcc = cv2.VideoWriter_fourcc(*'XVID')
    video_writer = cv2.VideoWriter(destination_path,
                                   fourcc,
                                   60.0,
                                   (image_list[0].shape[1], image_list[0].shape[0]))
    for image in image_list:
        video_writer.write(image)
    video_writer.release()