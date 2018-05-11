import numpy as np
import tensorflow as tf
import argparse
import shutil
import gym
import logging
import os
import pandas as pd
import math


# -----------------------------------------------------------


def prepro(I):
    """ prepro 210x160x3 uint8 frame into 6400 (80x80) 1D float vector """
    I = I[35:195]  # crop
    I = I[::2, ::2, 0]  # downsample by factor of 2
    I[I == 144] = 0  # erase background (background type 1)
    I[I == 109] = 0  # erase background (background type 2)
    I[I != 0] = 1  # everything else (paddles, ball) just set to 1
    return I.astype(np.float).ravel()


def discount_rewards(rewards, discount_factor):
    discounted_rewards = np.zeros_like(rewards)
    normalized_zero = rewards[0]
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


action_dictionary = {
    0:0,
    1:2,
    2:3
}


# -----------------------------------------------------------


def conv_layer(input, size_in, size_out, name="conv"):
    with tf.name_scope(name):
        w = tf.Variable(tf.truncated_normal([5, 5, size_in, size_out], stddev=0.1), name="W")
        b = tf.Variable(tf.constant(0.1, shape=[size_out]), name="B")
        conv = tf.nn.conv2d(input, w, strides=[1, 1, 1, 1], padding="SAME")
        act = tf.nn.relu(conv + b)
        tf.summary.histogram("weights", w)
        tf.summary.histogram("biases", b)
        tf.summary.histogram("activations", act)
        return tf.nn.max_pool(act, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding="SAME")


def fc_layer(input, size_in, size_out, name="fc"):
    with tf.name_scope(name):
        w = tf.Variable(tf.truncated_normal([size_in, size_out], stddev=0.1), name="W")
        b = tf.Variable(tf.constant(0.1, shape=[size_out]), name="B")
        act = tf.matmul(input, w) + b
        tf.summary.histogram("weights", w)
        tf.summary.histogram("biases", b)
        tf.summary.histogram("activations", act)
        return act


def calculate_cross_entropy(one_hot,logit,weights=None):
    one_hot = pd.DataFrame(one_hot).astype(bool)
    logit = pd.DataFrame(logit)
    if weights:
        weights = pd.Series(weights)
        return logit.mask(one_hot).sum(axis=1)*weights
    else:
        return logit.mask(one_hot).sum(axis=1)


# -- ARGUMENT PARSING ---------------------------------------
# -----------------------------------------------------------
parser = argparse.ArgumentParser()
parser.add_argument('--hidden_layer_size', type=int, default=224)
parser.add_argument('-mp', '--model_folder', type=str, required=True)
args = parser.parse_args()
# -----------------------------------------------------------

# -- CONSTANTS ----------------------------------------------
# -----------------------------------------------------------
OBSERVATION_SIZE = 6400
ENVIRONMENT = 'Pong-v0'

# -- INITIALIZING STEPS -------------------------------------
# -----------------------------------------------------------
sess = tf.Session()

observations = tf.placeholder(tf.float32,
                              [None, OBSERVATION_SIZE],
                              name='observations')

# -- TRAINING SETUP -----------------------------------------
# -----------------------------------------------------------
Y = fc_layer(input=observations,
             size_in=OBSERVATION_SIZE,
             size_out=args.hidden_layer_size,
             name='hidden_layer')
Y_out = tf.nn.relu(Y)
Ylogits = fc_layer(input=Y_out,
                   size_in=args.hidden_layer_size,
                   size_out=3,
                   name='ylogits')

sample_op = tf.multinomial(logits=tf.reshape(Ylogits, shape=(1, 3)), num_samples=1)
# -----------------------------------------------------------
env = gym.make(ENVIRONMENT)
sess.run(tf.global_variables_initializer())
saver = tf.train.Saver()

with sess:

    saver.restore(sess, tf.train.latest_checkpoint(args.model_folder))
    previous_pix = prepro(env.reset())
    game_state, _, done, _ = env.step(env.action_space.sample())

    while not done:
        current_pix = prepro(game_state)
        observation = current_pix - previous_pix
        previous_pix = current_pix

        env.render()
        action = sess.run(sample_op, feed_dict={observations: [observation]})
        game_state, reward, done, info = env.step(action_dictionary[int(action)])
