import numpy as np
import tensorflow as tf
import argparse
import shutil
import gym
import logging
import os
import pandas as pd
import math
from environment_utils import prepro
from network_utils import fc_layer, conv_layer

# -----------------------------------------------------------


action_dictionary = {
    0:0,
    1:2,
    2:3
}


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
