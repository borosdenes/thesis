"""
Emulator to visualize trained agent behaviour.
"""

import tensorflow as tf
import argparse
import gym
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
parser.add_argument('-mp', '--model_folder', type=str, required=True)
args = parser.parse_args()
# -----------------------------------------------------------

# -- CONSTANTS ----------------------------------------------
# -----------------------------------------------------------
ENVIRONMENT = 'Pong-v0'

# -- INITIALIZING STEPS -------------------------------------
# -----------------------------------------------------------
sess = tf.Session()

observations = tf.placeholder(tf.float32,
                              [None, 80, 80],
                              name='observations')
observation_images = tf.reshape(observations, [-1, 80, 80, 1])

actions = tf.placeholder(dtype=tf.int32, shape=[None], name='actions')


# -- TRAINING SETUP -----------------------------------------
# -----------------------------------------------------------

conv_1 = conv_layer(input=observation_images,
                    in_channels=1,
                    out_channels=2,
                    filter_size=3,
                    max_pool=False)

conv_2 = conv_layer(input=conv_1,
                    in_channels=2,
                    out_channels=1,
                    filter_size=3,
                    max_pool=False)

flattened = tf.reshape(conv_2, [-1, 80*80])

Ylogits = fc_layer(input=flattened,
                   size_in=80*80,
                   size_out=3,
                   name='ylogits')

sample_op = tf.multinomial(logits=tf.reshape(Ylogits, shape=(1, 3)), num_samples=1)
# -----------------------------------------------------------
env = gym.make(ENVIRONMENT)
sess.run(tf.global_variables_initializer())
saver = tf.train.Saver()

with sess:

    saver.restore(sess, tf.train.latest_checkpoint(args.model_folder))
    previous_pix = prepro(env.reset(), flatten=False, color='gray', downsample='pil')
    game_state, _, done, _ = env.step(env.action_space.sample())

    while not done:
        current_pix = prepro(game_state, flatten=False, color='gray', downsample='pil')
        observation = current_pix - previous_pix
        previous_pix = current_pix

        env.render()
        action = sess.run(sample_op, feed_dict={observations: [observation]})
        game_state, reward, done, info = env.step(action_dictionary[int(action)])
