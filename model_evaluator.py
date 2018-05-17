"""
Emulator to visualize trained agent behaviour.
"""

import tensorflow as tf
import argparse
import gym
from environment_utils import prepro, bin_prepro
import numpy as np

# -- ARGUMENT PARSING ---------------------------------------
# -----------------------------------------------------------
parser = argparse.ArgumentParser()
parser.add_argument('-env', '--environment', type=str, required=True)
parser.add_argument('-id', '--identifier', type=str, required=True)
args = parser.parse_args()
# -----------------------------------------------------------
given_id = args.identifier
network_type, network_details, preprocess_type, hyperparameters = given_id.split('__')

network_details = network_details.split('_')
cnn_layers = [element for element in network_details if 'c' in element]
dense_layers = [element for element in network_details if 'f' in element]
lstm_layers = [element for element in network_details if 'l' in element]


def unwrap(string_with_parathesis):
    return string_with_parathesis[string_with_parathesis.find('(')+1:string_with_parathesis.find(')')]


preprocessors = unwrap(preprocess_type).split('_')
if 'crop' in preprocessors:
    do_crop = args.environment
else:
    do_crop = False
do_grayscale = 'gray' in preprocessors

# -----------------------------------------------------------

if args.environment == 'Pong-v0':
    action_dictionary = {
        0: 0,
        1: 2,
        2: 3
    }
elif args.environment == 'Enduro-v0':
    action_dictionary = {
        0: 1,
        1: 2,
        2: 3
        # 3: 4,
        # 4: 5,
        # 5: 6,
        # 6: 7,
        # 7: 8
    }
else:
    raise NotImplementedError
# -----------------------------------------------------------

if network_type == 'mlp':
    observations = tf.placeholder(tf.float32,
                                  [None, 80*80],
                                  name='observations')
elif network_type == 'cnn':
    if ('gray' in preprocessors) or ('bin' in preprocessors):
        observations = tf.placeholder(tf.float32,
                                      [None, None, None, 1],
                                      name='observations')
    else:
        observations = tf.placeholder(tf.float32,
                                      [None, None, None, 3],
                                      name='observations')
else:
    raise NotImplementedError

layers = []
if network_type == 'mlp':
    for idx, layer in enumerate(dense_layers):
        input_len, output_len = map(int, unwrap(layer).split('.'))
        if idx == 0:
            layers.append(
                tf.layers.dense(inputs=observations,
                                uints=output_len,
                                activation=tf.nn.relu)
            )
        else:
            layers.append(
                tf.layers.dense(inputs=layers[idx-1],
                                uints=output_len,
                                activation=tf.nn.relu)
            )

elif network_type == 'cnn':
    # observations = tf.map_fn(tf.image.per_image_standardization, observations)
    for idx, layer in enumerate(cnn_layers):
        input_channels, kernel_size, output_channels = map(int, unwrap(layer).split('.'))
        if idx == 0:
            layers.append(
                tf.layers.conv2d(inputs=observations,
                                 filters=output_channels,
                                 kernel_size=[kernel_size, kernel_size],
                                 padding='same',
                                 activation=tf.nn.relu)
            )
        else:
            layers.append(
                tf.layers.conv2d(inputs=layers[idx-1],
                                 filters=output_channels,
                                 kernel_size=[kernel_size, kernel_size],
                                 padding='same',
                                 activation=tf.nn.relu)
            )

    if ('crop' in preprocessors) & (args.environment in ['Pong-v0', 'Enduro-v0']):
        layers.append(
            tf.reshape(layers[-1], [-1, 80*80])
        )
    elif args.environment == 'Pong-v0':
        layers.append(
            tf.reshape(layers[-1], [-1, 210*160])
        )
    else:
        raise NotImplementedError

    for layer in dense_layers:
        input_len, output_len = map(int, unwrap(layer).split('.'))
        layers.append(
            tf.layers.dense(inputs=layers[-1],
                            units=output_len,
                            activation=tf.nn.relu)
        )

sample_op = tf.multinomial(logits=tf.reshape(layers[-1], shape=(1, len(action_dictionary))), num_samples=1)

sess = tf.Session()
actions = tf.placeholder(dtype=tf.int32, shape=[None], name='actions')

# -----------------------------------------------------------
env = gym.make(args.environment)
sess.run(tf.global_variables_initializer())
saver = tf.train.Saver()

with sess:
    saver.restore(sess, tf.train.latest_checkpoint('checkpoints/{}/{}'.format(args.environment, args.identifier)))

    if 'bin' in preprocessors:
        previous_pix = bin_prepro(env.reset())
    else:
        previous_pix = prepro(env.reset(), crop=do_crop, grayscale=do_grayscale, resize_to=80)

    game_state, _, done, _ = env.step(env.action_space.sample())

    while not done:
        if 'bin' in preprocessors:
            current_pix = bin_prepro(game_state)
        else:
            current_pix = prepro(game_state, crop=do_crop, grayscale=do_grayscale, resize_to=80)

        if 'diff' in preprocessors:
            observation = current_pix - previous_pix

        # Rescale input feature to [-1, 1]
        observation = \
            (current_pix - np.min(current_pix, axis=(0, 1), keepdims=True)) / (
                    np.max(current_pix, axis=(0, 1), keepdims=True) - np.min(current_pix, axis=(0, 1), keepdims=True))

        previous_pix = current_pix

        env.render()
        action = sess.run(sample_op, feed_dict={observations: [observation]})
        game_state, reward, done, info = env.step(action_dictionary[int(action)])
