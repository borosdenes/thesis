"""
Main trianing script for pong.

Corresponding releases:
    1.0 as mlp_224_bin_b6400
    1.1 as mlp_32_bin_b6400
    2.0 as conv_132_231_gray_b6400
"""

import numpy as np
import tensorflow as tf
import argparse
import shutil
import gym
import logging
import os
import PIL

from environment_utils import prepro, discount_rewards
from network_utils import fc_layer, conv_layer


# -- ARGUMENT PARSING ---------------------------------------
# -----------------------------------------------------------
parser = argparse.ArgumentParser()
parser.add_argument('--hidden_layer_size', type=int, default=224)
parser.add_argument('--learning_rate', type=float, default=0.001)
parser.add_argument('--batch_size', type=int, required=True,
                    help='6400')
parser.add_argument('--checkpoint_every_n_episodes', type=int, default=10)
parser.add_argument('--discount_factor', type=int, default=0.99)
parser.add_argument('--render', type=int)
parser.add_argument('--summarize_every_n_episodes', type=int, default=5)
parser.add_argument('-id', '--identifier', type=str, required=True,
                    help='You can define loss type, optimizer and etc here. (e.g. xent_adam_l001_d99')
parser.add_argument('--clean', action='store_true')
parser.add_argument('--show_observation', action='store_true')
args = parser.parse_args()

# -----------------------------------------------------------

# -- CONSTANTS ----------------------------------------------
# -----------------------------------------------------------
# OBSERVATION_SIZE = 6400
BATCH_SIZE = args.batch_size
ENVIRONMENT = 'Pong-v0'

action_dictionary = {
    0:0,
    1:2,
    2:3
}

if args.clean:
    try:
        os.remove('./logs/{env}/{id}.log'.format(env=ENVIRONMENT,
                                                 id=args.identifier))
    except OSError:
        pass
    shutil.rmtree('./checkpoints/{env}/{id}'.format(env=ENVIRONMENT,
                                                    id=args.identifier),
                  ignore_errors=True)
    shutil.rmtree('./summaries/{env}/{id}'.format(env=ENVIRONMENT,
                                                  id=args.identifier),
                  ignore_errors=True)

# -- LOGGING INITIALIZER ------------------------------------
# -----------------------------------------------------------

log_formatter = logging.Formatter('%(asctime)s %(levelname)s: %(message)s')
log_path = './logs/{env}/{id}.log'.format(env=ENVIRONMENT,
                                          id=args.identifier)
if not os.path.exists(os.path.dirname(log_path)):
    os.makedirs(os.path.dirname(log_path))

logger = logging.getLogger()

file_handler = logging.FileHandler(log_path)
file_handler.setFormatter(log_formatter)
logger.addHandler(file_handler)


logger.setLevel(logging.DEBUG)
logger.debug('''Setting constants ...'
                BATCH_SIZE = {}
                ENVIRONMENT = {})'''
             .format(BATCH_SIZE, ENVIRONMENT))
# -----------------------------------------------------------

# -- INITIALIZING STEPS -------------------------------------
# -----------------------------------------------------------
sess = tf.Session()
logger.debug('Setting up training placeholders, variables and graph ...')

observations = tf.placeholder(tf.float32,
                              [None, 80, 80],
                              name='observations')
observation_images = tf.reshape(observations, [-1, 80, 80, 1])
# observations = tf.placeholder(tf.float32,
#                               [None, 6400],
#                               name='observations')

actions = tf.placeholder(dtype=tf.int32, shape=[None], name='actions')
tf.summary.histogram('actions', actions)

rewards = tf.placeholder(dtype=tf.float32, shape=[None], name='rewards')
mean_rewards = tf.placeholder(dtype=tf.float32, name='mean_rewards')
tf.summary.scalar('mean_rewards', mean_rewards)
mean_game_length = tf.placeholder(dtype=tf.float32, name='mean_game_length')
tf.summary.scalar('mean_game_length', mean_game_length)
global_step = tf.Variable(0, name='global_step', trainable=False)
increment_global_step = tf.assign_add(global_step, 1,
                                      name='increment_global_step')
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

# Y = fc_layer(input=observations,
#              size_in=6400,
#              size_out=32,
#              name='hidden_layer')
# Y_out = tf.nn.relu(Y)
# Ylogits = fc_layer(input=Y_out,
#                    size_in=32,
#                    size_out=3,
#                    name='ylogits')

sample_op = tf.multinomial(logits=tf.reshape(Ylogits, shape=(1, 3)), num_samples=1)

logger.debug('Setting up cross entropy loss ...')

with tf.name_scope('cross_entropy'):
    cross_entropy = tf.losses.softmax_cross_entropy(onehot_labels=tf.one_hot(actions, 3),
                                                    logits=Ylogits,
                                                    weights=rewards)
    tf.summary.scalar('cross_entropy', cross_entropy)


with tf.name_scope('training'):
    optimizer = tf.train.AdamOptimizer(learning_rate=args.learning_rate)
    train_op = optimizer.minimize(cross_entropy, global_step=global_step)

# -----------------------------------------------------------
env = gym.make(ENVIRONMENT)
writer = tf.summary.FileWriter('./summaries/{env}/{id}/'
                               .format(env=ENVIRONMENT, id=args.identifier))
writer.add_graph(sess.graph)
summ = tf.summary.merge_all()
saver = tf.train.Saver()

model_folder = './checkpoints/{env}/{id}'\
                 .format(env=ENVIRONMENT, id=args.identifier)


logger.info('Everything is initialized. Starting training ...')

with sess:

    model_folder = './checkpoints/{env}/{id}'\
                 .format(env=ENVIRONMENT, id=args.identifier)
    if os.path.exists(model_folder):
        try:
            logger.info('Restoring model from {} ...'.format(model_folder))
            saver.restore(sess, tf.train.latest_checkpoint(model_folder))
        except ValueError:
            raise
    else:
        sess.run(tf.global_variables_initializer())
        os.makedirs(model_folder)

    while True:
        logger.info('Starting echo #{} ...'.format(global_step.eval()))
        game_counter = 0
        _observations = []
        _actions = []
        _rewards = []

        while len(_observations) < BATCH_SIZE:
            before_counter = len(_observations)
            logger.debug('Collected {} of {} observations ({:.1f} %).'.format(before_counter,
                                                                              BATCH_SIZE,
                                                                              float(before_counter)/BATCH_SIZE*100))
            previous_pix = prepro(env.reset(), flatten=False, color='gray', downsample='pil')
            # previous_pix = prepro(env.reset())
            game_state, _, done, _ = env.step(env.action_space.sample())
            game_counter += 1

            while not done:
                current_pix = prepro(game_state, flatten=False, color='gray', downsample='pil')
                # current_pix = prepro(game_state)
                observation = current_pix - previous_pix
                if args.show_observation:
                    PIL.Image.fromarray(observation).show()
                previous_pix = current_pix

                if args.render:
                    if (global_step.eval() != 0) & (global_step.eval() % args.render == 0) & (game_counter == 1):
                        env.render()
                action = sess.run(sample_op, feed_dict={observations: [observation]})
                game_state, reward, done, info = env.step(action_dictionary[int(action)])

                _observations.append(observation)
                _actions.append(action)
                _rewards.append(reward)

            logger.debug('Episode #{} has been finished with {} data-points.'.format(game_counter,
                                                                                     len(_observations)-before_counter))

        avg_rewards = float(sum(_rewards))/game_counter
        logger.info('Played {} games. Mean reward is {:.2f}'
                    .format(game_counter, avg_rewards))

        logger.debug('Future-discounting rewards ...')
        processed_rewards = discount_rewards(_rewards, args.discount_factor)

        logger.debug('Normalizing rewards ...')
        _rewards -= np.mean(_rewards)
        _rewards /= np.std(_rewards)

        logger.debug('Training on current batch ...')
        feed_dict = {observations: _observations,
                     actions: np.squeeze(np.hstack(_actions)),
                     rewards: np.squeeze(np.hstack(processed_rewards)),
                     mean_rewards: avg_rewards,
                     mean_game_length: float(len(_actions))/game_counter}

        _, s = sess.run([train_op, summ], feed_dict=feed_dict)

        if global_step.eval() % args.summarize_every_n_episodes == 0:
            logger.debug('Summarizing (every {} epoch) ...'.format(args.summarize_every_n_episodes))
            writer.add_summary(s, global_step.eval())

        if global_step.eval() % args.checkpoint_every_n_episodes == 0:
            logger.debug('Saving checkpoint (every {} epoch) ...'.format(args.checkpoint_every_n_episodes))
            saver.save(sess,
                       os.path.join(model_folder, 'model.ckpt'))
