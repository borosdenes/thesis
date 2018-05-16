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

# -- ARGUMENT PARSING ---------------------------------------
# -----------------------------------------------------------
parser = argparse.ArgumentParser()
parser.add_argument('--environment', type=str, required=True,
                    help='Pong-v0')
parser.add_argument('--network_type', type=str, required=True,
                    help='[mlp, cnn, mlp_lstm, cnn_lstm]')
parser.add_argument('--learning_rate', type=float, default=0.001)
parser.add_argument('--batch_size', type=int, required=True,
                    help='6400')
parser.add_argument('--load', type=str, default=None,
                    help='Path to model folder to load. Must have same architecture as defined by args.')
parser.add_argument('--checkpoint_every_n_episodes', type=int, default=10)
parser.add_argument('--discount_factor', type=float, default=0.99)
parser.add_argument('--render', type=int, help='How often to render during training.')
parser.add_argument('--summarize_every_n_episodes', type=int, default=1)
parser.add_argument('-id', '--identifier', type=str, required=True,
                    help='You can define loss type, optimizer and etc here.'
                         'e.g. cnn__c132_c231_f6400-3__b6400_l001_d99')
parser.add_argument('--prepro', type=str, default='gray',
                    help='[gray, bin]')
parser.add_argument('--clean', action='store_true')
parser.add_argument('--show_observation', action='store_true')
args = parser.parse_args()

# -----------------------------------------------------------
action_dictionary = {
    0:0,
    1:2,
    2:3
}

if args.clean:
    try:
        os.remove('./logs/{env}/{id}.log'.format(env=args.environment,
                                                 id=args.identifier))
    except OSError:
        pass
    shutil.rmtree('./checkpoints/{env}/{id}'.format(env=args.environment,
                                                    id=args.identifier),
                  ignore_errors=True)
    shutil.rmtree('./summaries/{env}/{id}'.format(env=args.environment,
                                                  id=args.identifier),
                  ignore_errors=True)

# -- LOGGING INITIALIZER ------------------------------------
# -----------------------------------------------------------

log_formatter = logging.Formatter('%(asctime)s %(levelname)s: %(message)s')
log_path = './logs/{env}/{id}.log'.format(env=args.environment,
                                          id=args.identifier)
if not os.path.exists(os.path.dirname(log_path)):
    os.makedirs(os.path.dirname(log_path))

logger = logging.getLogger()
file_handler = logging.FileHandler(log_path)
file_handler.setFormatter(log_formatter)
logger.addHandler(file_handler)


logger.setLevel(logging.DEBUG)
logger.debug('''Setting up training ...'
                Environment : {}
                Batch size : {}
                Network type : {}
                (!) LOADING FROM : {}
                Training id : {}
                
                )'''
             .format(args.environment, args.batch_size, args.network_type, args.load, args.identifier))
# -----------------------------------------------------------

# -- INITIALIZING STEPS -------------------------------------
# -----------------------------------------------------------
config = tf.ConfigProto()
config.gpu_options.allow_growth = True
sess = tf.Session(config=config)
logger.debug('Setting up training placeholders, variables and graph ...')

if args.network_type == 'mlp':
    observations = tf.placeholder(tf.float32,
                                  [None, 80*80],
                                  name='observations')
elif args.network_type == 'cnn':
    observations = tf.placeholder(tf.float32,
                              [None, 80, 80],
                              name='observations')
    observation_images = tf.reshape(observations, [-1, 80, 80, 1])


actions = tf.placeholder(dtype=tf.int32, shape=[None], name='actions')
tf.summary.histogram('actions', actions)

rewards = tf.placeholder(dtype=tf.float32, shape=[None], name='rewards')
mean_rewards = tf.placeholder(dtype=tf.float32, name='mean_rewards')
tf.summary.scalar('mean_rewards', mean_rewards)
mean_game_length = tf.placeholder(dtype=tf.float32, name='mean_game_length')
tf.summary.scalar('mean_game_length', mean_game_length)

global_step = tf.Variable(0, name='global_step', trainable=False)
# -- TRAINING SETUP -----------------------------------------
# -----------------------------------------------------------
if args.network_type == 'mlp':

    Y = tf.layers.dense(inputs=observations,
                        units=256,
                        activation=tf.nn.relu)

    Ylogits = tf.layers.dense(inputs=Y,
                              units=3)


if args.network_type == 'cnn':
    conv_1 = tf.layers.conv2d(inputs=observation_images,
                              filters=2,
                              kernel_size=[3, 3],
                              padding='same',
                              activation=tf.nn.relu)

    conv_2 = tf.layers.conv2d(inputs=conv_1,
                              filters=1,
                              kernel_size=[3, 3],
                              padding='same',
                              activation=tf.nn.relu)

    flattened = tf.reshape(conv_2, [-1, 80*80])

    # hidden = tf.layers.dense(inputs=flattened,
    #                          units=256,
    #                          activation=tf.nn.relu)

    Ylogits = tf.layers.dense(inputs=flattened,
                              units=3)

# TODO: LSTM with logit output of size 3

sample_op = tf.multinomial(logits=tf.reshape(Ylogits, shape=(1, 3)), num_samples=1)

with tf.name_scope('cross_entropy'):
    cross_entropy = tf.losses.softmax_cross_entropy(onehot_labels=tf.one_hot(actions, 3),
                                                    logits=Ylogits,
                                                    weights=rewards)
    tf.summary.scalar('cross_entropy', cross_entropy)

with tf.name_scope('training'):
    optimizer = tf.train.AdamOptimizer(learning_rate=args.learning_rate)
    train_op = optimizer.minimize(cross_entropy, global_step=global_step)


env = gym.make(args.environment)
writer = tf.summary.FileWriter('./summaries/{env}/{id}/'
                               .format(env=args.environment, id=args.identifier))
writer.add_graph(sess.graph)
summ = tf.summary.merge_all()
saver = tf.train.Saver()

model_folder = './checkpoints/{env}/{id}'\
                 .format(env=args.environment, id=args.identifier)


logger.info('Everything is initialized. Starting training ...')

with sess:

    if args.load:
        logger.info('RESTORING MODEL FROM ANOTHER MODEL!\n'
                    '{}'.format(args.load))
        saver.restore(sess, tf.train.latest_checkpoint('./checkpoints/{}/{}'.format(args.environment, args.load)))


    model_folder = './checkpoints/{env}/{id}'\
                 .format(env=args.environment, id=args.identifier)
    if os.path.exists(model_folder):
        logger.info('Restoring model from {} ...'.format(model_folder))
        saver.restore(sess, tf.train.latest_checkpoint(model_folder))
    else:
        if not args.load:
            sess.run(tf.global_variables_initializer())
        os.makedirs(model_folder)

    while True:
        logger.info('Starting global_step #{} ...'.format(global_step.eval()))
        game_counter = 0
        _observations = []
        _actions = []
        _rewards = []

        while len(_observations) < args.batch_size:
            before_counter = len(_observations)
            logger.debug('Collected {} of {} observations ({:.1f} %).'.format(before_counter,
                                                                              args.batch_size,
                                                                              float(before_counter) / args.batch_size * 100))
            if args.prepro == 'gray':
                previous_pix = prepro(env.reset(), flatten=False, color='gray', downsample='pil')
            elif args.prepro == 'bin':
                previous_pix = prepro(env.reset())

            game_state, _, done, _ = env.step(env.action_space.sample())
            game_counter += 1

            while not done:
                if args.prepro == 'gray':
                    current_pix = prepro(game_state, flatten=False, color='gray', downsample='pil')
                elif args.prepro == 'bin':
                    current_pix = prepro(game_state)

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

        processed_rewards = discount_rewards(_rewards, args.discount_factor)

        processed_rewards -= np.mean(processed_rewards)
        processed_rewards /= np.std(processed_rewards)

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
