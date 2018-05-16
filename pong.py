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
parser.add_argument('-id', '--identifier', type=str, required=True,
                    help='''
                    You can define loss type, optimizer and etc here.
                    Format: networktype__network_details__preprotype__hyperparams
                    
                    cnn__c(1.3.2)_c(2.3.1)_f(6400.3)__prepro(gray)__b(6400)_l(0.001)_d(0.99)
                    cnn__c(3.3.10)_c(10.3.1)_f(6400.3)__prepro(None)__b(6400)_l(0.001)_d(0.99)
                    mlp__f(6400.256)_f(256.3)__prepro(bin)__b(6400)_l(0.001)_d(0.99)''')

parser.add_argument('--environment', type=str, required=True,
                    help='Pong-v0')
parser.add_argument('--load', type=str, default=None,
                    help='Path to model folder to load. Must have same architecture as defined by args.')
parser.add_argument('--checkpoint_every_n_episodes', type=int, default=10)
parser.add_argument('--render', type=int, help='How often to render during training.')
parser.add_argument('--summarize_every_n_episodes', type=int, default=1)
parser.add_argument('--clean', action='store_true')
args = parser.parse_args()

if args.environment not in ['Pong-v0']:
    raise NotImplementedError

# -- VARIABLE INITIALIZER BASED ON GIVEN ID -----------------
# -----------------------------------------------------------
given_id = args.identifier
network_type, network_details, preprocess_type, hyperparameters = given_id.split('__')

network_details = network_details.split('_')
cnn_layers = [element for element in network_details if 'c' in element]
dense_layers = [element for element in network_details if 'f' in element]
lstm_layers = [element for element in network_details if 'l' in element]

if len(lstm_layers) != 0:
    raise NotImplementedError


def unwrap(string_with_parathesis):
    return string_with_parathesis[string_with_parathesis.find('(')+1:string_with_parathesis.find(')')]


preprocess_type = unwrap(preprocess_type)
if preprocess_type == 'None':
    preprocess_type = None

batch_size, learning_rate, discount_rate = map(unwrap, hyperparameters.split('_'))
batch_size = int(batch_size)
learning_rate = float(learning_rate)
discount_rate = float(discount_rate)

# -----------------------------------------------------------
if args.environment == 'Pong-v0':
    action_dictionary = {
        0: 0,
        1: 2,
        2: 3
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
             .format(args.environment, batch_size, network_type, args.load, args.identifier))
# -----------------------------------------------------------

# -- INITIALIZING STEPS -------------------------------------
# -----------------------------------------------------------
config = tf.ConfigProto()
config.gpu_options.allow_growth = True
sess = tf.Session(config=config)
logger.debug('Setting up training placeholders, variables and graph ...')

if network_type == 'mlp':
    observations = tf.placeholder(tf.float32,
                                  [None, 80*80],
                                  name='observations')
elif network_type == 'cnn':
    if preprocess_type == 'gray':
        observations = tf.placeholder(tf.float32,
                                      [None, None, None, 1],
                                      name='observations')
    elif preprocess_type == None:
        observations = tf.placeholder(tf.float32,
                                      [None, None, None, 3],
                                      name='observations')
    else:
        raise NotImplementedError

    # observation_images = tf.reshape(observations, [-1, 80, 80, 1])


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

    if preprocess_type == 'gray':
        layers.append(
            layers.reshape(layers[-1], [-1, 80*80])
        )
    elif preprocess_type == None:
        layers.append(
            tf.reshape(layers[-1], [-1, 210*160])
        )

    for layer in dense_layers:
        input_len, output_len = map(int, unwrap(layer).split('.'))
        layers.append(
            tf.layers.dense(inputs=layers[-1],
                            units=output_len,
                            activation=tf.nn.relu)
        )

# TODO: LSTM with logit output of size 3

sample_op = tf.multinomial(logits=tf.reshape(layers[-1], shape=(1, 3)), num_samples=1)

with tf.name_scope('cross_entropy'):
    cross_entropy = tf.losses.softmax_cross_entropy(onehot_labels=tf.one_hot(actions, 3),
                                                    logits=layers[-1],
                                                    weights=rewards)
    tf.summary.scalar('cross_entropy', cross_entropy)

with tf.name_scope('training'):
    optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate)
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

        while len(_observations) < batch_size:
            before_counter = len(_observations)
            logger.debug('Collected {} of {} observations ({:.1f} %).'.format(before_counter,
                                                                              batch_size,
                                                                              float(before_counter) / batch_size * 100))
            if preprocess_type == 'gray':
                previous_pix = prepro(env.reset(), flatten=False, color='gray', downsample='pil')
            elif preprocess_type == 'bin':
                previous_pix = prepro(env.reset())
            elif preprocess_type != None:
                raise NotImplementedError
            else:
                previous_pix = env.reset()

            game_state, _, done, _ = env.step(env.action_space.sample())
            game_counter += 1

            while not done:
                if preprocess_type == 'gray':
                    current_pix = prepro(game_state, flatten=False, color='gray', downsample='pil')
                elif preprocess_type == 'bin':
                    current_pix = prepro(game_state)
                elif preprocess_type != None:
                    raise NotImplementedError
                else:
                    current_pix = game_state

                observation = current_pix
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

        processed_rewards = discount_rewards(_rewards, discount_rate)

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
