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
parser.add_argument('--learning_rate', type=float, default=0.001)
parser.add_argument('--batch_size', type=int, default=6400)
parser.add_argument('--checkpoint_every_n_episodes', type=int, default=10)
parser.add_argument('--discount_factor', type=int, default=0.99)
parser.add_argument('--render', action='store_true')
parser.add_argument('--summarize_every_n_episodes', type=int, default=5)
parser.add_argument('-id', '--identifier', type=str, required=True,
                    help='You can define loss type, optimizer and etc here. (e.g. xent_adam_l001_d99')
args = parser.parse_args()

generated_identifier = \
    '_'.join([str(args.hidden_layer_size), str(args.batch_size), args.identifier])
# -----------------------------------------------------------

# -- CONSTANTS ----------------------------------------------
# -----------------------------------------------------------
OBSERVATION_SIZE = 6400
BATCH_SIZE = args.batch_size
ENVIRONMENT = 'Pong-v0'

# -- LOGGING INITIALIZER ------------------------------------
# -----------------------------------------------------------

log_formatter = logging.Formatter('%(asctime)s %(levelname)s: %(message)s')
log_path = './logs/{env}/{id}.log'.format(env=ENVIRONMENT,
                                          id=generated_identifier)
if not os.path.exists(os.path.dirname(log_path)):
    os.makedirs(os.path.dirname(log_path))

logger = logging.getLogger()

file_handler = logging.FileHandler(log_path)
file_handler.setFormatter(log_formatter)
logger.addHandler(file_handler)


logger.setLevel(logging.DEBUG)
logger.debug('''Setting constants ...'
                OBSERVATION_SIZE = {}
                BATCH_SIZE = {}
                ENVIRONMENT = {})'''
             .format(OBSERVATION_SIZE, BATCH_SIZE, ENVIRONMENT))
# -----------------------------------------------------------

# -- INITIALIZING STEPS -------------------------------------
# -----------------------------------------------------------
sess = tf.Session()

logger.debug('Setting up training placeholders, variables and graph ...')
observations = tf.placeholder(tf.float32,
                              [None, OBSERVATION_SIZE],
                              name='observations')
actions = tf.placeholder(dtype=tf.int32, shape=[None], name='actions')
tf.summary.histogram('actions', actions)
rewards = tf.placeholder(dtype=tf.float32, shape=[None], name='rewards')
mean_rewards = tf.placeholder(dtype=tf.float32, name='mean_rewards')
mean_game_length = tf.placeholder(dtype=tf.float32, name='mean_game_length')
tf.summary.scalar('mean_rewards', mean_rewards)


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

logger.debug('Setting up cross entropy loss ...')

with tf.name_scope('cross_entropy'):
    cross_entropy = tf.losses.softmax_cross_entropy(onehot_labels=tf.one_hot(actions, 3),
                                                    logits=Ylogits,
                                                    weights=rewards)
    tf.summary.scalar('cross_entropy', cross_entropy)

# with tf.name_scope('log_loss'):
#     loss = tf.losses.log_loss(
#         labels=actions,
#         predictions=Ylogits,
#         weights=rewards)
#     tf.summary.scalar('log_loss', loss)

with tf.name_scope('training'):
    # optimizer = tf.train.RMSPropOptimizer(learning_rate=0.005, decay=0.99)
    optimizer = tf.train.AdamOptimizer(learning_rate=args.learning_rate)
    train_op = optimizer.minimize(cross_entropy)

# -----------------------------------------------------------
env = gym.make(ENVIRONMENT)
sess.run(tf.global_variables_initializer())
writer = tf.summary.FileWriter('./summaries/{env}/{id}/'
                               .format(env=ENVIRONMENT, id=generated_identifier))
writer.add_graph(sess.graph)
summ = tf.summary.merge_all()
saver = tf.train.Saver()

model_folder = './checkpoints/{env}/{id}'\
                 .format(env=ENVIRONMENT, id=generated_identifier)


logger.info('Everything is initialized. Starting training ...')

with sess:

    model_folder = './checkpoints/{env}/{id}'\
                 .format(env=ENVIRONMENT, id=generated_identifier)
    if os.path.exists(model_folder):
        logger.info('Restoring model from {} ...'.format(model_folder))
        saver.restore(sess, tf.train.latest_checkpoint(model_folder))
    else:
        os.makedirs(model_folder)

    echo_no = 0
    while True:
        game_counter = 0
        _observations = []
        _actions = []
        _rewards = []

        while len(_observations) < BATCH_SIZE:
            before_counter = len(_observations)
            logger.debug('Collected {} of {} observations ({:.1f} %).'.format(before_counter,
                                                                              BATCH_SIZE,
                                                                              float(before_counter)/BATCH_SIZE*100))
            previous_pix = prepro(env.reset())
            game_state, _, done, _ = env.step(env.action_space.sample())
            game_counter += 1

            while not done:
                current_pix = prepro(game_state)
                observation = current_pix - previous_pix
                previous_pix = current_pix

                if args.render:
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
        feed_dict = {observations: np.squeeze(np.vstack(_observations)),
                     actions: np.squeeze(np.hstack(_actions)),
                     rewards: np.squeeze(np.hstack(processed_rewards)),
                     mean_rewards: avg_rewards,
                     mean_game_length: float(len(_actions))/game_counter}

        # [train_accuracy, s] = sess.run([accuracy, summ], feed_dict=feed_dict)
        _, s = sess.run([train_op, summ], feed_dict=feed_dict)

        if echo_no % args.summarize_every_n_episodes == 0:
            writer.add_summary(s, echo_no)

        if echo_no % args.checkpoint_every_n_episodes == 0:
            saver.save(sess, model_folder+'/')

        echo_no += 1
