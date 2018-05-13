"""
Utilities to create different type of network elements.
Low-level tensorflow is being used for the sake of modifiability and transparancy.
"""

import tensorflow as tf


def conv_layer(input, in_channels, out_channels, filter_size=5, max_pool=True, name="conv"):
    """

    Args:
        input: tensor of shape [batch, in_height, in_width, in_channels]
        in_channels:
        out_channels:
        name:

    Returns:

    """
    with tf.name_scope(name):
        w = tf.Variable(tf.truncated_normal([filter_size, filter_size, in_channels, out_channels], stddev=0.1), name="W")
        # filter / kernel tensor of shape [filter_height, filter_width, in_channels, out_channels]
        b = tf.Variable(tf.constant(0.1, shape=[out_channels]), name="B")
        conv = tf.nn.conv2d(input, w, strides=[1, 1, 1, 1], padding="SAME")
        act = tf.nn.relu(conv + b)
        tf.summary.histogram("weights", w)
        tf.summary.histogram("biases", b)
        tf.summary.histogram("activations", act)
        if max_pool:
            return tf.nn.max_pool(act, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding="SAME")
        else:
            return act


def fc_layer(input, size_in, size_out, name="fc"):
    with tf.name_scope(name):
        w = tf.Variable(tf.truncated_normal([size_in, size_out], stddev=0.1), name="W")
        b = tf.Variable(tf.constant(0.1, shape=[size_out]), name="B")
        act = tf.matmul(input, w) + b
        tf.summary.histogram("weights", w)
        tf.summary.histogram("biases", b)
        tf.summary.histogram("activations", act)
        return act
