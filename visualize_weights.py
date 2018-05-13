"""
Script to visualize the weights of MLP (optionally CNN)
TODO: weighted sum of hidden neurons given by output weights to better visualize linear combination of hidden weights
"""

import tensorflow as tf
import PIL
import numpy as np
import os

model_id = 'mlp_32_bin_b6400'

sess = tf.Session()
saver = tf.train.import_meta_graph('./checkpoints/Pong-v0/{}/model.ckpt.meta'.format(model_id))
saver.restore(sess, tf.train.latest_checkpoint('./checkpoints/Pong-v0/{}'.format(model_id)))
graph = tf.get_default_graph()


with sess:
    hidden_neurons = graph.get_tensor_by_name('hidden_layer/W:0').eval()
    hidden_neurons = [hidden_neurons[:, idx].reshape(80, 80) for idx in range(hidden_neurons.shape[1])]
    output_neurons = graph.get_tensor_by_name('ylogits/W:0').eval()
    output_neurons = [output_neurons[:, idx].reshape(4, 8) for idx in range(output_neurons.shape[1])]

save_folder_hidden = 'visualization/{}/hidden'.format(model_id)
for idx, hidden_neuron in enumerate(hidden_neurons):
    rescaled_neuron = np.interp(hidden_neuron, (hidden_neuron.min(), hidden_neuron.max()), (0, 255)).astype(np.uint8)

    if not os.path.exists(save_folder_hidden):
        os.makedirs(save_folder_hidden)
    PIL.Image.fromarray(rescaled_neuron).save(os.path.join(save_folder_hidden, '{}.png'.format(idx)))

save_folder_output = 'visualization/{}/output'.format(model_id)
for idx, output_neuron in enumerate(output_neurons):
    rescaled_neuron = np.interp(output_neuron, (output_neuron.min(), output_neuron.max()), (0, 255)).astype(np.uint8)

    if not os.path.exists(save_folder_output):
        os.makedirs(save_folder_output)
    PIL.Image.fromarray(rescaled_neuron).save(os.path.join(save_folder_output, '{}.png'.format(idx)))

