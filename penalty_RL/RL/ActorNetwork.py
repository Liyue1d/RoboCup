#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import math

import tensorflow as tf

from metadata import metadata

class ActorNetwork:

    def __init__(self, args):

        # Set args as attributes
        for key in args:
            setattr(self, key, args[key])

    def inference(self, inputs, is_training):

        layers = [int(l) for l in self.layers.split(',')]

        print(layers)

        batch_norm_params = {'is_training': is_training, 'decay': 0.9, 'updates_collections': None}

        # TODO update collections faster
        net = tf.contrib.layers.batch_norm(inputs, **batch_norm_params)
        for i in range(len(layers)):
            net = tf.contrib.layers.fully_connected(net, num_outputs=layers[i],
                activation_fn=tf.nn.relu,
                weights_initializer=tf.contrib.layers.xavier_initializer(),
                biases_initializer=tf.constant_initializer(0.1),
                normalizer_fn=tf.contrib.layers.batch_norm,
                normalizer_params=batch_norm_params,
                scope='Layer{}'.format(i+1))


        action_choice = tf.contrib.layers.fully_connected(net, num_outputs=metadata.N_actions_choice(),
            activation_fn=None,
            weights_initializer=tf.contrib.layers.xavier_initializer(),
            biases_initializer=tf.constant_initializer(0.1),
            scope='ActionChoiceLayer')

        action_parameters = tf.contrib.layers.fully_connected(net, num_outputs=metadata.action_parameters_size(),
            activation_fn=None,
            weights_initializer=tf.contrib.layers.xavier_initializer(),
            biases_initializer=tf.constant_initializer(0.1),
            scope='ActionParameterLayer')

        return action_choice, action_parameters
