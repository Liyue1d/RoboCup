#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import math

import tensorflow as tf

class Network:

    def __init__(self, args, input_size, output_size):

        self.input_size = input_size
        self.output_size = output_size
        print("self, output_size", self.output_size)
        # Set args as attributes
        for key in args:
            setattr(self, key, args[key])

    def inference(self, inputs):
        self.inputs = inputs

        # TODO tf.contrib.layers.fully_connected
        # TODO batch norm
        layers = [int(l) for l in self.layers.split(',')]
        for i in range(0, len(layers)):
            if i == 0:
                previous_layer = self.inputs
                in_size = self.input_size
                out_size = layers[i]
            else:
                previous_layer = hidden
                in_size = layers[i-1]
                out_size = layers[i]
            with tf.variable_scope('hidden{}'.format(i)):
                weights = tf.get_variable('weights', initializer=tf.truncated_normal([in_size, out_size],
                                    stddev=1.0 / math.sqrt(float(in_size))))
                biases = tf.get_variable('biases', initializer=tf.zeros([out_size]))

                hidden = tf.nn.relu(tf.matmul(previous_layer, weights) + biases)

        with tf.variable_scope('action_choice'):
            weights = tf.get_variable('weights', initializer=tf.truncated_normal([layers[-1], 4],
                                stddev=1.0 / math.sqrt(float(layers[-1]))))
            biases = tf.get_variable('biases', initializer=tf.zeros([4]))

            action_choice = tf.matmul(hidden, weights) + biases

        with tf.variable_scope('action_parameters'):
            weights = tf.get_variable('weights', initializer=tf.truncated_normal([layers[-1], self.output_size - 4],
                                stddev=1.0 / math.sqrt(float(layers[-1]))))
            biases = tf.get_variable('biases', initializer=tf.zeros([self.output_size - 4]))

            action_parameters = tf.matmul(hidden, weights) + biases
            print("yo", action_parameters.get_shape(), self.output_size)


        return action_choice, action_parameters
