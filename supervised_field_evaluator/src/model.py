#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import math

import tensorflow as tf

class Model:

    def __init__(self, args, n_inputs):
        self.args = args
        self.n_inputs = n_inputs

        # Set args as attributes
        for key in args:
            setattr(self, key, args[key])

    def inference(self):

        with tf.name_scope('inputs'):
            self.inputs_placeholder = tf.placeholder(tf.float32, shape=(None,
                                                                self.n_inputs),
                                                                name='inputs_placeholder')
        with tf.name_scope('labels'):
            self.labels_placeholder = tf.placeholder(tf.float32, shape=(None, 1),
                                                        name='inputs_placeholder')

        layers = [int(l) for l in self.layers.split(',')]
        for i in range(0, len(layers)):
            if i == 0:
                previous_layer = self.inputs_placeholder
                in_size = self.n_inputs
                out_size = layers[i]
            else:
                previous_layer = hidden
                in_size = layers[i-1]
                out_size = layers[i]
            with tf.name_scope('hidden{}'.format(i)):
                weights = tf.Variable(
                    tf.truncated_normal([in_size, out_size],
                                        stddev=1.0 / math.sqrt(float(in_size))),
                    name='weights')
                biases = tf.Variable(tf.zeros([out_size]),
                                 name='biases')
                hidden = tf.nn.relu(tf.matmul(previous_layer, weights) + biases)

        # Linear output
        with tf.name_scope('linear_output'):
            weights = tf.Variable(
                tf.truncated_normal([layers[-1], 1],
                                    stddev=1.0 / math.sqrt(float(layers[-1]))),
                name='weights')

            biases = tf.Variable(tf.zeros([1.0]),
                                 name='biases')
            self.logits = tf.matmul(hidden, weights) + biases

        # TODO int32????
        return self.logits


    def loss(self):
      """Calculates the loss from the logits and the labels.
      Args:
        logits: Logits tensor, float - [batch_size, 1].
        labels: Labels tensor, int32 - [batch_size].
      Returns:
        loss: Loss tensor of type float.
      """
      with tf.name_scope('loss'):
          squared_loss = tf.square(self.labels_placeholder - self.logits)
          self.loss = tf.reduce_mean(squared_loss, name='squared_loss_mean')
      return self.loss


    def training(self):
      """Sets up the training Ops.
      Creates a summarizer to track the loss over time in TensorBoard.
      Creates an optimizer and applies the gradients to all trainable variables.
      The Op returned by this function is what must be passed to the
      `sess.run()` call to cause the model to train.
      Args:
        loss: Loss tensor, from loss().
        learning_rate: The learning rate to use for gradient descent.
      Returns:
        train_op: The Op for training.
      """
      with tf.name_scope('Training'):

          # Create the gradient descent optimizer with the given learning rate.
          optimizer = tf.train.GradientDescentOptimizer(self.learning_rate)
          # Create a variable to track the global step.
          global_step = tf.Variable(0, name='global_step', trainable=False)
          # Use the optimizer to apply the gradients that minimize the loss
          # (and also increment the global step counter) as a single training step.
          self.train_op = optimizer.minimize(self.loss, global_step=global_step)
      return self.train_op
