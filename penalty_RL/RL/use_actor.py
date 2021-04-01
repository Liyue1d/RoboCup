#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import tensorflow as tf
import numpy as np

import argparse
import os
import datetime

from ActorNetwork import ActorNetwork
from rl_memory import RLMemory
from metadata import metadata
# from Dataset import MNIST
#
# from sklearn.decomposition import PCA
# import matplotlib.pyplot as plt


def load_model(args):

    actor = ActorNetwork(args)

    with tf.name_scope('state_input'):
        state_input = tf.placeholder(tf.float32, shape=(None,
                                                            metadata.state_size()),
                                                            name='state_input')
    with tf.name_scope('is_training'):
        is_training = tf.placeholder(tf.bool, name='is_training')

    with tf.variable_scope('networks') as scope:
        action_choice, action_parameters = actor.inference(state_input, is_training)


    # Add the variable initializer Op.
    init = tf.initialize_all_variables()

    # Create a saver for writing training checkpoints.
    saver = tf.train.Saver()

    # Create a session for running Ops on the Graph.
    sess = tf.Session()

    # Run the Op to initialize the variables.
    sess.run(init)

    saver.restore(sess, "./model.ckpt")

    def eval_actor(state):
        feed_dict = {
            state_input: state,
            is_training: False,
        }
        action_choice_ev, action_parameters_ev = sess.run([action_choice, action_parameters],
                               feed_dict=feed_dict)
        return action_choice_ev, action_parameters_ev
    return eval_actor




if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--layers", type=str, default='1024, 500, 300, 100',
                    help="Comma separated layer sizes")
    args = vars(parser.parse_args())
    load_model(args)


# if args['restore'] == 1:
#     saver.restore(sess, "./model.ckpt")
#
#         feed_dict = {
#             state_input: groups[k]
#         }
#         output = sess.run([left_output],
#                                feed_dict=feed_dict)[0]
