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


def main(args):

    # rl_memory = RLMemory('penalty_agent2D2.data')
    rl_memory = RLMemory('penalty_with_random.data')
    print("get_memory_size", rl_memory.get_memory_size())

    # s, a, r, s_prime = rl_memory.sample_batch(1)[0]
    #
    # print("s", s)
    # print("a", a)
    # print("r", r)
    # print("s_prime", s_prime)

    # TODO state size, action size
    actor = ActorNetwork(args)
    print("action_size", metadata.action_size())

    with tf.name_scope('state_input'):
        state_input = tf.placeholder(tf.float32, shape=(None,
                                                            metadata.state_size()),
                                                            name='state_input')
    with tf.name_scope('action_choice_label'):
        action_choice_label = tf.placeholder(tf.float32, shape=(None,
                                                            metadata.N_actions_choice()),
                                                            name='action_choice_label')
    with tf.name_scope('action_parameters_label'):
        action_parameters_label = tf.placeholder(tf.float32, shape=(None,
                                                            metadata.action_parameters_size()),
                                                            name='action_parameters_label')

    with tf.name_scope('is_training'):
        is_training = tf.placeholder(tf.bool, name='is_training')

    with tf.variable_scope('networks') as scope:
        action_choice, action_parameters = actor.inference(state_input, is_training)

    action_choice_softmax = tf.nn.softmax(action_choice)
    # action_choice = tf.gather(action_output, [0, 1, 2, 3])
    # print("action_output.get_shape()", actu.get_shape())
    # action_choice = tf.slice(action_output, 0, 4)
    print("action_choice.get_shape()", action_choice.get_shape())
    print("action_choice_label.get_shape()", action_choice_label.get_shape())
    # exit(0)
    # action_parameters = tf.gather(action_output, [i for i in range(4, rl_memory.action_size)])

    # TODO check that should be minimized or maximized
    action_cross_entropy = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(action_choice, action_choice_label))

    print("action_parameters.get_shape()", action_parameters.get_shape())
    print("action_parameters_label.get_shape()", action_parameters_label.get_shape())

    with tf.variable_scope("BN_loss") as scope:
        batch_norm_params = {'is_training': False, 'decay': 0.9, 'updates_collections': None, 'scope': scope}
        # TODO update collections faster
        action_parameters_bn = tf.contrib.layers.batch_norm(action_parameters, **batch_norm_params)
        scope.reuse_variables()
        batch_norm_params = {'is_training': is_training, 'decay': 0.9, 'updates_collections': None, 'scope': scope}
        action_parameters_label_bn = tf.contrib.layers.batch_norm(action_parameters_label, **batch_norm_params)

    # action_parameters_bn = tf.Print(action_parameters_bn, ['-------------------------------------'])
    # action_parameters_bn = tf.Print(action_parameters_bn, ['action_parameters', action_parameters], summarize=8)
    # action_parameters_bn = tf.Print(action_parameters_bn, ['action_parameters_bn', action_parameters_bn], summarize=8)
    # action_parameters_bn = tf.Print(action_parameters_bn, ['action_parameters_label', action_parameters_label], summarize=8)
    # action_parameters_bn = tf.Print(action_parameters_bn, ['action_parameters_label_bn', action_parameters_label_bn], summarize=8)



    mask = [
        [True, True, False, False, False, False, False, True], # DASH parameters
        [False, False, True, True, False, False, False, True], # KICK parameters
        [False, False, False, False, True, True, False, True], # MOVE parameters
        [False, False, False, False, False, False, True, True], # TURN parameters
    ]
    a = tf.where(tf.cast(tf.to_int32(action_choice_label), tf.bool))
    # action_parameters = tf.Print(action_parameters, ['action_choice_label', action_choice_label, tf.shape(action_choice_label)], summarize=10)
    action_indice = a[:, 1]
    # action_parameters = tf.Print(action_parameters, ['action_indice', action_indice, tf.shape(action_indice)], summarize=10)
    params_mask = tf.gather(mask, action_indice)

    weight = [[1/4.58290528e-01 for _ in range(8)],
                [1/1.04298336e-01 for _ in range(8)],
                [0 for _ in range(8)],
                [1/4.29648654e-01 for _ in range(8)]] # TODO
    params_weight = tf.gather(weight, action_indice)

    # params_mask = mask[0] # TODO

    # action_parameters_subset = tf.boolean_mask(params_weight*action_parameters_bn, params_mask)
    # action_parameters_label_subset = tf.boolean_mask(params_weight*action_parameters_label_bn, params_mask)
    action_parameters_subset = action_parameters_bn
    action_parameters_label_subset = action_parameters_label_bn

    parameters_loss = tf.reduce_mean(tf.square(action_parameters_subset - action_parameters_label_subset))


    print("action_cross_entropy.get_shape()", action_cross_entropy.get_shape())
    print("parameters_loss.get_shape()", parameters_loss.get_shape())
    mu = 0.5
    loss = (1-mu)*action_cross_entropy +  mu*parameters_loss

    optimizer = tf.train.AdamOptimizer(args['learning_rate'])

    # actor_output = tf.concat(1, [action_choice, action_parameters])
    # grads = tf.gradients(loss, [action_choice, action_parameters])
    #
    # # HIGH = np.array([100 for _ in range(metadata.action_parameters_size())])
    # # LOW = np.array([-100 for _ in range(metadata.action_parameters_size())])
    # # if_pos = (HIGH - action_parameters)/(HIGH - LOW)
    # # if_neg = (action_parameters - LOW)/(HIGH - LOW)
    # # TODO why -grads[1]?
    #
    # # grads[1] = tf.select(tf.equal(grads[1], tf.zeros_like(grads[1])), tf.ones_like(grads[1]), grads[1])
    # # inverted_gradient = tf.select(tf.greater(grads[1], tf.zeros_like(grads[1])), if_pos, if_neg)
    #
    #
    # loss = tf.Print(loss, ['---------------------------------------------------------------------'])
    # loss = tf.Print(loss, ['action_parameters', action_parameters[0,:], tf.shape(action_parameters[0,:])], summarize=8)
    # # loss = tf.Print(loss, ['grads[1]', grads[1][0,:], tf.shape(grads[1][0,:])], summarize=8)
    # # loss = tf.Print(loss, ['greater', tf.greater(grads[1], tf.zeros_like(grads[1]))[0,:], tf.shape(tf.greater(grads[1], tf.zeros_like(grads[1]))[0,:])], summarize=8)
    # # loss = tf.Print(loss, ['if_pos', if_pos[0,:], tf.shape(if_pos[0,:])], summarize=8)
    # # loss = tf.Print(loss, ['if_neg', if_neg[0,:], tf.shape(if_neg[0,:])], summarize=8)
    # # loss = tf.Print(loss, ['inverted_gradient', inverted_gradient[0,:], tf.shape(inverted_gradient[0,:])], summarize=8)
    # #
    # #
    # # # grads[1] = grads[1]*inverted_gradient
    # #
    # # loss = tf.Print(loss, ['final', grads[1][0,:], tf.shape(grads[1][0,:])], summarize=8)
    #
    # grad_concat = tf.concat(1, [grads[0], grads[1]])
    # train_op = optimizer.minimize(actor_output, grad_loss=grad_concat)

    train_op = optimizer.minimize(loss)

    loss_summary = tf.scalar_summary('training/loss', loss)
    loss_summary = tf.scalar_summary('details/parameters_loss', parameters_loss)
    loss_summary = tf.scalar_summary('details/action_cross_entropy', action_cross_entropy)
    merged_train_summaries = tf.merge_all_summaries()

    # Add the variable initializer Op.
    init = tf.initialize_all_variables()

    # Create a saver for writing training checkpoints.
    saver = tf.train.Saver()

    # Create a session for running Ops on the Graph.
    sess = tf.Session()

    # Instantiate a SummaryWriter to output summaries and the Graph.
    date = datetime.datetime.now().strftime('%d-%m-%y@%H-%M')
    name = "{date}@".format(date=date)
    for arg in sorted(args.keys()):
        name += "{}={},".format(arg, args[arg])
    summary_writer = tf.train.SummaryWriter(os.path.join(args['train_dir'], name), sess.graph)

    # Run the Op to initialize the variables.
    sess.run(init)
    #
    for step in range(10**20):
        # start_time = time.time()
        batch = rl_memory.sample_batch(args['batch_size'])
        # print("b", batch.shape)
        # print("len batch", len(batch))
        s = []
        a_c = []
        a_p = []
        for i in range(len(batch)):
            s.append(batch[i][0])
            a_c.append(batch[i][1][0:4])
            # print("ac", i, batch[i][1][0:4])
            a_p.append(batch[i][1][4:])
            # print("ap", batch[i][1][4:])

        feed_dict = {
            state_input: s,
            action_choice_label: a_c,
            action_parameters_label: a_p,
            is_training: True,
        }
        _, loss_value, action_parameters_ev, action_choice_ev = sess.run([train_op, loss, action_parameters, action_choice_softmax],
                               feed_dict=feed_dict)

        # duration = time.time() - start_time
        if step%100 == 0:
            print(step, loss_value)
            def printer(v):
                string = ""
                for i in range(len(v)):
                    string += "{:>8.2f}".format(v[i])
                return string

            p = np.argmax(action_choice_ev[0])
            q = np.argmax(a_c[0])
            print("what", p, a_c[0], action_choice_ev[0])
            if p == q:
                print("CORRECT!")
            else:
                print("NOPE!")
            print("actor", printer(action_parameters_ev[0]))
            print("target", printer(a_p[0]))
            summary = sess.run(merged_train_summaries, feed_dict=feed_dict)
            summary_writer.add_summary(summary, step)
            summary_writer.flush()
            saver.save(sess, "./model.ckpt")

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--learning_rate", type=float, default=0.0001,
                    help="Initial learning rate.")
    parser.add_argument("--batch_size", type=int, default=512,
                    help="Batch size.")
    # parser.add_argument("--positive_ratio", type=float, default=0.5,
    #                 help="Ratio of positive pair in each batch.")
    parser.add_argument("--train_dir", type=str, default='data/',
                    help="Path to the train dir.")
    # parser.add_argument("--layers", type=str, default='1024, 500, 300, 100',
    parser.add_argument("--layers", type=str, default='2048, 2048, 1024, 1024, 1024',
                    help="Comma separated layer sizes")
    # parser.add_argument("--output_size", type=int, default=20,
    #                 help="Size of the embedding space.")
    # parser.add_argument("--restore", type=int, default=0,
    #                 help="Restore.")
    args = vars(parser.parse_args())
    main(args)


# if args['restore'] == 1:
#     saver.restore(sess, "./model.ckpt")
#
#         feed_dict = {
#             state_input: groups[k]
#         }
#         output = sess.run([left_output],
#                                feed_dict=feed_dict)[0]
