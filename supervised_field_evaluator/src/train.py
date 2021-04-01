#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import time
import datetime
import argparse

import tensorflow as tf

from model import Model
from input_data import H5pyDataset


class Trainer:
    def __init__(self, args, dataset, model):

        # Set args as attributes
        for key in args:
            setattr(self, key, args[key])

        self.dataset = dataset
        self.model = model



    def fill_feed_dict(self, inputs, labels):
        feed_dict = {
            self.model.inputs_placeholder: inputs,
            self.model.labels_placeholder: labels,
        }
        return feed_dict

    def run(self):

        with tf.Graph().as_default():

            # Build a Graph that computes predictions from the inference model.
            self.model.inference()

            # Add to the Graph the Ops for loss calculation.
            loss = self.model.loss()

            # Add a scalar summary for the snapshot loss.
            loss_summary = tf.scalar_summary(loss.op.name, loss)

            # Add to the Graph the Ops that calculate and apply gradients.
            train_op = self.model.training()


            # Add the variable initializer Op.
            init = tf.initialize_all_variables()

            # Create a saver for writing training checkpoints.
            saver = tf.train.Saver()

            # Create a session for running Ops on the Graph.
            sess = tf.Session()

            # Instantiate a SummaryWriter to output summaries and the Graph.
            run_name = datetime.datetime.now().strftime('%H-%M@')+"{}_b={}_s={}".format(self.layers, self.batch_size, self.learning_rate)
            summary_writer = tf.train.SummaryWriter("{}/{}".format(self.train_dir, run_name), sess.graph)

            # And then after everything is built:

            # Run the Op to initialize the variables.
            sess.run(init)

            # Start the training loop.
            for step in range(self.max_steps):
              start_time = time.time()

              # Fill a feed dictionary with the actual set of images and labels
              # for this particular training step.
            #   inputs, labels =
              feed_dict = self.fill_feed_dict(*self.dataset.train.next_batch(self.batch_size))

              # Run one step of the model.  The return values are the activations
              # from the `train_op` (which is discarded) and the `loss` Op.  To
              # inspect the values of your Ops or variables, you may include them
              # in the list passed to sess.run() and the value tensors will be
              # returned in the tuple from the call.
              _, loss_value = sess.run([train_op, loss],
                                       feed_dict=feed_dict)

              duration = time.time() - start_time

              # Write the summaries and print an overview fairly often.
              if step % 100 == 0:
                # Print status to stdout.
                print('Step %d: loss = %.2f (%.3f sec)' % (step, loss_value, duration))
                # Update the events file.
                summary_writer.add_summary(sess.run(loss_summary, feed_dict=feed_dict), step)
                summary_writer.flush()

              # Save a checkpoint and evaluate the model periodically.
              if (step) % 1000 == 0 or (step + 1) == self.max_steps:
                saver.save(sess, self.train_dir, global_step=step)
                # Evaluate against the training set.
                print('Training Data Eval:')
                start_time = time.time()
                training_error_summary = tf.scalar_summary('Training set error', loss)
                inputs, labels = self.dataset.train.all_dataset()
                feed_dict = self.fill_feed_dict(inputs, labels)
                summary_writer.add_summary(sess.run(training_error_summary, feed_dict=feed_dict), step)
                duration = time.time() - start_time
                print("duration: {:.3f}/{}={:.4f}.10-3".format(duration, labels.shape[0], 1000*(duration/labels.shape[0])))

                print("Test Data Eval:")
                test_error_summary = tf.scalar_summary('Test set error', loss)
                feed_dict = self.fill_feed_dict(*self.dataset.test.all_dataset())
                summary_writer.add_summary(sess.run(test_error_summary, feed_dict=feed_dict), step)


                summary_writer.flush()

def main():

    parser = argparse.ArgumentParser()
    parser.add_argument("--learning_rate", type=float, default=0.0001,
                    help="Initial learning rate.")
    parser.add_argument("--max_steps", type=int, default=2000000,
                    help="Number of steps to run trainer.")
    # parser.add_argument("-hidden1_units", type=int, default=128,
    #                 help="Number of units in hidden layer 1.")
    # parser.add_argument("-hidden2_units", type=int, default=128,
    #                 help="Number of units in hidden layer 2.")
    parser.add_argument("--batch_size", type=int, default=512,
                    help="Batch size.")
    parser.add_argument("--data_file", type=str, default='data.hdf5',
                    help="Path to the data file.")
    parser.add_argument("--train_dir", type=str, default='data/',
                    help="Path to the train dir.")
    parser.add_argument("--layers", type=str, default='128,128',
                    help="Comma separated layer sizes")
    parser.add_argument("--n_sample_normalization", type=int, default=10000,
                    help="Number of samples to compute normalization")
    parser.add_argument("--queue_size", type=int, default=10000,
                    help="Split dataset queue size")

    args = vars(parser.parse_args())
    print(args['layers'], args['layers'].split(','))
    # with H5pyDatasetInMemory(args['data_file']) as dataset:
    #     model = Model(args, dataset.n_inputs)
    #     trainer = Trainer(args, dataset, model)
    #     trainer.run()
    with H5pyDataset(args['data_file'], args['n_sample_normalization'], args['queue_size']) as dataset:
        model = Model(args, dataset.n_inputs)
        trainer = Trainer(args, dataset, model)
        trainer.run()
        pass

if __name__ == '__main__':
    main()
