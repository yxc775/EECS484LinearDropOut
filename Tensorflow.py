from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import argparse
import sys
import time

from tensorflow.examples.tutorials.mnist import input_data

import tensorflow as tf

FLAGS = None


def main(_):
    logs_path = 'path/to/logs'
    # Import data
    mnist = input_data.read_data_sets(FLAGS.data_dir, one_hot=True)
    start_time = time.time()
    with tf.name_scope('input'):
      # Create the model
      x = tf.placeholder(tf.float32, [None, 784])

      # Input layer
    with tf.name_scope('input_layer_weight'):
      W2 = tf.Variable(tf.random_uniform([784, 392], minval=-1, maxval=1, dtype=tf.float32))
    with tf.name_scope('input_layer_bias'):
      b2 = tf.Variable(tf.random_uniform([392], minval=-1, maxval=1, dtype=tf.float32))
    with tf.name_scope('input_layer_output'):
      y2 = tf.nn.sigmoid(tf.matmul(x, W2) + b2)

      # Drop out layers
    with tf.name_scope('drop_out'):
      keep_prob = tf.placeholder(tf.float32)
      y_drop = tf.nn.dropout(y2, keep_prob)

      # Output Layer
    with tf.name_scope('output_layer_weight'):
      W1 = tf.Variable(tf.zeros([392, 10]))
    with tf.name_scope('output_layer_bias'):
      b1 = tf.Variable(tf.zeros([10]))
    with tf.name_scope('output_layer_output'):
      y1 = tf.nn.softmax(tf.matmul(y_drop, W1) + b1)

    with tf.name_scope('Training_set'):
      # Define loss and optimizer
      y_ = tf.placeholder(tf.float32, [None, 10])

    with tf.name_scope('cross_entropy'):
      cross_entropy = tf.reduce_mean(-tf.reduce_sum(y_ * tf.log(y1),reduction_indices=[1]))
      train_step = tf.train.GradientDescentOptimizer(0.5).minimize(cross_entropy)
      sess = tf.InteractiveSession()
      writer = tf.summary.FileWriter(logs_path, graph = sess.graph)
      tf.global_variables_initializer().run()
      # Train
      for _ in range(2000):
        batch_xs, batch_ys = mnist.train.next_batch(100)
        sess.run(train_step, feed_dict={x: batch_xs, y_: batch_ys,keep_prob: 1})

      # Test trained model
      correct_prediction = tf.equal(tf.argmax(y1, 1), tf.argmax(y_, 1))
      accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
      writer.close()
      print(sess.run(accuracy, feed_dict={x: mnist.test.images,
                                          y_: mnist.test.labels,keep_prob:1}))
      #Record the training time
      print("----%s seconds ----" % (time.time() - start_time))

if __name__ == '__main__':
  parser = argparse.ArgumentParser()
  parser.add_argument('--data_dir', type=str, default='/tmp/tensorflow/mnist/input_data',
                      help='Directory for storing input data')
  FLAGS, unparsed = parser.parse_known_args()
  tf.app.run(main=main, argv=[sys.argv[0]] + unparsed)


