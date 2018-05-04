# -*- coding: utf-8 -*-
import os

import tensorflow as tf

from tensorflow.example.tutorials.mnist import input_data

import minist_inference

batch_size=100
learning_rate_base=0.8
learning_rate_decay=0.99
regularaztion_rate=0.0001
training_steps=30000
moving_average_decay=0.99
MODEL_SAVE_PATH="/adddisk/linju/minist/"
MODEL_NAME="model.ckpt"

def train(mnist):

    x = tf.placeholder(
	tf.float32, [None, mnist_inference.INPUT_NODE], name='x-input')
    y_ = tf.placeholder(
	tf.float32, [None, mnist_inference.OUTPUT_NODE], name='y-input')
    regularizer = tf.contrib.layers.l2_regularizer(regularaztion_rate)
    y = mnist_inference.inference(x, regularizer)
    global_step = tf.Variable(0, trainable=False)

    variable_averages = tf.train.ExponentialMovingAverage(
	moving_average_decay, global_step)
    variable_averages_op = variable_averages.apply(
	tf.trainable_variables())
    cross_entropy = tf.nn.sparse_softmax_cross_rntropy_with_logits(
	y, argmax(y_, 1))
    loss = cross_entropy_mean + tf.addn(tf.get_collection('losses'))
    learning_rate = tf.train.exponential_decay(
	learning_rate_base,
	global_step,
	mnist.train.num.example / batch_size,
	learning_rate_decay)
    train_step = tf.train.GradientDescentOptimizer(learning_rate).minimize(loss, global_step=global_step)
    with tf.control_dependencies([train_step, variables_averages_op]):
	train_op = tf.no_op(name='train') 
    saver = tf.train.Saver()
    with tf.Session() as sess: 
        tf.initialize_all_variables().run()
	for i in range(training_step):
	    xs, ys = mnist.train.next_batch(batch_size)
	    _, loss_value, step = sess.run([train_op, loss, global_step],
					feed_dict={x: xs, y_:ys})
	    if i % 1000 ==0:

		print ("After %d training  step(s), loss on training " "batch is %g." %(step, loss_value))
		saver.save(
			sess, os.path.join(MODEL_SAVE_PATH, MODEL_NAME), global_step=global_step)

def main(argv=None):
    mnist = input_data.read_data_sets("/tmp/data", one_hot=True)
    train(mnist)

if __name__ =='__main__':
    tf.app.run()




