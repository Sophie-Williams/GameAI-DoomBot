#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
@author: jk
Paul
Creates the model and returns functions that can perform learning/prediction
"""
from __future__ import division
from __future__ import print_function
from vizdoom import *
import itertools as it
from random import sample, randint, random, seed
from time import time, sleep
import numpy as np
import skimage.color, skimage.transform
import tensorflow as tf
import matplotlib.pyplot as plt
from tqdm import trange

#my other files
from DoomMain import *
from ReplayMemory import *
from settings import *
from LearningStep import *

def create_linear_dqn_network(session, available_actions_count):

    
    # Create the input variables
    image_state = tf.placeholder(tf.float32, [None] + list(resolution) + [1], name="State")
    
    
    a_ = tf.placeholder(tf.int32, [None], name="Action")
    target_q_ = tf.placeholder(tf.float32, [None, available_actions_count], name="TargetQ")

    #declare network with 2 convolution layers and 1 fully connected layer
    conv1 = tf.contrib.layers.convolution2d(image_state, num_outputs=8, kernel_size=[6, 6], stride=[3, 3],
                                            activation_fn=tf.nn.relu,
                                            weights_initializer=tf.contrib.layers.xavier_initializer_conv2d(),
                                            biases_initializer=tf.constant_initializer(0.1))
    conv2 = tf.contrib.layers.convolution2d(conv1, num_outputs=8, kernel_size=[3, 3], stride=[2, 2],
                                            activation_fn=tf.nn.relu,
                                            weights_initializer=tf.contrib.layers.xavier_initializer_conv2d(),
                                            biases_initializer=tf.constant_initializer(0.1))
    conv2_flat = tf.contrib.layers.flatten(conv2)
    #best so far 300 neurons
    fc1 = tf.contrib.layers.fully_connected(conv2_flat, num_outputs=300, activation_fn=tf.nn.relu,
                                            weights_initializer=tf.contrib.layers.xavier_initializer(),
                                            biases_initializer=tf.constant_initializer(0.1))

    q = tf.contrib.layers.fully_connected(fc1, num_outputs=available_actions_count, activation_fn=None,
                                          weights_initializer=tf.contrib.layers.xavier_initializer(),
                                          biases_initializer=tf.constant_initializer(0.1))
    #choses action with the estimated highest future reward
    best_a = tf.argmax(q, 1)

    loss = tf.losses.mean_squared_error(q, target_q_)

    optimizer = tf.train.RMSPropOptimizer(learning_rate)
    # Update the parameters according to the computed gradient using RMSProp.
    train_step = optimizer.minimize(loss)
    #function to perform learning step
    def function_learn(s1, target_q):
        feed_dict = {image_state: s1, target_q_: target_q}
        l, _ = session.run([loss, train_step], feed_dict=feed_dict)
        return l

    def function_get_q_values(state):
        return session.run(q, feed_dict={image_state: state})

    def function_get_best_action(state):
        return session.run(best_a, feed_dict={image_state: state})

    def function_simple_get_best_action(state):
        return function_get_best_action(state.reshape([1, resolution[0], resolution[1], 1]))[0]

    return function_learn, function_get_q_values, function_simple_get_best_action
