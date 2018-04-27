#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
@author: jk
Settings file to allow changing the senario and neural network parameters
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


#successful run at lr = .000288 ,df = .996, epochs 150, lspe = 3000, rms = 3000, minibatch = 32


# Q-learning settings
learning_rate = 0.000288
# learning_rate = 0.0001
discount_factor = 0.996
epochs = 155
learning_steps_per_epoch = 3000
replay_memory_size = 3000

# NN learning settings
batch_size = 32

# Training regime
test_episodes_per_epoch = 100

# Other parameters
frame_repeat = 12
resolution = (30, 45)
episodes_to_watch = 35


config_file_path = "../../scenarios/DEADCKR.cfg"

scoreList = np.zeros((3,epochs), dtype=np.float32)