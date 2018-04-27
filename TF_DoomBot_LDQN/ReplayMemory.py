#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Apr 22 18:56:04 2018

@author: jk
Paul Murray
This file holds to replymemory class to enable minibatch optimazation
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
from LinearDeepQNetwork import *
from settings import *
from LearningStep import *
#replay memory class to allow for the use of mini batches
class ReplayMemory:
    #initializes memory to replaymemory size
    def __init__(self, capacity):
        channels = 1
        state_shape = (capacity, resolution[0], resolution[1], channels)
        self.s1 = np.zeros(state_shape, dtype=np.float32)
        self.s2 = np.zeros(state_shape, dtype=np.float32)
        self.a = np.zeros(capacity, dtype=np.int32)
        self.r = np.zeros(capacity, dtype=np.float32)
        self.isterminal = np.zeros(capacity, dtype=np.float32)
        self.healths1 = np.zeros(capacity, dtype=np.float32)
        self.healths2 = np.zeros(capacity, dtype=np.float32)

        self.capacity = capacity
        self.size = 0
        self.pos = 0
    #add a single transition to the replay memory consisting of a begining state
    # the action taken, the end state after the action, is this is a terminal action
    #the reward from the action and the players health
    def add_transition(self, s1, action, s2, isterminal, reward,healths1, healths2):
        self.s1[self.pos, :, :, 0] = s1
        self.a[self.pos] = action
        if not isterminal:
            self.s2[self.pos, :, :, 0] = s2
        self.isterminal[self.pos] = isterminal
        self.r[self.pos] = reward
        self.healths1[self.pos] = healths1;
        self.healths2[self.pos] = healths2;

        self.pos = (self.pos + 1) % self.capacity
        self.size = min(self.size + 1, self.capacity)
    #return a memory sample to use in mini batch 
    def get_sample(self, sample_size):
        i = sample(range(0, self.size), sample_size)
        return self.s1[i], self.a[i], self.s2[i], self.isterminal[i], self.r[i]
    
    def get_last_health(self):
        #if(healths1 != healths2)
        #not currently in use
        return 1;