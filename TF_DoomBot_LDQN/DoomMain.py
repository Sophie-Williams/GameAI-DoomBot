#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
@author: jk
name Paul Murray
TensorFlow Doom Run File
Structure 2 convolution layers, 1 hidden fully connected layer
Uses mini batch gradient descent
based on tensorflow tutorial from http://vizdoom.cs.put.edu.pl/
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
from ReplayMemory import *
from LinearDeepQNetwork import *
from settings import *
from LearningStep import *


#code to handle model save/loading
#Currently set to load a model
model_savefile = "TF_DoomBot/DoomModel.ckpt"
save_model = False
load_model = True
skip_learning = True

#plot performance
def plotVsScore(scoreList):
    
    
    plt.figure(1)
    print("Min");
    plt.plot(scoreList[0])
    
    plt.figure(2)
    print("Mean");
    plt.plot(scoreList[1])
    
    plt.figure(3)
    print("Max");
    plt.plot(scoreList[2])
    plt.show()

# Converts and down-samples the input image
def preprocess(img):
    img = skimage.transform.resize(img, resolution)
    img = img.astype(np.float32)
    return img

# Creates and initializes ViZDoom environment.
def initialize_vizdoom(config_file_path):
    print("Initializing doom...")
    game = DoomGame()
    game.load_config(config_file_path)
    game.set_window_visible(False)
    game.set_mode(Mode.PLAYER)
    game.set_screen_format(ScreenFormat.GRAY8)
    game.set_screen_resolution(ScreenResolution.RES_640X480)
    game.init()
    print("Doom initialized.")
    return game


if __name__ == '__main__':
        #reset graph to get rid of residual network data
    tf.reset_default_graph();

    # initialize visdoom game 
    game = initialize_vizdoom(config_file_path)

    # Action = which buttons are pressed
    n = game.get_available_buttons_size()
    print(n)
    actions = [list(a) for a in it.product([0, 1], repeat=n)]

    # Create replay memory which will store the transitions
    memory = ReplayMemory(capacity=replay_memory_size)
    
    #set living reward
    game.set_living_reward(-3.0)

    #declare tensorflow session
    session = tf.Session()
    
    #create_linear_dqn_network takes in tensorflow session and a list of possible actions and returns functions
    #to 

    learn, get_q_values, get_best_action = create_linear_dqn_network(session, len(actions))
    saver = tf.train.Saver()
    
    #if load model is set to true load model weights from the specified model folder
    #otherwise initialize new weights
    if load_model:
        print("Loading model from: ", model_savefile)
        saver.restore(session, model_savefile)
    else:
        init = tf.global_variables_initializer()
        session.run(init)

    print("Starting")

    #start timer to track run time
    time_start = time()
    #block of code to perform learning steps if skip skip_learning is set to false
    if not skip_learning:
        for epoch in range(epochs):
            print("\nEpoch number %d\n" % (epoch + 1))
            train_episodes_finished = 0
            train_scores = []
            
            print("Begining Training")
            game.new_episode()
            for step in trange(learning_steps_per_epoch, leave=False):
                learning_step(epoch,game,actions,memory,learn, get_q_values, get_best_action)
                if game.is_episode_finished():
                    score = game.get_total_reward()
                    train_scores.append(score)
                    game.new_episode()
                    train_episodes_finished += 1
    
            print("Episode %d Played" % train_episodes_finished)
    
            train_scores = np.array(train_scores)
        
            print("Results: mean: %.1f±%.1f," % (train_scores.mean(), train_scores.std()), \
                  "min: %.1f," % train_scores.min(), "max: %.1f," % train_scores.max())
        
            print("\nTesting...")
            test_episode = []
            test_scores = []
            for test_episode in trange(test_episodes_per_epoch, leave=False):
                game.new_episode()
                while not game.is_episode_finished():
                    state = preprocess(game.get_state().screen_buffer)
                    best_action_index = get_best_action(state)
                    game.make_action(actions[best_action_index], frame_repeat)
                
                
                reward = game.get_total_reward()
                test_scores.append(reward)
                
            print("Saving the network weigths to:", model_savefile)
            saver.save(session, model_savefile)
        
            test_scores = np.array(test_scores)
            print("Results: mean: %.1f±%.1f," % (test_scores.mean(), test_scores.std()), "min: %.1f" % test_scores.min(),"max: %.1f" % test_scores.max())
            print("Total elapsed time: %.2f minutes" % ((time() - time_start) / 60.0))
            scoreList[0][epoch] = test_scores.min()
            scoreList[1][epoch] = test_scores.mean()
            scoreList[2][epoch] = test_scores.max()
                
    game.close()
    print("======================================")
    print("Training finished")
    print("======================================")
    print("Printing minimum, mean and maximum scores graph");
    plotVsScore(scoreList);
    
    #wait until the user presses enter to run replays
    temp = input("Press enter in the console to Watch replay");
    # Reinitialize the game with window visible
    game.set_window_visible(True)
    game.set_mode(Mode.ASYNC_PLAYER)
    game.init()

    for _ in range(episodes_to_watch):
        game.new_episode()
        while not game.is_episode_finished():
            #get the current image from the game
            state = preprocess(game.get_state().screen_buffer)
            #use network to choose the best action
            best_action_index = get_best_action(state)

            # Instead of make_action(a, frame_repeat) in order to make the animation smooth
            game.set_action(actions[best_action_index])
            for _ in range(frame_repeat):
                game.advance_action()

        # Sleep between episodes
        sleep(1.0)
        #retrieve final internal reward
        score = game.get_total_reward()
        print("Total score is: ", score)
    game.close()   
    
    
    #wait until the user presses enter to run second round of replays- useful for recording
    temp = input("Press enter in the console to Watch replay");
    # Reinitialize the game with window visible
    game.set_window_visible(True)
    game.set_mode(Mode.ASYNC_PLAYER)
    game.init()
    
    for _ in range(episodes_to_watch):
        game.new_episode()
        while not game.is_episode_finished():
            #get the current image from the game
            state = preprocess(game.get_state().screen_buffer)
            #use network to choose the best action
            best_action_index = get_best_action(state)

            # Instead of make_action(a, frame_repeat) in order to make the animation smooth
            game.set_action(actions[best_action_index])
            for _ in range(frame_repeat):
                game.advance_action()

        # Sleep between episodes
        sleep(1.0)
        score = game.get_total_reward()
        print("Total score is: ", score)

