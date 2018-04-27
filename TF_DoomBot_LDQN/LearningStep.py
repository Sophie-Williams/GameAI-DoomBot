#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
@author: jk
Paul

"""
#my other files
from DoomMain import *
from ReplayMemory import *
from settings import *
from LinearDeepQNetwork import *


#preprocess an image
def preprocess(img):
    img = skimage.transform.resize(img, resolution)
    img = img.astype(np.float32)
    return img

# performs learning for a single action
def learn_from_memory(memory,learn, get_q_values, get_best_action):
    

    # Get a random minibatch from the replay memory and learns from it.
    if memory.size > batch_size:
        s1, a, s2, isterminal, r = memory.get_sample(batch_size)

        q2 = np.max(get_q_values(s2), axis=1)
        target_q = get_q_values(s1)
        # target differs from q only for the selected action. The following means:
        # target_Q(s,a) = r + gamma * max Q(s2,_) if isterminal else r
        target_q[np.arange(target_q.shape[0]), a] = r + discount_factor * (1 - isterminal) * q2
        #call learn function create_linear_dqn_network
        learn(s1, target_q)


def learning_step(epoch,game,actions,memory,learn, get_q_values, get_best_action):

    #linear decay
    # returns a variable associated with the chance to choose random action or 
    # use the network for the next action
    def exploration_rate(epoch):

        start_eps = 1.0
        end_eps = 0.1
        const_eps_epochs = 0.2 * epochs  
        eps_decay_epochs = 0.6 * epochs  

        if epoch < const_eps_epochs:
            return start_eps
        elif epoch < eps_decay_epochs:
            # Linear decay
            return start_eps - (epoch - const_eps_epochs) / \
                               (eps_decay_epochs - const_eps_epochs) * (start_eps - end_eps)
        else:
            return end_eps

    s1 = preprocess(game.get_state().screen_buffer)
    healths1 = game.get_game_variable(GameVariable.HEALTH)

    # If random is lessthan or equal to the exploraton rate
    # choose a random action
    #otherwise choose best action based on network
    eps = exploration_rate(epoch)
    if random() <= eps:
        a = randint(0, len(actions) - 1)
    else:
        # Choose the best action according to the network.
        a = get_best_action(s1)
    reward = game.make_action(actions[a], frame_repeat)
    
    
    isterminal = game.is_episode_finished()
    s2 = preprocess(game.get_state().screen_buffer) if not isterminal else None
    healths2 = game.get_game_variable(GameVariable.HEALTH) if not isterminal else None

    #if player health decreases this step lower the reward
    if(healths1 != healths2):
        reward = reward - 20;
    

    # Remember the transition that was just experienced.
    memory.add_transition(s1, a, s2, isterminal, reward,healths1,healths2)
    #perform a learning step 
    learn_from_memory(memory,learn, get_q_values, get_best_action)