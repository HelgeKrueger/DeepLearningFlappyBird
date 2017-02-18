#!/usr/bin/env python
from __future__ import print_function

import cv2
import sys
sys.path.append("game/")
import wrapped_flappy_bird as game
import random
import numpy as np

GAME = 'bird' # the name of the game being played for log files
ACTIONS = 2 # number of valid actions
GAMMA = 0.99 # decay rate of past observations
OBSERVE = 500. # timesteps to observe before training
EXPLORE = 2000000. # frames over which to anneal epsilon
FINAL_EPSILON = 0.0001 # final value of epsilon
INITIAL_EPSILON = 0.1 # starting value of epsilon

from replay_memory import ReplayMemory
from neural_net import NeuralNet

def log_info(t, epsilon, action_index, r_t):
    state = ""
    if t <= OBSERVE:
        state = "observe"
    elif t > OBSERVE and t <= OBSERVE + EXPLORE:
        state = "explore"
    else:
        state = "train"

    print("TIMESTEP", t, "/ STATE", state, "/ EPSILON", epsilon, "/ ACTION", action_index, "/ REWARD", r_t)

def run_action(game_state, action):
    x_t, r_0, terminal = game_state.frame_step(action)
    x_t = cv2.cvtColor(cv2.resize(x_t, (80, 80)), cv2.COLOR_BGR2GRAY)
    ret, x_t = cv2.threshold(x_t,1,255,cv2.THRESH_BINARY)

    return r_0, x_t, terminal

def trainNetwork():
    game_state = game.GameState()
    replay_memory = ReplayMemory(50000, 32)
    neural_net = NeuralNet()

    # get the first state by doing nothing and preprocess the image to 80x80x4
    do_nothing = np.zeros(ACTIONS)
    do_nothing[0] = 1

    r_0, x_t, terminal = run_action(game_state, do_nothing)
    s_t = np.stack((x_t, x_t), axis=2)

    # start training
    epsilon = INITIAL_EPSILON
    t = 0

    while "flappy bird" != "angry bird":
        a_t = np.zeros([ACTIONS])
        action_index = 0

        if t % 1 == 0:
            action = neural_net.predict(s_t)
            if random.random() <= epsilon:
                print("----------Random Action----------")
                action_index = random.randrange(ACTIONS)
            else:
                action_index = np.argmax(action)

        a_t[action_index] = 1

        # scale down epsilon
        if epsilon > FINAL_EPSILON and t > OBSERVE:
            epsilon -= (INITIAL_EPSILON - FINAL_EPSILON) / EXPLORE

        # run the selected action and observe next state and reward
        r_t, x_t1, terminal = run_action(game_state, a_t)

        x_t1 = np.reshape(x_t1, (80, 80, 1))
        s_t1 = np.append(x_t1, s_t[:, :, :1], axis=2)

        replay_memory.append({
            'state': s_t,
            'action': action_index,
            'reward': r_t,
            'next_state': s_t1,
            'terminal': terminal
        })

        # only train if done observing
        if t > OBSERVE:
            minibatch = replay_memory.get_training_sample()

            inputs = np.array([ele['state'] for ele in minibatch])
            inputs_next = np.array([ele['next_state'] for ele in minibatch])

            targets = neural_net.predict(inputs)
            predictions_next = neural_net.predict(inputs_next)

	    for i in range(0, len(minibatch)):
                sample = minibatch[i]
                targets[i, sample['action']] = sample['reward']
		if not sample['terminal']:
		    targets[i, sample['action']] += GAMMA * np.max(predictions_next[i])

            neural_net.train(inputs, targets)

        if t % 100:
            neural_net.save()

        # update the old values
        s_t = s_t1
        t += 1

        log_info(t, epsilon, action_index, r_t)

if __name__ == "__main__":
    trainNetwork()
