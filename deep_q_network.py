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

from replay_memory import ReplayMemory
from neural_net import NeuralNet
from random_action_generator import RandomActionGenerator
from keyboard_action import KeyboardAction

def run_action(game_state, action):
    a_t = np.zeros([ACTIONS])
    a_t[action] = 1

    x_t, r_0, terminal = game_state.frame_step(a_t)
    x_t = cv2.cvtColor(cv2.resize(x_t, (80, 80)), cv2.COLOR_BGR2GRAY)
    ret, x_t = cv2.threshold(x_t,1,255,cv2.THRESH_BINARY)

    return r_0, x_t, terminal

def trainNetwork():
    game_state = game.GameState()
    replay_memory = ReplayMemory(50000, 32)
    neural_net = NeuralNet()

    r_0, x_t, terminal = run_action(game_state, 0)
    s_t = np.stack((x_t, x_t), axis=2)

    random_action_generator = RandomActionGenerator()
    keyboard_action = KeyboardAction()
    t = 0

    while "flappy bird" != "angry bird":
        if t > -1:
            action_index = np.argmax(neural_net.predict(s_t))
            action_index = random_action_generator.adapt_action(action_index)
        else:
            action_index = keyboard_action.action()

        r_t, x_t1, terminal = run_action(game_state, action_index)

        x_t1 = np.reshape(x_t1, (80, 80, 1))
        s_t1 = np.append(x_t1, s_t[:, :, :1], axis=2)

        replay_memory.append({
            'state': s_t,
            'action': action_index,
            'reward': r_t,
            'next_state': s_t1,
            'terminal': terminal
        })

        if t % 100 == 0:
            neural_net.save()
            replay_memory.save()
        if replay_memory.can_train():
            for i in range(0, 3):
                neural_net.train_on_replay_memory(replay_memory)

        s_t = s_t1
        t += 1

        print("TIMESTEP", t, "/ ACTION", action_index, "/ REWARD", r_t)

if __name__ == "__main__":
    trainNetwork()
