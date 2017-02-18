#!/usr/bin/env python
from __future__ import print_function

import numpy as np
import sys

from replay_memory import ReplayMemory
from neural_net import NeuralNet
from random_action_generator import RandomActionGenerator
from keyboard_action import KeyboardAction
from game_wrapper import Game

def trainNetwork(godmode):
    game = Game()
    replay_memory = ReplayMemory(5000, 32)
    neural_net = NeuralNet()

    r_0, x_t, terminal = game.run_action(0)
    s_t = np.stack((x_t, x_t), axis=2)

    random_action_generator = RandomActionGenerator()
    keyboard_action = KeyboardAction()

    for t in range(1, 1000): 
        if godmode:
            action_index = keyboard_action.action()
        else:
            action_index = np.argmax(neural_net.predict(s_t))
            action_index = random_action_generator.adapt_action(action_index)

        r_t, x_t1, terminal = game.run_action(action_index)

        print("TIMESTEP", t, "/ ACTION", action_index, "/ REWARD", r_t, neural_net.state())

        x_t1 = np.reshape(x_t1, (80, 80, 1))
        s_t1 = np.append(x_t1, s_t[:, :, :1], axis=2)

        replay_memory.append({
            'state': s_t,
            'action': action_index,
            'reward': r_t,
            'next_state': s_t1,
            'terminal': terminal
        })
        s_t = s_t1

    replay_memory.save()

if __name__ == "__main__":
    if len(sys.argv) > 1:
        trainNetwork(True)
    else:
        trainNetwork(False)
