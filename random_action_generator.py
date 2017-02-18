import random

FINAL_EPSILON = 0.0001 # final value of epsilon

class RandomActionGenerator():
    def __init__(self):
        self.epsilon = 0.01

    def adapt_action(self, action_index):
        if random.random() < self.epsilon:
            print('---- opposite action ----')
            if action_index == 1:
                action_index = 0
            else:
                action_index = 1;

        return action_index

