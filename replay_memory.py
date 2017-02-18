from collections import deque
import random
import pickle
import os.path

class ReplayMemory():
    def __init__(self, max_size, mini_batch_size):
        self.deque = deque()
        self.max_size = max_size
        self.mini_batch_size = mini_batch_size

        self.filename = 'flappy_replay_memory.pickle'

        if os.path.isfile(self.filename):
            self.deque = pickle.load(open(self.filename, 'r'))
        else:
            self.deque = deque()


    def can_train(self):
        return len(self.deque) > self.mini_batch_size


    def append(self, item):
        self.deque.append(item)
        if len(self.deque) > self.max_size:
            self.deque.popleft()

    def get_training_sample(self):
        return random.sample(self.deque, self.mini_batch_size)

    def save(self):
        pickle.dump(self.deque, open(self.filename, 'w'))
