from collections import deque
import random

class ReplayMemory():
    def __init__(self, max_size, mini_batch_size):
        self.deque = deque()
        self.max_size = max_size
        self.mini_batch_size = mini_batch_size

    def append(self, item):
        self.deque.append(item)
        if len(self.deque) > self.max_size:
            self.deque.popleft()

    def get_training_sample(self):
        return random.sample(self.deque, self.mini_batch_size)
