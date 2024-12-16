from collections import deque
import random


class ReplayMemory:
    def __init__(self, capacity):
        self.capacity = capacity
        self.buffer = deque(maxlen=self.capacity)

    def push(self, state, action, reward, next_state, done):
        """save a transition"""
        self.buffer.append((state, action, reward, next_state, done))

    def sample(self, batch_size, sequential):
        if batch_size > len(self.buffer):
            batch_size = len(self.buffer)
        if sequential:
            rand = random.randint(0, len(self.buffer) - batch_size)
            batch = [self.buffer[i] for i in range(rand, rand + batch_size)]
            return zip(*batch)
        else:
            batch = random.sample(self.buffer, batch_size)
            return zip(*batch)

    def clear(self):
        self.buffer.clear()

    def __len__(self):
        return len(self.buffer)
