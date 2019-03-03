import os
import numpy as np


class ReplayBuffer:
    def __init__(self, max_length=10000):
        self.buffer = []
        self.max_length = max_length

    def get_fill(self):
        fill = 100 * len(self.buffer) / self.max_length
        return fill

    def add(self, transition):
        # transiton is tuple of (state, action, reward, next_state, done)
        if len(self.buffer) > self.max_length:
            self.buffer.pop(0)
        self.buffer.append(transition)

    def count(self):
        return len(self.buffer)

    def sample(self, batch_size):
        indexes = np.random.randint(0, len(self.buffer), size=batch_size)
        state, action, reward, next_state, done = [], [], [], [], []

        for i in indexes:
            s, a, r, s_, d = self.buffer[i]
            state.append(np.array(s, copy=False))
            action.append(np.array(a, copy=False))
            reward.append(np.array(r, copy=False))
            next_state.append(np.array(s_, copy=False))
            done.append(np.array(d, copy=False))

        return np.array(state), np.array(action), np.array(reward), np.array(next_state), np.array(done)


def mkdir(base, name):
    path = os.path.join(base, name)
    if not os.path.exists(path):
        os.makedirs(path)
    return path
