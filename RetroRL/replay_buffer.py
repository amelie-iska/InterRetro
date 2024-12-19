
import numpy as np
import random

class ReplayBuffer:
    def __init__(self, capacity):
        self.capacity = capacity
        self.buffer = {
            'obs': [[] for _ in range(int(capacity))],
            'next_obs': [[] for _ in range(int(capacity))],
            'rewards': [[] for _ in range(int(capacity))],
            'dones': [[] for _ in range(int(capacity))],
        }
        self.size = 0
        self.pos = 0

    def add(self, obs, next_obs, reward, done):
        idx = int(self.pos % self.capacity)
        self.buffer['obs'][idx] = obs
        self.buffer['next_obs'][idx] = next_obs
        self.buffer['rewards'][idx] = reward
        self.buffer['dones'][idx] = done

        self.pos += 1
        self.size = int(min(self.size + 1, self.capacity))
        return {'buffer/space': self.size/self.capacity,
                'buffer/size': self.size}

    def sample(self, batch_size):
        indices = random.sample(range(self.size), batch_size)
        return {
            'obs': [self.buffer['obs'][i] for i in indices],
            'next_obs': [self.buffer['next_obs'][i] for i in indices],
            'rewards': [self.buffer['rewards'][i] for i in indices],
            'dones': [self.buffer['dones'][i] for i in indices]
        }

    def __len__(self):
        return self.size