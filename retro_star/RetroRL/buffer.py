
import numpy as np
import random

class ReplayBuffer:
    def __init__(self, capacity, obs_dim, action_dim):
        self.capacity = capacity
        self.obs_dim = obs_dim
        self.action_dim = action_dim
        self.buffer = {
            'obs': np.zeros((int(capacity), int(obs_dim)), dtype=np.float32),
            'next_obs': [[] for _ in range(int(capacity))],
            'actions': np.zeros((int(capacity), 1), dtype=np.float32),
            'rewards': [[] for _ in range(int(capacity))],
            'dones': [[] for _ in range(int(capacity))],
        }
        self.size = 0
        self.pos = 0

    def add(self, obs, action, next_obs, reward, done):
        idx = int(self.pos % self.capacity)
        self.buffer['obs'][idx] = obs
        self.buffer['next_obs'][idx] = next_obs
        self.buffer['rewards'][idx] = reward
        self.buffer['dones'][idx] = done
        self.buffer['actions'][idx] = np.argmax(action)

        self.pos += 1
        self.size = int(min(self.size + 1, self.capacity))
        return {'buffer/space': self.size/self.capacity,
                'buffer/size': self.size}

    def sample(self, batch_size):
        indices = random.sample(range(self.size), batch_size)

        action_indices = self.buffer['actions'][indices]
        actions = np.zeros((batch_size, self.action_dim), dtype=int)
        actions[np.arange(batch_size), action_indices.astype(int)] = 1
        return {
            'obs': self.buffer['obs'][indices],
            'next_obs': [self.buffer['next_obs'][i] for i in indices],
            'actions': actions,
            'rewards': [self.buffer['rewards'][i] for i in indices],
            'dones': [self.buffer['dones'][i] for i in indices]
        }

    def __len__(self):
        return self.size