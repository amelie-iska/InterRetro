
import numpy as np
import random

class ReplayBuffer:
    def __init__(self, capacity, obs_dim, action_dim):
        """
        Initialize the replay buffer.

        :param capacity: Maximum number of experiences to store.
        :param obs_dim: Dimension of the observation space (e.g., 2048 for 2048-dimensional observations).
        :param action_dim: Dimension of the action space (discrete or continuous).
        """
        self.capacity = capacity
        self.obs_dim = obs_dim
        self.action_dim = action_dim
        self.buffer = {
            'obs': np.zeros((int(capacity), int(obs_dim)), dtype=np.float32),
            'next_obs': [[] for _ in range(int(capacity))],
            'actions': np.zeros((int(capacity), int(action_dim)), dtype=np.float32),
            'rewards': [[] for _ in range(int(capacity))],
            'dones': np.zeros(int(capacity), dtype=np.float32),
        }
        self.size = 0
        self.pos = 0

    def add(self, obs, action, next_obs, reward):
        """
        Add an experience to the buffer.

        :param obs: Current observation (shape: [obs_dim]).
        :param next_obs: Next observation (shape: [obs_dim]).
        :param action: Action taken (shape: [action_dim]).
        :param reward: Reward received (float).
        :param done: Whether the episode has ended (bool).
        """
        idx = int(self.pos % self.capacity)
        self.buffer['obs'][idx] = obs
        self.buffer['next_obs'][idx] = next_obs
        self.buffer['actions'][idx] = action
        self.buffer['rewards'][idx] = reward
        self.pos += 1
        self.size = int(min(self.size + 1, self.capacity))
        return {'buffer/space': self.size/self.capacity}

    def sample(self, batch_size):
        """
        Sample a batch of experiences from the buffer.

        :param batch_size: Number of experiences to sample.
        :return: A dictionary of sampled experiences.
        """
        indices = random.sample(range(self.size), batch_size)
        return {
            'obs': self.buffer['obs'][indices],
            'next_obs': [self.buffer['next_obs'][i] for i in indices],
            'actions': self.buffer['actions'][indices],
            'rewards': [self.buffer['rewards'][i] for i in indices],
        }

    def __len__(self):
        return self.size