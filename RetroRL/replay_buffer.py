
import numpy as np
import random

class ReplayBuffer:
    def __init__(self, capacity, agent):
        self.agent = agent
        self.capacity = capacity
        self.buffer = {
            'obs': [[] for _ in range(int(capacity))],
            'actions': [[] for _ in range(int(capacity))],
            'next_obs': [[] for _ in range(int(capacity))],
            'rewards': [[] for _ in range(int(capacity))],
            'dones': [[] for _ in range(int(capacity))],
            'graphs': [[] for _ in range(int(capacity))],
        }
        self.size = 0
        self.pos = 0

    def add(self, obs, actions, next_obs, rewards, dones):
        idx = int(self.pos % self.capacity)
        self.buffer['obs'][idx] = obs
        self.buffer['actions'][idx] = actions
        self.buffer['next_obs'][idx] = next_obs
        self.buffer['rewards'][idx] = rewards
        self.buffer['dones'][idx] = dones
        
        reactant_string = '.'.join(next_obs)
        graph_seq = self.agent.build_graph_seq(obs, reactant_string, actions)
        if len(graph_seq) > 0:
            self.buffer['graphs'][idx] = graph_seq
        
        self.pos += 1
        self.size = int(min(self.size + 1, self.capacity))
        overlap = len(set(self.buffer['obs'][:self.pos])) / self.pos 
        return {'buffer/space': self.size/self.capacity,
                'buffer/size': self.size,
                'buffer/overlap': overlap}

    def sample(self, batch_size):
        indices = random.sample(range(self.size), batch_size)
        return {
            'obs': [self.buffer['obs'][i] for i in indices],
            'actions': [self.buffer['actions'][i] for i in indices],
            'next_obs': [self.buffer['next_obs'][i] for i in indices],
            'rewards': [self.buffer['rewards'][i] for i in indices],
            'dones': [self.buffer['dones'][i] for i in indices],
            'graphs': [self.buffer['graphs'][i] for i in indices],
        }

    def __len__(self):
        return self.size