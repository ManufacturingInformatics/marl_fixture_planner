import torch 
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from collections import namedtuple, deque
import random

Transition = namedtuple('Transition', 
                        ('state', 'action', 'reward'))

class ReplayMemory(object):
    
    def __init__(self, capacity):
        self.memory = deque([], maxlen=capacity)
        
    def push(self, *args):
        """
        Saves a transition for the agent
        """
        self.memory.append(Transition(*args))
        
    def sample(self, batch_size):
        """
        Returns a set of transitions based on the batch size value

        Args:
            batch_size (int): number of batches to return

        Returns:
            Transition: named tuple of transitions
        """
        return random.sample(self.memory, batch_size)
    
    def __len__(self):
        return len(self.memory)
    

class DQN(nn.Module):
    
    def __init__(self, n_obs, n_actions, hidden_size=256):
        super(DQN, self).__init__()
        self.layer1 = nn.Linear(n_obs[0], hidden_size)
        self.layer2 = nn.Linear(hidden_size, hidden_size)
        self.layer3 = nn.Linear(hidden_size, hidden_size)
        self.layer4 = nn.Linear(hidden_size, int(n_actions))
        
    def forward(self, x):
        
        x = F.relu(self.layer1(x))
        x = F.relu(self.layer2(x))
        x = F.relu(self.layer3(x))
        return self.layer4(x)
