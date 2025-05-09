import numpy as np
import random
from collections import deque

class ReplayBuffer:
    """
    Experience replay buffer for storing and sampling experiences
    """
    def __init__(self, buffer_size=1000000):
        """
        Initialize replay buffer with given size
        """
        self.buffer = deque(maxlen=buffer_size)
        self.buffer_size = buffer_size
    
    def add(self, state, action, reward, next_state, done):
        """
        Add experience to buffer
        """
        experience = (state, action, reward, next_state, done)
        self.buffer.append(experience)
    
    def sample(self, batch_size):
        """
        Sample a batch of experiences from buffer
        """
        if len(self.buffer) < batch_size:
            batch = random.sample(self.buffer, len(self.buffer))
        else:
            batch = random.sample(self.buffer, batch_size)
        
        # Separate batch into components
        states = np.array([experience[0] for experience in batch])
        actions = np.array([experience[1] for experience in batch])
        rewards = np.array([experience[2] for experience in batch])
        next_states = np.array([experience[3] for experience in batch])
        dones = np.array([experience[4] for experience in batch])
        
        return (states, actions, rewards, next_states, dones)
    
    def size(self):
        """
        Return current size of buffer
        """
        return len(self.buffer)
    
    def clear(self):
        """
        Clear buffer
        """
        self.buffer.clear()