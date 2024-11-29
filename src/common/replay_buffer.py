import numpy as np
from collections import deque
from typing import Tuple, Dict, Any

class ReplayBuffer:
    """
    Experience replay buffer for DQN training.
    Stores transitions (state, action, reward, next_state, done) and provides random sampling.
    """
    def __init__(self, capacity: int):
        """
        Initialize replay buffer.
        """
        self.buffer = deque(maxlen=capacity)
    
    def push(self, state: np.ndarray, action: np.ndarray, 
             reward: float, next_state: np.ndarray, done: bool):
        """
        Store a transition in the buffer.
        """
        self.buffer.append((state, action, reward, next_state, done))
    
    def sample(self, batch_size: int):
        """
        Sample a batch of experiences from the replay buffer.
        """
        indices = np.random.choice(len(self.buffer), batch_size, replace=False)
        states, actions, rewards, next_states, dones = zip(*[self.buffer[idx] for idx in indices])
        return (np.array(states),
                np.array(actions),
                np.array(rewards, dtype=np.float32),
                np.array(next_states),
                np.array(dones, dtype=np.float32))
    
    def __len__(self) -> int:
        """Return current size of the buffer"""
        return len(self.buffer)