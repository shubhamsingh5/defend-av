from abc import ABC, abstractmethod
import torch

class BaseAgent(ABC):
    """Abstract base class for all agents"""
    
    def __init__(self, state_dim, action_dim, config):
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.config = config
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.training_steps = 0
    
    @abstractmethod
    def select_action(self, state, training=True):
        """Select an action given current state"""
        pass
    
    @abstractmethod
    def train_step(self):
        """Perform one step of training"""
        pass
    
    @abstractmethod
    def store_transition(self, state, action, reward, next_state, done):
        """Store transition in memory"""
        pass

    def save(self, path):
        """Save agent's state"""
        pass
    
    def load(self, path):
        """Load agent's state"""
        pass