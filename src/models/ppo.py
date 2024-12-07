import torch
import torch.nn as nn
import torch.nn.functional as F

class Actor(nn.Module):
    def __init__(self, state_dim, action_dim, hidden_size=256):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(state_dim, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, hidden_size),
            nn.ReLU()
        )
        self.mean = nn.Linear(hidden_size, action_dim)
        self.log_std = nn.Parameter(torch.zeros(action_dim))
        
    def forward(self, x):
        x = self.net(x)
        mean = self.mean(x)
        # Apply different activation for motor and steering
        mean[:, 0] = 0.5 * (torch.tanh(mean[:, 0]) + 1.0)  # Motor: [0, 1] range
        mean[:, 1] = torch.tanh(mean[:, 1])  # Steering: [-1, 1] range
        return mean, self.log_std

class Critic(nn.Module):
    def __init__(self, state_dim, hidden_size=256):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(state_dim, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, 1)
        )
        
    def forward(self, x):
        return self.net(x)