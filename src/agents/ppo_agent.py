import torch
import torch.optim as optim
from common.replay_buffer import ReplayBuffer
from .base_agent import BaseAgent
from models.ppo import PPO
import numpy as np

class PPOAgent(BaseAgent):
    def __init__(self, state_dim, action_dim, config):
        super().__init__(state_dim, action_dim, config)
        self.model = PPO(state_dim, action_dim, config['hidden_size']).to(self.device)
        self.optimizer = optim.Adam(self.model.parameters(), lr=config['learning_rate'])
        self.gamma = config['gamma']
        self.clip_ratio = config['clip_ratio']
        self.entropy_coeff = config['entropy_coeff']
        self.epoch = config['epoch']
        self.memory = ReplayBuffer(capacity=config['buffer_size'])

    def select_action(self, state, training=True):
        state_tensor = self.process_state(state).to(self.device)
        with torch.no_grad():
            action_probs, _ = self.model(state_tensor.unsqueeze(0))
        return torch.tanh(action_probs.squeeze()).cpu().numpy()

    def train_step(self):
        if len(self.memory) < self.config['batch_size']:
            return

        states, actions, rewards, next_states, dones = self.memory.sample(self.config['batch_size'])
        states = torch.FloatTensor(states).to(self.device)
        actions = torch.FloatTensor(actions).to(self.device)
        rewards = torch.FloatTensor(rewards).to(self.device)
        dones = torch.FloatTensor(dones).to(self.device)

        _, values = self.model(states)
        advantages = rewards + self.gamma * values * (1 - dones) - values
        advantages = advantages.detach()

        for _ in range(self.epoch):
            # PPO surrogate loss
            action_probs, values = self.model(states)
            ratios = torch.exp(action_probs - torch.log(actions))
            surrogate1 = ratios * advantages
            surrogate2 = torch.clamp(ratios, 1 - self.clip_ratio, 1 + self.clip_ratio) * advantages
            actor_loss = -torch.min(surrogate1, surrogate2).mean()
            critic_loss = (values - rewards).pow(2).mean()
            entropy_loss = -action_probs.mean()

            loss = actor_loss + 0.5 * critic_loss + self.entropy_coeff * entropy_loss
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()

        return loss.item()

    def store_transition(self, state, action, reward, next_state, done):
        state_tensor = self.process_state(state)
        next_state_tensor = self.process_state(next_state)
        self.memory.push(state_tensor.cpu().numpy(), action, reward, next_state_tensor.cpu().numpy(), done)
