import time
from torch.distributions import Normal
import torch
import numpy as np
import torch.nn as nn
import torch.optim as optim

from models.ppo import Actor, Critic
from common.utils import get_dynamic_steering_scale

class PPOAgent:
    def __init__(self, state_dim, action_dim, config):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        self.actor = Actor(state_dim, action_dim, config['hidden_size']).to(self.device)
        self.critic = Critic(state_dim, config['hidden_size']).to(self.device)
        
        self.actor_optimizer = optim.Adam(self.actor.parameters(), lr=config['actor_lr'])
        self.critic_optimizer = optim.Adam(self.critic.parameters(), lr=config['critic_lr'])
        
        self.clip_param = config.get('clip_param', 0.2)
        self.ppo_epochs = config.get('ppo_epochs', 10)
        self.batch_size = config.get('batch_size', 64)
        self.gamma = config.get('gamma', 0.99)
        self.gae_lambda = config.get('gae_lambda', 0.95)
        self.max_grad_norm = config.get('max_grad_norm', 0.5)
        self.entropy_coef = 0.7  # Add entropy bonus
        self.episode_count = 0
        self.std_init = 0.7  # Higher initial exploration
        self.std_decay = 0.9995  # Much slower decay
        self.std_min = 0.1 # Higher minimum exploration
        self.current_std = self.std_init
        self.action_momentum = 0.5
        self.speed_scale = 0.0  # For curriculum learning

        
        self.reset_buffer()
        
    def reset_buffer(self):
        self.buffer = {
            'states': [],
            'actions': [],
            'rewards': [],
            'values': [],
            'log_probs': [],
            'masks': [],
        }
        
    def select_action(self, state, training=True):
        state_tensor = torch.FloatTensor(state).unsqueeze(0).to(self.device)
        
        with torch.no_grad():
            mean, _ = self.actor(state_tensor)

            if training:
                # Fixed exploration std instead of network output
                std = torch.full_like(mean, self.current_std).to(self.device)
                dist = Normal(mean, std)
                action = dist.sample()
                
                # Store trajectory
                log_prob = dist.log_prob(action).sum(dim=-1)
                value = self.critic(state_tensor)
                self.buffer['states'].append(state)
                self.buffer['actions'].append(action.cpu().squeeze(0).numpy())
                self.buffer['log_probs'].append(log_prob.cpu().item())
                self.buffer['values'].append(value.cpu().item())
            else:
                action = mean
                
        # Update exploration and curriculum
        self.current_std = max(self.std_min, self.current_std * self.std_decay)
        self.speed_scale = min(1.0, self.speed_scale + 0.000001)  # Gradually increase speed
        
        action_np = action.cpu().squeeze(0).numpy()
        
        # Apply action momentum
        if len(self.buffer['actions']) > 0:
            prev_action = self.buffer['actions'][-1]
            action_np = self.action_momentum * prev_action + (1 - self.action_momentum) * action_np
            
        lidar_data = state[6:]
        # steering_scale = get_dynamic_steering_scale(lidar_data)

        # Convert actions
        motor = 0.1 * (action_np[0] + 1.0)  # Convert to [0,1]
        motor = 0.2 + (1.0 - 0.2) * motor * self.speed_scale  # Apply curriculum
        steering = np.clip(action_np[1], -0.5, 0.5 )
        # print(f"State: {state}")
        # print(f"Motor: {motor}, Steering: {steering}")
        # time.sleep(0.5)
        
        return {
            'motor': float(motor),
            'steering': float(steering)
        }
    
    def store_transition(self, state, action, reward, next_state, terminated):
        self.buffer['rewards'].append(reward)
        self.buffer['masks'].append(1 - float(terminated))
    
    def train_step(self):
        if len(self.buffer['states']) == 0:
            return None

        # Convert buffer to tensors
        states = torch.FloatTensor(np.array(self.buffer['states'])).to(self.device)
        actions = torch.FloatTensor(np.array(self.buffer['actions'])).to(self.device)
        old_log_probs = torch.FloatTensor(self.buffer['log_probs']).to(self.device)
        returns = self.compute_returns().to(self.device)
        advantages = (returns - torch.FloatTensor(self.buffer['values']).to(self.device))
        advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)
        
        dataset_size = len(states)
        indices = np.arange(dataset_size)
        
        total_loss = 0
        for _ in range(self.ppo_epochs):
            np.random.shuffle(indices)
            for start in range(0, dataset_size, self.batch_size):
                end = start + self.batch_size
                batch_indices = indices[start:end]
                
                # Get batch data
                state_batch = states[batch_indices]
                action_batch = actions[batch_indices]
                old_log_prob_batch = old_log_probs[batch_indices]
                advantage_batch = advantages[batch_indices]
                return_batch = returns[batch_indices]
                advantage_batch = (advantage_batch - advantage_batch.mean()) / (advantage_batch.std() + 1e-8)
                # Current policy
                mean, log_std = self.actor(state_batch)
                std = log_std.exp()
                dist = Normal(mean, std)
                new_log_prob = dist.log_prob(action_batch).sum(-1)
                
                # PPO policy loss
                ratio = (new_log_prob - old_log_prob_batch).exp()
                surr1 = ratio * advantage_batch
                surr2 = torch.clamp(ratio, 1.0 - self.clip_param, 1.0 + self.clip_param) * advantage_batch
                policy_loss = -torch.min(surr1, surr2).mean()
                
                # Value loss
                value_pred = self.critic(state_batch).squeeze()
                value_loss = 0.5 * (return_batch - value_pred).pow(2).mean()
                
                # Dynamic entropy coefficient
                current_entropy_coef = self.entropy_coef * (1 - self.speed_scale)
                entropy = dist.entropy().mean()
                loss = policy_loss + value_loss - current_entropy_coef * entropy
                
                # Update networks
                self.actor_optimizer.zero_grad()
                self.critic_optimizer.zero_grad()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(self.actor.parameters(), self.max_grad_norm)
                torch.nn.utils.clip_grad_norm_(self.critic.parameters(), self.max_grad_norm)
                self.actor_optimizer.step()
                self.critic_optimizer.step()
                
                total_loss += loss.item()

        avg_loss = total_loss / (self.ppo_epochs * dataset_size / self.batch_size)
        self.reset_buffer()
        return avg_loss
    
    def compute_returns(self):
        rewards = torch.FloatTensor(self.buffer['rewards'])
        masks = torch.FloatTensor(self.buffer['masks'])
        values = torch.FloatTensor(self.buffer['values'])
        
        returns = torch.zeros_like(rewards)
        gae = 0
        for t in reversed(range(len(rewards))):
            if t == len(rewards) - 1:
                next_value = 0
            else:
                next_value = values[t + 1]
            
            delta = rewards[t] + self.gamma * next_value * masks[t] - values[t]
            gae = delta + self.gamma * self.gae_lambda * masks[t] * gae
            returns[t] = gae + values[t]
            
        return returns

    def save(self, path):
        torch.save({
            'actor': self.actor.state_dict(),
            'critic': self.critic.state_dict(),
            'actor_optimizer': self.actor_optimizer.state_dict(),
            'critic_optimizer': self.critic_optimizer.state_dict()
        }, path)
    
    def load(self, path):
        checkpoint = torch.load(path)
        self.actor.load_state_dict(checkpoint['actor'])
        self.critic.load_state_dict(checkpoint['critic'])
        self.actor_optimizer.load_state_dict(checkpoint['actor_optimizer'])
        self.critic_optimizer.load_state_dict(checkpoint['critic_optimizer'])