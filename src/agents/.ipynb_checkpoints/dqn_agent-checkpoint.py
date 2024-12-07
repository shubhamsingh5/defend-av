import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from collections import deque
import random
from .base_agent import BaseAgent
from models.dqn import DQN
from common.replay_buffer import ReplayBuffer

class DQNAgent(BaseAgent):
    def __init__(self, state_dim, action_dim, config):
        super().__init__(state_dim, action_dim, config)
        
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print(f"Using device: {self.device}")
        
        # Initialize networks
        self.policy_net = DQN(state_dim, action_dim, config['hidden_size']).to(self.device)
        self.target_net = DQN(state_dim, action_dim, config['hidden_size']).to(self.device)
        self.target_net.load_state_dict(self.policy_net.state_dict())
        self.training_steps = 0

        # Training parameters
        self.gamma = config['gamma']
        self.epsilon = config['epsilon_start']
        self.epsilon_end = config['epsilon_end']
        self.epsilon_decay = config['epsilon_decay']
        self.action_step_size = config['action_step_size']
        self.batch_size = config['batch_size']
        self.target_update_freq = config['target_update_freq']
        
        # Initialize optimizer
        self.optimizer = optim.Adam(self.policy_net.parameters(), lr=config['learning_rate'])
        
        # Initialize replay buffer
        self.memory = ReplayBuffer(capacity=config['buffer_size'])

    def process_state(self, state):
        """Convert state dict to tensor"""
        try:
            if isinstance(state, dict):
                # Handle dictionary observation (unwrapped environment)
                state_components = []
                
                # Add components in a specific order
                for key in ['pose', 'velocity', 'acceleration', 'lidar', 'time']:
                    if key in state:
                        val = np.array(state[key], dtype=np.float32).flatten()
                        state_components.append(val)
                
                if not state_components:
                    raise ValueError(f"No valid components found in state: {state.keys()}")
                
                state_vector = np.concatenate(state_components)
                return torch.FloatTensor(state_vector)
            else:
                # Handle already processed state (from wrapper)
                return torch.FloatTensor(state)
                
        except Exception as e:
            print(f"Error in process_state: {e}")
            print(f"State type: {type(state)}")
            if isinstance(state, dict):
                print(f"State keys: {state.keys()}")
            raise e

    def select_action(self, state, training=True):
        if training and random.random() < self.epsilon:
            # Random action
            return {
                'motor': random.uniform(-1, 1),
                'steering': random.uniform(-1, 1)
            }
        
        with torch.no_grad():
            state_tensor = self.process_state(state).unsqueeze(0).to(self.device)
            q_values = self.policy_net(state_tensor)
            action_idx = q_values.max(1)[1].item()
            
            # Convert discrete action index to continuous actions
            motor_idx = action_idx % 9
            steering_idx = action_idx // 9
            
            motor = motor_idx * self.action_step_size - 1.0  # [-1, 1]
            steering = steering_idx * self.action_step_size - 1.0  # [-1, 1]
            
            return {
                'motor': motor,
                'steering': steering
            }

    def store_transition(self, state, action, reward, next_state, done):
        state_tensor = self.process_state(state)
        next_state_tensor = self.process_state(next_state)
        
        # Convert to numpy once during storage
        state_np = state_tensor.cpu().numpy() if torch.is_tensor(state_tensor) else state_tensor
        next_state_np = next_state_tensor.cpu().numpy() if torch.is_tensor(next_state_tensor) else next_state_tensor
        
        # Convert continuous actions to discrete indices
        motor_idx = int((action['motor'] + 1) / 0.25)
        steering_idx = int((action['steering'] + 1) / 0.25)
        action_idx = motor_idx + steering_idx * 9
        
        self.memory.push(state_np, 
                        np.array([action_idx]), 
                        np.array([reward]), 
                        next_state_np,
                        np.array([float(done)]))


    def train_step(self):
        if len(self.memory) < self.batch_size:
            print(f"Not enough samples: {len(self.memory)} < {self.batch_size}")  # Add this
            return None

        states, actions, rewards, next_states, dones = self.memory.sample(self.batch_size)
        states = torch.FloatTensor(states).to(self.device)
        actions = torch.LongTensor(actions).to(self.device)
        rewards = torch.FloatTensor(rewards).to(self.device)
        next_states = torch.FloatTensor(next_states).to(self.device)
        dones = torch.FloatTensor(dones).to(self.device)

        # Generate adversarial states
        adversarial_states = torch.stack([self.generate_adversarial_example(s) for s in states])
        
        actions = torch.clamp(actions, 0, self.policy_net(states).shape[1] - 1)
        actions = actions.view(-1, 1)

        # Train on both clean and adversarial data
        for train_states in [states, adversarial_states]:
            q_values = self.policy_net(train_states).gather(1, actions.view(-1, 1))
            with torch.no_grad():
                next_q_values = self.target_net(next_states).max(1)[0]
                expected_q_values = rewards + (1 - dones) * self.gamma * next_q_values

            loss = nn.MSELoss()(q_values, expected_q_values.unsqueeze(1))
            self.optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(self.policy_net.parameters(), max_norm=10) # Clip gradients (optional, but can help with stability)
            self.optimizer.step()

        # Update target network
        if self.training_steps % self.target_update_freq == 0:
            self.target_net.load_state_dict(self.policy_net.state_dict())

        # Update epsilon
        self.epsilon = max(self.epsilon_end, self.epsilon * self.epsilon_decay)
        self.training_steps += 1

        return loss.item()

    def save(self, path):
        torch.save({
            'policy_net_state_dict': self.policy_net.state_dict(),
            'target_net_state_dict': self.target_net.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'training_steps': self.training_steps,
            'epsilon': self.epsilon
        }, path)

    def load(self, path):
        checkpoint = torch.load(path)
        self.policy_net.load_state_dict(checkpoint['policy_net_state_dict'])
        self.target_net.load_state_dict(checkpoint['target_net_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        self.training_steps = checkpoint['training_steps']
        self.epsilon = checkpoint['epsilon']

    def generate_adversarial_example(self, state, epsilon=0.01):
        """
        Generate adversarial examples using FGSM.
        :param state: Original state
        :param epsilon: Perturbation magnitude
        :return: Perturbed state
        """
        state_tensor = self.process_state(state).unsqueeze(0).to(self.device).requires_grad_(True)
        q_values = self.policy_net(state_tensor)
        loss = q_values.max()  # Target the highest Q-value
        self.optimizer.zero_grad()
        loss.backward()
        perturbation = epsilon * state_tensor.grad.sign()
        adversarial_state = state_tensor + perturbation
        return torch.clamp(adversarial_state, -1, 1).squeeze(0).cpu()