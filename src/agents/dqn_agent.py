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
        self.num_motor_bins = 9  # Number of discrete motor actions
        self.num_steering_bins = 9  # Number of discrete steering actions
        total_actions = self.num_motor_bins * self.num_steering_bins

        self.policy_net = DQN(state_dim, total_actions, config["hidden_size"]).to(
            self.device
        )
        self.target_net = DQN(state_dim, total_actions, config["hidden_size"]).to(
            self.device
        )
        self.target_net.load_state_dict(self.policy_net.state_dict())
        self.training_steps = 0

        # Training parameters
        self.gamma = config["gamma"]
        self.epsilon = config["epsilon_start"]
        self.epsilon_end = config["epsilon_end"]
        self.epsilon_decay = config["epsilon_decay"]
        self.batch_size = config["batch_size"]

        # Calculate step sizes for continuous action conversion
        self.motor_step_size = 2.0 / (self.num_motor_bins - 1)  # For range [-1, 1]
        self.steering_step_size = 2.0 / (
            self.num_steering_bins - 1
        )  # For range [-1, 1]

        # Initialize optimizer
        self.optimizer = optim.Adam(
            self.policy_net.parameters(), lr=config["learning_rate"]
        )

        # Initialize replay buffer
        self.memory = ReplayBuffer(capacity=config["buffer_size"])

    @staticmethod
    def create_agent(state_dim, action_dim, config):
        agent = DQNAgent(state_dim, action_dim, config)
        if config.get("checkpoint"):
            print(f"Loading checkpoint from: {config['checkpoint']}")
            agent.load(config["checkpoint"])
        return agent

    def process_state(self, state):
        """Convert numpy state to tensor"""
        return torch.FloatTensor(state)

    def select_action(self, state, training=True):
        if training and random.random() < self.epsilon:
            return {"motor": random.uniform(-1, 1), "steering": random.uniform(-1, 1)}

        with torch.no_grad():
            state_tensor = self.process_state(state).unsqueeze(0).to(self.device)
            q_values = self.policy_net(state_tensor)

            # Find all actions that have the maximum Q-value
            max_q = q_values.max().item()
            max_actions = torch.where(q_values == max_q)[1].cpu().numpy()

            # Randomly select one of the max actions
            action_idx = np.random.choice(max_actions)

            # Convert discrete action index to continuous actions
            motor_idx = action_idx % self.num_motor_bins
            steering_idx = action_idx // self.num_motor_bins

            # Convert indices to continuous values
            motor = motor_idx * self.motor_step_size - 1.0
            steering = steering_idx * self.steering_step_size - 1.0

            return {"motor": motor, "steering": steering}

    def store_transition(self, state, action, reward, next_state, done):
        # Process states
        state_tensor = self.process_state(state)
        next_state_tensor = self.process_state(next_state)

        state_np = (
            state_tensor.cpu().numpy()
            if torch.is_tensor(state_tensor)
            else state_tensor
        )
        next_state_np = (
            next_state_tensor.cpu().numpy()
            if torch.is_tensor(next_state_tensor)
            else next_state_tensor
        )

        # Convert continuous actions to discrete indices
        motor_idx = int((action["motor"] + 1) / self.motor_step_size)
        steering_idx = int((action["steering"] + 1) / self.steering_step_size)

        # Clip indices to valid ranges
        motor_idx = np.clip(motor_idx, 0, self.num_motor_bins - 1)
        steering_idx = np.clip(steering_idx, 0, self.num_steering_bins - 1)

        # Convert to single action index
        action_idx = motor_idx + steering_idx * self.num_motor_bins

        self.memory.push(
            state_np,
            np.array([action_idx]),
            np.array([reward]),
            next_state_np,
            np.array([float(done)]),
        )

    def train_step(self):
        if len(self.memory) < self.batch_size:
            return None

        # Sample batch
        states, actions, rewards, next_states, dones = self.memory.sample(
            self.batch_size
        )

        batch = {
            "state": torch.FloatTensor(states).to(self.device),
            "action": torch.LongTensor(actions).to(self.device),
            "reward": torch.FloatTensor(rewards).to(self.device),
            "next_state": torch.FloatTensor(next_states).to(self.device),
            "done": torch.FloatTensor(dones).to(self.device),
        }

        # Ensure actions are valid
        total_actions = self.num_motor_bins * self.num_steering_bins
        batch["action"] = torch.clamp(batch["action"], 0, total_actions - 1)
        batch["action"] = batch["action"].view(-1, 1)

        # Compute Q(s_t, a)
        all_q_values = self.policy_net(batch["state"])
        current_q_values = all_q_values.gather(1, batch["action"])

        # Compute expected Q values
        with torch.no_grad():
            next_q_values = self.target_net(batch["next_state"]).max(1)[0].unsqueeze(1)
            expected_q_values = (
                batch["reward"] + (1 - batch["done"]) * self.gamma * next_q_values
            )

        # Compute loss and optimize
        loss = nn.MSELoss()(current_q_values, expected_q_values)
        self.optimizer.zero_grad()
        loss.backward()

        # Clip gradients
        torch.nn.utils.clip_grad_norm_(self.policy_net.parameters(), max_norm=10)

        self.optimizer.step()

        # Update target network
        self.target_net.load_state_dict(self.policy_net.state_dict())

        # Update epsilon
        self.epsilon = max(self.epsilon_end, self.epsilon * self.epsilon_decay)
        self.training_steps += 1

        return loss.item()

    def save(self, path):
        torch.save(
            {
                "policy_net_state_dict": self.policy_net.state_dict(),
                "target_net_state_dict": self.target_net.state_dict(),
                "optimizer_state_dict": self.optimizer.state_dict(),
                "training_steps": self.training_steps,
                "epsilon": self.epsilon,
            },
            path,
        )

    def load(self, path):
        checkpoint = torch.load(path)
        self.policy_net.load_state_dict(checkpoint["policy_net_state_dict"])
        self.target_net.load_state_dict(checkpoint["target_net_state_dict"])
        self.optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
        self.training_steps = checkpoint["training_steps"]
        self.epsilon = checkpoint["epsilon"]
