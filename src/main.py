import argparse
import os
import yaml
import numpy as np
import gymnasium as gym

import racecar_gym.envs.gym_api
import pygame
import cv2
from pathlib import Path
from datetime import datetime

from agents.dqn_agent import DQNAgent
from agents.ppo_agent import PPOAgent
from common.env_wrapper import RacecarWrapper
from scripts.train import train

class Visualizer:
    def __init__(self, render_mode='rgb_array_follow', record=False, log_dir=None, width=640, height=480):
        self.render_mode = render_mode
        self.record = record
        self.width = width
        self.height = height
        self.screen = None
        self.video_writer = None
        
        if record and log_dir:
            video_path = log_dir / f"run_{datetime.now().strftime('%Y%m%d_%H%M%S')}.mp4"
            fourcc = cv2.VideoWriter_fourcc(*'mp4v')
            self.video_writer = cv2.VideoWriter(str(video_path), fourcc, 30, (width, height))
    
    def init_screen(self):
        if self.render_mode == 'rgb_array_follow':
            pygame.init()
            self.screen = pygame.display.set_mode((self.width, self.height))
    
    def render_frame(self, frame):
        if frame is not None:
            if self.screen is not None:
                surf = pygame.surfarray.make_surface(frame.swapaxes(0, 1))
                self.screen.blit(surf, (0, 0))
                pygame.display.flip()
            
            if self.record and self.video_writer is not None:
                self.video_writer.write(cv2.cvtColor(frame, cv2.COLOR_RGB2BGR))
    
    def close(self):
        if self.video_writer is not None:
            self.video_writer.release()
        if self.screen is not None:
            pygame.quit()

def get_scenario_path(config):
    current_dir = os.path.dirname(os.path.abspath(__file__))
    scenario_name = config.get('scenario', 'austria.yml')
    scenario_path = os.path.join(current_dir, 'scenarios', scenario_name)
    print(f"Looking for scenario at: {scenario_path}")
    return scenario_path

def load_config(config_path):
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
        print(f"Loading configuration from: {config_path}")
        return config

def setup_environment(render_mode='rgb_array_follow', config=None):
    scenario_path = get_scenario_path(config)
    base_env = gym.make('SingleAgentRaceEnv-v0',
                     render_mode=render_mode,
                     scenario=scenario_path)
    env = RacecarWrapper(base_env)
    return env

def setup_logging(config):
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_dir = Path(f"runs/{config['model_type']}/{timestamp}")
    log_dir.mkdir(parents=True, exist_ok=True)
    
    # Save config
    with open(log_dir / "config.yaml", "w") as f:
        yaml.dump(config, f)
    
    return log_dir

def create_agent(config, state_dim, action_dim):
    """Create agent based on config type"""
    print(f"Creating {config['model_type']} agent...")
    if config['model_type'] == 'dqn':
        return DQNAgent(state_dim, action_dim, config)
    elif config['model_type'] == 'ppo':
        return PPOAgent(state_dim, action_dim, config)
    else:
        raise ValueError(f"Unknown model type: {config['model_type']}")

def main():
    parser = argparse.ArgumentParser(description="Train/Evaluate DQN or PPO agent on racing environment")
    parser.add_argument('--config', type=str, required=True, help="Path to the configuration file")
    args = parser.parse_args()

    # Load configuration
    config = load_config(args.config)
    print("Configuration loaded:", config)
    
    # Setup environment and wrapper
    render_mode = 'rgb_array_follow' if config['render'] else 'human'
    env = setup_environment(render_mode, config)
    
    # Get state and action dimensions from wrapped environment
    observation, _ = env.reset()
    print(f"Initial observation shape: {observation.shape}")
    
    # Get dimensions directly from the observation
    state_dim = observation.shape[0]
    
    # Action dimensions differ between DQN and PPO
    if config['model_type'] == 'dqn':
        action_dim = 81  # 9x9 discrete actions (motor x steering)
    else:  # PPO uses continuous actions
        action_dim = 2   # motor and steering directly
    
    print(f"State dimension: {state_dim}")
    print(f"Action dimension: {action_dim}")
    print(f"Using model type: {config['model_type']}")

    # Setup logging
    log_dir = setup_logging(config)
    print(f"Logging to: {log_dir}")
    
    # Setup visualization
    visualizer = Visualizer(
        render_mode=render_mode,
        record=config['record'],
        log_dir=log_dir
    )
    if config['render']:
        visualizer.init_screen()
    
    # Create agent
    agent = create_agent(config, state_dim, action_dim)
    
    # Load checkpoint if provided
    if config['checkpoint']:
        print(f"Loading checkpoint from: {config['checkpoint']}")
        agent.load(config['checkpoint'])
    
    try:
        if config['mode'] == 'train':
            print("Starting training...")
            train(env, agent, config, log_dir, visualizer)
        else:
            print("Starting evaluation...")
            evaluate(env, agent, config, log_dir, visualizer)
    
    except KeyboardInterrupt:
        print("\nTraining interrupted by user")
    except Exception as e:
        print(f"Error during execution: {e}")
        raise e
    finally:
        print("Cleaning up...")
        visualizer.close()
        env.close()
        print("Done!")

if __name__ == "__main__":
    main()