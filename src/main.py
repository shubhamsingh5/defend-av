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
from common.env_wrapper import RacecarWrapper
from scripts.train import train

def get_scenario_path():
    current_dir = os.path.dirname(os.path.abspath(__file__))  # Gets src directory
    project_root = os.path.dirname(current_dir)  # Goes up one level to root
    scenario_path = os.path.join(project_root, 'racecar_gym', 'scenarios', 'austria.yml')
    print(f"Looking for scenario at: {scenario_path}")
    return scenario_path

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

def load_config(config_path='src/configs/dqn_config.yaml'):
    with open(config_path, 'r') as f:
        return yaml.safe_load(f)

def setup_environment(render_mode='rgb_array_follow'):
    scenario_path = get_scenario_path()
    env = gym.make('SingleAgentRaceEnv-v0',
                   render_mode=render_mode,
                   scenario=scenario_path)
    env = RacecarWrapper(env)  # Add the wrapper
    return env

def setup_logging(config):
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_dir = Path(f"runs/dqn/{timestamp}")
    log_dir.mkdir(parents=True, exist_ok=True)
    
    # Save config
    with open(log_dir / "config.yaml", "w") as f:
        yaml.dump(config, f)
    
    return log_dir

def main():
    # Load configuration
    config = load_config()
    print("Loaded config:", config)
    
    # Setup environment and wrapper
    render_mode = 'rgb_array_follow' if config['render'] else 'human'
    env = setup_environment(render_mode)
    
    # Get state and action dimensions from wrapped environment
    observation, _ = env.reset()
    state_dim = observation.shape[0]  # This is now a numpy array from our wrapper
    action_dim = 81  # 9 discrete actions for each of the 2 continuous dimensions
    
    print(f"State dimension: {state_dim}")
    print(f"Action dimension: {action_dim}")
    
    # Setup logging
    log_dir = setup_logging(config)
    
    # Setup visualization
    visualizer = Visualizer(
        render_mode=render_mode,
        record=config['record'],
        log_dir=log_dir
    )
    if config['render']:
        visualizer.init_screen()
    
    # Create agent
    agent = DQNAgent(state_dim, action_dim, config)
    
    # Load checkpoint if provided
    if config['checkpoint']:
        agent.load(config['checkpoint'])
    
    try:
        if config['mode'] == 'train':
            train(env, agent, config, log_dir, visualizer)
        else:
            from test import evaluate
            evaluate(env, agent, config, log_dir, visualizer)
    
    finally:
        visualizer.close()
        env.close()


if __name__ == "__main__":
    main()