import argparse
import os
import yaml
import gymnasium as gym


import racecar_gym.envs.gym_api
from pathlib import Path
from datetime import datetime

from agents.dqn_agent import DQNAgent
from agents.ppo_agent import PPOAgent
from agents.sb3_ppo import SB3PPOAgent
from common.env_wrapper import RacecarWrapper
from scripts.train import train


def load_config(config_path):
    with open(config_path, "r") as f:
        config = yaml.safe_load(f)
        print(f"Loading configuration from: {config_path}")

        # Convert numerical values to proper types
        numerical_keys = [
            "learning_rate",
            "buffer_size",
            "batch_size",
            "gamma",
            "hidden_size",
            "episodes",
            "train_freq",
            "min_samples",
            "epsilon_start",
            "epsilon_end",
            "epsilon_decay",
        ]

        for key in numerical_keys:
            if key in config:
                config[key] = float(config[key])
                # Convert to int for parameters that should be integers
                if key in [
                    "buffer_size",
                    "batch_size",
                    "hidden_size",
                    "episodes",
                    "train_freq",
                    "min_samples",
                ]:
                    config[key] = int(config[key])

        print(f"Learning rate type: {type(config['learning_rate'])}")
        return config


def get_scenario_path(config):
    current_dir = os.path.dirname(os.path.abspath(__file__))
    scenario_name = config.get("scenario", "austria.yml")
    scenario_path = os.path.join(current_dir, "scenarios", scenario_name)
    print(f"Looking for scenario at: {scenario_path}")
    return scenario_path


def setup_environment(render_mode="rgb_array_follow", config=None):
    scenario_path = get_scenario_path(config)
    base_env = gym.make(
        "SingleAgentRaceEnv-v0", render_mode=render_mode, scenario=scenario_path
    )

    env = RacecarWrapper(base_env)

    return env


def setup_logging(config):
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_dir = Path(f"runs/{config['model']}/{timestamp}")
    log_dir.mkdir(parents=True, exist_ok=True)

    # Save config
    with open(log_dir / "config.yaml", "w") as f:
        yaml.dump(config, f)

    return log_dir


def create_agent(env, config, state_dim, action_dim, log_dir):
    """Create agent based on config type"""
    print(f"Creating {config['model']} agent...")
    if config["model"] == "dqn":
        return DQNAgent.create_agent(state_dim, action_dim, config)
    elif config["model"] == "ppo":
        return PPOAgent.create_agent(state_dim, action_dim, config)
    elif config["model"] == "ppo_sb3":
        return SB3PPOAgent.create_agent(env, config, log_dir, use_lstm=False)
    else:
        raise ValueError(f"Unknown model type: {config['model']}")


def main():
    parser = argparse.ArgumentParser(
        description="Train/Evaluate DQN or PPO agent on racing environment"
    )
    parser.add_argument(
        "--config", type=str, required=True, help="Path to the configuration file"
    )
    args = parser.parse_args()

    # Load configuration
    config = load_config(args.config)
    print("Configuration loaded:", config)

    # Setup logging
    log_dir = setup_logging(config)
    print(f"Logging to: {log_dir}")

    # Setup environment and wrapper
    # TODO: add back rgb_array_follow
    render_mode = "human"
    env = setup_environment(render_mode, config)

    # Create agent
    observation, _ = env.reset()
    state_dim = observation.shape[0]
    if config["model"] == "dqn":
        action_dim = 81  # 9x9 discrete actions (motor x steering)
    else:  # PPO uses continuous actions
        action_dim = 2  # motor and steering directly

    print(f"State dimension: {state_dim}")
    print(f"Action dimension: {action_dim}")

    agent = create_agent(env, config, state_dim, action_dim, log_dir)

    # Load checkpoint if provided
    try:
        if config["mode"] == "train":
            print("Starting training...")
            if config["model"] == "ppo_sb3":
                SB3PPOAgent.train(agent, config, log_dir)
            else:
                train(env, agent, config, log_dir)
        else:
            print("Starting evaluation...")
            # evaluate(env, agent, config, log_dir)

    except KeyboardInterrupt:
        print("\nTraining interrupted by user")
    except Exception as e:
        print(f"Error during execution: {e}")
        raise e
    finally:
        print("Cleaning up...")
        env.close()
        print("Done!")


if __name__ == "__main__":
    main()
