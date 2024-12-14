from stable_baselines3.common.callbacks import BaseCallback
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

class SB3Plotter(BaseCallback):
    def __init__(self, log_dir: Path, verbose=0):
        super(SB3Plotter, self).__init__(verbose)
        self.log_dir = log_dir
        self.episode_rewards = []
        self.episode_lengths = []
        self.losses = []
        self.policy_gradient_losses = []
        self.value_losses = []
        self.entropy_losses = []

    def _on_step(self) -> bool:
        # Collect reward and episode length
        if "episode" in self.locals["infos"][0]:  # Works with VecEnvs
            info = self.locals["infos"][0]["episode"]
            self.episode_rewards.append(info["r"])
            self.episode_lengths.append(info["l"])

        return True

    def _on_rollout_end(self) -> None:
        # Collect metrics from SB3 logger
        if "loss" in self.logger.name_to_value.keys():
            self.losses.append(self.logger.name_to_value["loss"])
        if "policy_gradient_loss" in self.logger.name_to_value.keys():
            self.policy_gradient_losses.append(self.logger.name_to_value["policy_gradient_loss"])
        if "value_loss" in self.logger.name_to_value.keys():
            self.value_losses.append(self.logger.name_to_value["value_loss"])
        if "entropy_loss" in self.logger.name_to_value.keys():
            self.entropy_losses.append(self.logger.name_to_value["entropy_loss"])

    def _on_training_end(self) -> None:
        # Save plots at the end of training
        metrics = {
            "episode_rewards": self.episode_rewards,
            "avg_rewards": self.compute_moving_average(self.episode_rewards),
            "episode_lengths": self.episode_lengths,
            "losses": self.losses,
            "policy_gradient_losses": self.policy_gradient_losses,
            "value_losses": self.value_losses,
            "entropy_losses": self.entropy_losses,
        }
        self.plot_metrics(metrics)

    def compute_moving_average(self, data, window=100):
        return np.convolve(data, np.ones(window) / window, mode="valid") if len(data) >= window else data

    def plot_metrics(self, metrics):
        self.log_dir.mkdir(parents=True, exist_ok=True)

        # Plot episode rewards
        plt.figure(figsize=(10, 6))
        plt.plot(metrics["episode_rewards"], alpha=0.3, label="Episode Reward")
        if metrics["avg_rewards"] is not None:
            plt.plot(metrics["avg_rewards"], label="Average Reward (100 episodes)")
        plt.xlabel("Episode")
        plt.ylabel("Reward")
        plt.legend()
        plt.savefig(self.log_dir / "rewards.png")
        plt.close()

        # Plot losses
        plt.figure(figsize=(10, 6))
        plt.plot(metrics["losses"], alpha=0.5, label="Total Loss")
        plt.plot(metrics["policy_gradient_losses"], label="Policy Gradient Loss")
        plt.plot(metrics["value_losses"], label="Value Loss")
        plt.plot(metrics["entropy_losses"], label="Entropy Loss")
        plt.xlabel("Training Step")
        plt.ylabel("Loss")
        plt.yscale("log")
        plt.legend()
        plt.savefig(self.log_dir / "losses.png")
        plt.close()