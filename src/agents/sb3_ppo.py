from stable_baselines3 import PPO
from sb3_contrib import RecurrentPPO
from stable_baselines3.common.callbacks import CheckpointCallback, CallbackList
from stable_baselines3.common.vec_env import DummyVecEnv
from stable_baselines3.common.monitor import Monitor

from common.sb3_plotter import SB3Plotter


class SB3PPOAgent:
    @staticmethod
    def create_agent(env, config, log_dir, use_lstm=False):

        # Setup monitoring and vectorization
        env = Monitor(env, str(log_dir))
        env = DummyVecEnv([lambda: env])

        # Load checkpoint if provided
        if config.get("checkpoint"):
            print(f"Loading model from: {config['checkpoint']}")
            if use_lstm:
                return RecurrentPPO.load(config["checkpoint"], env=env, device="cpu")
            else:
                return PPO.load(config["checkpoint"], env=env, device="cpu")

        # Common parameters for both PPO variants
        params = {
            "env": env,
            "learning_rate": config.get("learning_rate", 3e-4),
            "device": "cpu",
            "n_steps": config.get("n_steps", 256 if use_lstm else 2048),
            "batch_size": config.get("batch_size", 32 if use_lstm else 64),
            "n_epochs": config.get("n_epochs", 10),
            "gamma": config.get("gamma", 0.99),
            "gae_lambda": config.get("gae_lambda", 0.95),
            "ent_coef": config.get("ent_coef", 0.01),
            "verbose": config.get("verbose", 1),
            "tensorboard_log": str(log_dir / "tensorboard"),
        }

        if use_lstm:
            # LSTM-specific parameters
            policy_kwargs = dict(
                lstm_hidden_size=128,
                n_lstm_layers=1,
            )
            return RecurrentPPO(
                "MultiInputLstmPolicy", **params, policy_kwargs=policy_kwargs
            )
        else:
            # Regular PPO
            return PPO(
                (
                    "MultiInputPolicy"
                    if isinstance(env.observation_space, dict)
                    else "MlpPolicy"
                ),
                **params,
            )

    @staticmethod
    def train(agent, config, log_dir):
        map = config["scenario"]
        total_timesteps = config.get("total_timesteps", 1_000_000)

        # Setup callbacks
        checkpoint_callback = CheckpointCallback(
            save_freq=10000, save_path=str(log_dir / "checkpoints"), name_prefix="ppo"
        )
        metrics_callback = SB3Plotter(log_dir=log_dir)
        callbacks = CallbackList([checkpoint_callback, metrics_callback])

        # Train the agent
        agent.learn(
            total_timesteps=total_timesteps,
            callback=callbacks,
            reset_num_timesteps=False,
            progress_bar=True,
            tb_log_name=f"PPO_{map}",
        )

        # Save the final model
        final_model_path = str(log_dir / "final_model")
        agent.save(final_model_path)
        print(f"Final model saved to: {final_model_path}")

        # Clean up checkpoints
        checkpoint_dir = log_dir / "checkpoints"
        if checkpoint_dir.exists():
            import shutil

            shutil.rmtree(checkpoint_dir)
            print(f"Cleaned up checkpoints at: {checkpoint_dir}")
