import numpy as np
import torch
from pathlib import Path
from datetime import datetime
import matplotlib.pyplot as plt
from collections import deque
import json
from tqdm import tqdm

def update_metrics(metrics, episode_reward, episode_length, progress, loss, learning_rate=None):
    metrics['episode_rewards'].append(episode_reward)
    metrics['episode_lengths'].append(episode_length)
    metrics['progress_values'].append(progress)
    metrics['losses'].append(loss)
    if learning_rate is not None:
        metrics['learning_rates'].append(learning_rate)
    
    # Update running averages
    window = 100
    metrics['avg_rewards'].append(
        np.mean(metrics['episode_rewards'][-window:])
    )

def plot_metrics(metrics, log_dir):
    # Plot episode rewards
    plt.figure(figsize=(10, 6))
    plt.plot(metrics['episode_rewards'], alpha=0.3, label='Episode Reward')
    plt.plot(metrics['avg_rewards'], label='Average Reward (100 episodes)')
    plt.xlabel('Episode')
    plt.ylabel('Reward')
    plt.legend()
    plt.savefig(log_dir / 'rewards.png')
    plt.close()
    
    # Plot episode lengths
    plt.figure(figsize=(10, 6))
    plt.plot(metrics['episode_lengths'])
    plt.xlabel('Episode')
    plt.ylabel('Episode Length')
    plt.savefig(log_dir / 'episode_lengths.png')
    plt.close()
    
    # Plot loss curve
    plt.figure(figsize=(10, 6))
    plt.plot(metrics['losses'], alpha=0.3, label='Training Loss')
    # Plot smoothed loss
    window = 100
    smoothed_loss = np.convolve(metrics['losses'], np.ones(window)/window, mode='valid')
    plt.plot(smoothed_loss, label=f'Average Loss ({window} steps)')
    plt.xlabel('Training Step')
    plt.ylabel('Loss')
    plt.yscale('log')  # Loss is often better viewed in log scale
    plt.legend()
    plt.savefig(log_dir / 'loss.png')
    plt.close()
    
    if metrics['learning_rates']:
        plt.figure(figsize=(10, 6))
        plt.plot(metrics['learning_rates'])
        plt.xlabel('Episode')
        plt.ylabel('Learning Rate')
        plt.yscale('log')
        plt.savefig(log_dir / 'learning_rates.png')
        plt.close()

def save_metrics(metrics, log_dir):
    metrics_file = log_dir / 'metrics.json'
    with open(metrics_file, 'w') as f:
        json.dump(metrics, f, indent=4)

def train(env, agent, config, log_dir, visualizer):
    # Initialize metrics tracking
    metrics = {
        'episode_rewards': [],
        'episode_lengths': [],
        'progress_values': [],
        'learning_rates': [],
        'avg_rewards': [],
        'losses': []
    }
    
    # Training parameters
    train_freq = config.get('train_freq', 100)     # Train every 100 steps
    batch_size = config.get('batch_size', 320)
    min_samples = max(batch_size * 2, 1000)        # Minimum samples before training
    
    # Short episode handling
    short_episode_threshold = config.get('short_episode_threshold', 50)
    max_short_episodes = config.get('max_short_episodes', 5)
    short_episode_counter = 0
    original_epsilon = agent.epsilon if hasattr(agent, 'epsilon') else None
    
    best_reward = float('-inf')
    episode_loss = []
    total_steps = 0
    
    # Initialize progress bar for episodes
    pbar = tqdm(range(config['episodes']), desc='Training')
    
    for episode in pbar:
        state, _ = env.reset()
        episode_reward = 0
        episode_steps = 0
        episode_loss.clear()
        steps_since_train = 0
        
        # Check if stuck in short episodes
        if short_episode_counter >= max_short_episodes and hasattr(agent, 'epsilon'):
            agent.epsilon = min(1.0, agent.epsilon * 2)
            short_episode_counter = 0
            print("\nIncreasing exploration due to repeated short episodes")
        
        while True:
            # Select and perform action
            action = agent.select_action(state, training=True)
            next_state, reward, terminated, truncated, info = env.step(action)
            
            # Store transition
            agent.store_transition(state, action, reward, next_state, terminated)
            total_steps += 1
            
            # Initial sample collection phase
            if total_steps < min_samples:
                if total_steps % 100 == 0:
                    pbar.set_postfix({'status': f'Collecting initial samples: {total_steps}/{min_samples}'})
            else:
                # Normal training phase
                steps_since_train += 1
                if steps_since_train >= train_freq:
                    loss = agent.train_step()
                    if loss is not None:
                        episode_loss.append(loss)
                    steps_since_train = 0

            
            # Render if visualizer is active
            if episode_steps % 2 == 0:
                frame = env.render()
                visualizer.render_frame(frame)
            
            episode_reward += reward
            episode_steps += 1
            state = next_state
            
            if terminated or truncated:
                # # Handle short episode detection
                # if episode_steps < short_episode_threshold:
                #     short_episode_counter += 1
                # else:
                #     short_episode_counter = 0
                #     if hasattr(agent, 'epsilon') and original_epsilon:
                #         agent.epsilon = original_epsilon
                break
        
        # Update metrics
        avg_episode_loss = np.mean(episode_loss) if episode_loss else 0
        update_metrics(
            metrics,
            episode_reward,
            episode_steps,
            info.get('progress', 0),
            avg_episode_loss,
            agent.learning_rate if hasattr(agent, 'learning_rate') else None
        )
        
        # Save best model
        if episode_reward > best_reward:
            best_reward = episode_reward
            agent.save(log_dir / 'best_model.pth')
        
        # Regular model checkpoint and plotting
        if episode % 100 == 0:
            agent.save(log_dir / f'model_episode_{episode}.pth')
            plot_metrics(metrics, log_dir)
            save_metrics(metrics, log_dir)
        
        # Update progress bar with current metrics
        avg_reward = np.mean(metrics['episode_rewards'][-100:]) if metrics['episode_rewards'] else 0
        pbar.set_postfix({
            'reward': f'{episode_reward:.2f}',
            'avg_reward': f'{avg_reward:.2f}',
            'progress': f'{info.get("progress", 0):.2f}',
            'loss': f'{avg_episode_loss:.3e}' if episode_loss else 'N/A',
            'samples': total_steps,
            'short_eps': short_episode_counter,
            'steps': episode_steps,
            'std': f'{agent.current_std:.2f}',
            'speed_scale': f'{agent.speed_scale:.2f}'
        })
    
    # Final saves
    agent.save(log_dir / 'final_model.pth')
    plot_metrics(metrics, log_dir)
    save_metrics(metrics, log_dir)
    
    return metrics