import time
import numpy as np
import torch
from pathlib import Path
from datetime import datetime
import matplotlib.pyplot as plt
import json
from tqdm import tqdm

def update_metrics(metrics, episode_reward, episode_length, progress, loss, epsilon=None):
    metrics['episode_rewards'].append(episode_reward)
    metrics['episode_lengths'].append(episode_length)
    metrics['progress_values'].append(progress)
    metrics['losses'].append(loss)
    if epsilon is not None:
        metrics['epsilon_values'].append(epsilon)
    
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
    plt.yscale('log')
    plt.legend()
    plt.savefig(log_dir / 'loss.png')
    plt.close()
    
    # Plot epsilon decay if using DQN
    if 'epsilon_values' in metrics and metrics['epsilon_values']:
        plt.figure(figsize=(10, 6))
        plt.plot(metrics['epsilon_values'])
        plt.xlabel('Episode')
        plt.ylabel('Epsilon')
        plt.title('Epsilon Decay Over Training')
        plt.savefig(log_dir / 'epsilon_decay.png')
        plt.close()

def save_metrics(metrics, log_dir):
    metrics_file = log_dir / 'metrics.json'
    with open(metrics_file, 'w') as f:
        json.dump(metrics, f, indent=4)

def train(env, agent, config, log_dir):
    # Initialize metrics tracking
    metrics = {
        'episode_rewards': [],
        'episode_lengths': [],
        'progress_values': [],
        'avg_rewards': [],
        'losses': [],
        'epsilon_values': [] if hasattr(agent, 'epsilon') else None
    }
    
    # Training parameters
    train_freq = config['train_freq']
    min_samples = config['min_samples']
    
    
    best_reward = float('-inf')
    episode_loss = []
    total_steps = 0
    
    # Initialize progress bar for episodes
    pbar = tqdm(range(config['episodes']), desc='Training')
    
    steps_since_train = 0
    for episode in pbar:
        state, _ = env.reset()
        episode_reward = 0
        episode_steps = 0
        episode_loss.clear()
        
        while True:
            # Select and perform action
            action = agent.select_action(state, training=True)
            next_state, reward, terminated, truncated, info = env.step(action)
            
            # Store transition
            agent.store_transition(state, action, reward, next_state, terminated)
            total_steps += 1
            
            if total_steps >= min_samples:
                steps_since_train += 1
                if steps_since_train >= train_freq:
                    loss = agent.train_step()
                    if loss is not None:
                        episode_loss.append(loss)
                    steps_since_train = 0
            
            episode_reward += reward
            episode_steps += 1
            state = next_state
            
            if terminated or truncated:
                break
        
        # Update metrics
        avg_episode_loss = np.mean(episode_loss) if episode_loss else 0
        update_metrics(
            metrics,
            episode_reward,
            episode_steps,
            info.get('progress', 0),
            avg_episode_loss,
            agent.epsilon if hasattr(agent, 'epsilon') else None
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
        pbar_postfix = {
            'reward': f'{episode_reward:.2f}',
            'avg_reward': f'{avg_reward:.2f}',
            'progress': f'{info.get("progress", 0):.2f}',
            'loss': f'{avg_episode_loss:.3e}' if episode_loss else 'N/A',
            'samples': total_steps,
            'steps': episode_steps,
        }
        
        # Add epsilon to progress bar if using DQN
        if hasattr(agent, 'epsilon'):
            pbar_postfix['epsilon'] = f'{agent.epsilon:.3f}'
            
        pbar.set_postfix(pbar_postfix)
    
    # Final saves
    agent.save(log_dir / 'final_model.pth')
    plot_metrics(metrics, log_dir)
    save_metrics(metrics, log_dir)
    
    return metrics