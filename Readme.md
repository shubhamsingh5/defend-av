# DefendAV

## Installation

1. Create and activate a virtual environment:
```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

2. Install dependencies:
```bash
pip install -r requirements.txt
```

## Project Structure
```
dl-project/
├── src/
│   ├── agents/            # RL agents (DQN, PPO implementations)
│   ├── common/            # Shared utilities (replay buffer, env wrapper)
│   ├── models/            # Neural network architectures
│   └── scripts/           # Training and evaluation scripts
├── configs/               # Configuration files
├── checkpoints/           # Saved model weights
└── results/               # Training logs and visualizations
```

## Usage

1. Modify the configuration in `configs/config.yaml`:
```yaml
mode: 'train'  # or 'eval'
agent: 'dqn'   # type of agent to use
render: false  # whether to render environment

# Training parameters
train:
  episodes: 1000
  train_freq: 100
  batch_size: 320
  # ... other parameters

# Agent parameters
dqn:
  hidden_size: 256
  learning_rate: 0.001
  # ... other parameters
```

2. Run training:
```bash
python src/main.py
```