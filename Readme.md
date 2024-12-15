# DefendAV

Project for UofT ECE1508: Applied Deep Learning

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
├── scenarios/            # Race track configurations
├── runs/                 # Training logs and model checkpoints
└── requirements.txt      # Project dependencies
```

## Usage

1. Modify the configuration in `configs/config.yaml`:
```yaml
mode: 'train'  # or 'eval'
agent: 'dqn'   # type of agent to use
render: false  # whether to render environment
### Training

1. Configure your training settings in `configs/{MODEL}_config.yaml`:
```yaml
model: "ppo"              # Options: "dqn", "ppo", "ppo_sb3"
mode: "train"             # Options: "train", "eval"
scenario: "austria.yml"   # Race track configuration
...other hyperparameters
```

2. Start training:
```bash
python src/main.py --config path/to/config.yaml
```

### Evaluation

1. Update your config file to evaluation mode and specify the checkpoint:
```yaml
mode: "eval"
checkpoint: "path/to/model_or_checkpoint"
```

2. Run evaluation:
```bash
python src/main.py --config path/to/config.yaml
```

## Models

The project includes three main types of agents:

1. **DQN (Deep Q-Network)**
   - Discrete action space
   - Experience replay
   - Target network for stable training

2. **PPO (Proximal Policy Optimization)**
   - Continuous action space
   - On-policy learning
   - Adaptive exploration

3. **SB3 PPO (Stable-Baselines3 Implementation)**

## Monitoring

Training progress is automatically logged and can be found in the `runs/` directory:
- Tensorboard logs
- Model checkpoints
- Performance plots
- Training metrics
