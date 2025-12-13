# Installation & Quick Start Guide

## Prerequisites

- Python 3.8 or higher
- pip package manager

## Installation

### Option 1: Full Installation (Recommended)

Install all dependencies including PyBullet simulation:

```bash
# Clone repository
git clone https://github.com/ansh1113/rl-locomotion-cbf.git
cd rl-locomotion-cbf

# Install package with all dependencies
pip install -e .
```

This will install:
- NumPy, SciPy (numerical computing)
- OSQP (QP solver)
- PyBullet (physics simulation)
- Gym/Gymnasium (RL environment interface)
- Stable-Baselines3 (PPO implementation)
- PyTorch (neural networks)
- Matplotlib (visualization)

### Option 2: Core Modules Only

If you only want to use the safety/dynamics modules without simulation:

```bash
# Install minimal dependencies
pip install numpy scipy osqp

# Test core functionality
python test_core.py
```

## Quick Start

### 1. Verify Installation

Test that everything is working:

```bash
# Test core modules (no PyBullet required)
python test_core.py

# Test full imports (requires all dependencies)
python -c "from rl_locomotion_cbf import QuadrupedEnv, CBFSafetyFilter, PPOPolicy; print('✓ All imports successful')"
```

### 2. Train Your First Policy

Train a PPO policy on flat terrain:

```bash
python scripts/train_ppo.py \
  --env flat \
  --total-timesteps 1000000 \
  --save-path models/ppo_flat
```

Expected training time: 1-2 hours on CPU, 20-30 minutes on GPU.

### 3. Evaluate with Safety Filter

Evaluate the trained policy with CBF safety filtering:

```bash
python scripts/evaluate_safe.py \
  --policy models/ppo_flat.zip \
  --episodes 10 \
  --render
```

### 4. Compare Safe vs Unsafe

Compare performance with and without safety:

```bash
# Without safety filter
python scripts/evaluate_unsafe.py --policy models/ppo_flat.zip --episodes 10

# With safety filter
python scripts/evaluate_safe.py --policy models/ppo_flat.zip --episodes 10
```

### 5. Visualize CBF Interventions

See when and how the safety filter intervenes:

```bash
python scripts/visualize_cbf.py \
  --policy models/ppo_flat.zip \
  --terrain stairs
```

This creates a visualization plot showing:
- Action corrections over time
- CBF intervention events
- Robot height
- Robot velocity

## Example Usage

### Python API

```python
from rl_locomotion_cbf import (
    create_quadruped_env,
    PPOPolicy,
    CBFSafetyFilter
)

# Create environment
env = create_quadruped_env(terrain_type='mixed', render=False)

# Load trained policy
policy = PPOPolicy.load('models/ppo_mixed.zip', env=env)

# Create safety filter
safety_filter = CBFSafetyFilter(alpha=1.0)

# Run episode with safety
obs = env.reset()
done = False

while not done:
    # Get action from policy
    action_raw, _ = policy.predict(obs, deterministic=True)
    
    # Filter through CBF safety layer
    dynamics = env.get_dynamics()
    action_safe = safety_filter.filter(obs, action_raw, dynamics)
    
    # Execute safe action
    obs, reward, done, info = env.step(action_safe)

# Check statistics
stats = safety_filter.get_statistics()
print(f"Intervention rate: {stats['intervention_rate']*100:.1f}%")

env.close()
```

## Training on Different Terrains

```bash
# Flat terrain (easy, good for initial training)
python scripts/train_ppo.py --env flat --total-timesteps 1000000

# Rough terrain (random heightfield)
python scripts/train_ppo.py --env rough --difficulty 0.5 --total-timesteps 2000000

# Slopes
python scripts/train_ppo.py --env slopes --difficulty 0.6 --total-timesteps 2000000

# Stairs
python scripts/train_ppo.py --env stairs --difficulty 0.7 --total-timesteps 2000000

# Mixed (most challenging)
python scripts/train_ppo.py --env mixed --difficulty 0.8 --total-timesteps 3000000
```

## Configuration

Customize training by editing YAML files:

- `config/ppo_config.yaml` - PPO hyperparameters
- `config/cbf_config.yaml` - Safety filter parameters
- `config/env_config.yaml` - Environment settings

## Running Tests

```bash
# Install pytest
pip install pytest

# Run all tests
pytest tests/ -v

# Run specific test file
pytest tests/test_cbf.py -v
```

## Troubleshooting

### Import Errors

If you see `ModuleNotFoundError`:
```bash
# Make sure package is installed
pip install -e .

# Or check Python path
python -c "import sys; print('\n'.join(sys.path))"
```

### PyBullet Installation Issues

PyBullet may take time to build from source. If it fails:
```bash
# Try installing from conda
conda install -c conda-forge pybullet

# Or use pre-built wheel
pip install pybullet --no-cache-dir
```

### NumPy Version Conflicts

If you see warnings about NumPy 2.0 and Gym:
```bash
# Install gymnasium (maintained fork of gym)
pip install gymnasium

# The code uses lazy loading, so this shouldn't affect core functionality
```

### Memory Issues During Training

If training crashes with memory errors:
```bash
# Reduce batch size in config/ppo_config.yaml
# Change: batch_size: 64 -> batch_size: 32
```

## Next Steps

1. **Read the Theory**: See `docs/theory.md` for CBF mathematics
2. **Training Guide**: See `docs/training_guide.md` for detailed instructions
3. **Customize**: Edit configs to tune hyperparameters
4. **Experiment**: Try different terrains and difficulty levels
5. **Deploy**: Use trained policies in your own applications

## Getting Help

- **Documentation**: Check `docs/` folder
- **Examples**: See `scripts/example_complete.py`
- **Issues**: Open an issue on GitHub
- **Code**: All code is well-commented

## Performance Benchmarks

### Training Time (CPU)
- Flat terrain: 1-2 hours (1M timesteps)
- Mixed terrain: 4-6 hours (2M timesteps)

### Inference Speed
- Policy: ~1-2 ms
- CBF filter: ~2-3 ms
- **Total**: ~3-5 ms (200+ Hz)

### Expected Results
| Terrain | Without CBF Falls | With CBF Falls | Speed Retained |
|---------|------------------|----------------|----------------|
| Flat    | 5%              | 0%            | 95%           |
| Rough   | 15%             | 0-1%          | 90%           |
| Mixed   | 25%             | 0-2%          | 85%           |

## License

MIT License - see LICENSE file for details.

## Citation

If you use this code in your research, please cite:

```bibtex
@software{rl_locomotion_cbf_2025,
  author = {Bhansali, Ansh},
  title = {RL Locomotion with Safety Layer using Control Barrier Functions},
  year = {2025},
  url = {https://github.com/ansh1113/rl-locomotion-cbf}
}
```

## Contact

Ansh Bhansali - anshbhansali5@gmail.com
