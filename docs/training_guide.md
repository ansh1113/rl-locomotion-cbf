# Training Guide

This guide explains how to train PPO policies for quadruped locomotion and integrate them with CBF safety filters.

## Quick Start

### 1. Train a Basic Policy

```bash
python scripts/train_ppo.py \
  --env flat \
  --total-timesteps 1000000 \
  --save-path models/ppo_flat
```

### 2. Evaluate with Safety

```bash
python scripts/evaluate_safe.py \
  --policy models/ppo_flat.zip \
  --episodes 10 \
  --render
```

### 3. Compare with Unsafe

```bash
python scripts/evaluate_unsafe.py \
  --policy models/ppo_flat.zip \
  --episodes 10
```

## Training Stages

### Stage 1: Flat Terrain (Baseline)

Train on flat terrain first to establish baseline locomotion:

```bash
python scripts/train_ppo.py \
  --env flat \
  --total-timesteps 1000000 \
  --learning-rate 3e-4 \
  --save-path models/ppo_flat
```

**Expected Results:**
- Training time: ~1-2 hours on CPU
- Final reward: 50-100
- Success rate: ~95%

### Stage 2: Rough Terrain

Train on rough terrain with randomized heightfields:

```bash
python scripts/train_ppo.py \
  --env rough \
  --difficulty 0.5 \
  --total-timesteps 2000000 \
  --save-path models/ppo_rough
```

**Expected Results:**
- Training time: ~2-4 hours
- More robust to perturbations
- Success rate: ~85%

### Stage 3: Mixed Terrain (Final)

Train on mixed terrain with obstacles:

```bash
python scripts/train_ppo.py \
  --env mixed \
  --difficulty 0.7 \
  --total-timesteps 3000000 \
  --save-path models/ppo_mixed
```

**Expected Results:**
- Training time: ~4-6 hours
- Handles various terrain types
- Success rate: ~80% without CBF, ~100% with CBF

## Hyperparameter Tuning

### Learning Rate

Default: `3e-4`

- Increase if training is too slow
- Decrease if training is unstable

```bash
--learning-rate 5e-4  # Faster
--learning-rate 1e-4  # More stable
```

### Training Duration

Default: `1000000` timesteps

- Flat terrain: 1M timesteps sufficient
- Complex terrain: 2-3M timesteps recommended

```bash
--total-timesteps 3000000
```

### Terrain Difficulty

Default: `0.5` (scale 0-1)

- Start with 0.3 for easier training
- Gradually increase to 0.8 for robustness

```bash
--difficulty 0.8
```

## PPO Hyperparameters

Edit `config/ppo_config.yaml` to tune PPO:

```yaml
ppo:
  learning_rate: 3.0e-4
  n_steps: 2048        # Rollout length
  batch_size: 64       # Minibatch size
  n_epochs: 10         # Optimization epochs
  gamma: 0.99          # Discount factor
  gae_lambda: 0.95     # GAE lambda
  clip_range: 0.2      # PPO clip range
  ent_coef: 0.01       # Entropy coefficient
```

### Key Parameters:

- **n_steps**: Higher = more stable but slower
- **batch_size**: Higher = more stable, needs more memory
- **ent_coef**: Higher = more exploration

## Reward Shaping

The reward function is defined in `config/env_config.yaml`:

```yaml
reward:
  forward_velocity: 1.5      # Encourage forward motion
  lateral_penalty: -0.5      # Discourage sideways drift
  energy_penalty: -0.01      # Encourage efficiency
  smoothness_penalty: -0.05  # Encourage smooth motion
  survival_bonus: 0.1        # Encourage staying upright
```

### Tips:

1. **Forward velocity** should be primary reward
2. **Energy penalty** prevents extreme actions
3. **Smoothness penalty** improves gait quality
4. Balance exploration vs exploitation with entropy

## Curriculum Learning

For challenging terrains, use curriculum learning:

### Method 1: Difficulty Ramp

```bash
# Stage 1: Easy terrain
python scripts/train_ppo.py --env mixed --difficulty 0.3 --total-timesteps 1000000 --save-path models/stage1

# Stage 2: Medium terrain (continue from stage 1)
python scripts/train_ppo.py --env mixed --difficulty 0.6 --total-timesteps 1000000 --policy models/stage1.zip --save-path models/stage2

# Stage 3: Hard terrain
python scripts/train_ppo.py --env mixed --difficulty 0.9 --total-timesteps 1000000 --policy models/stage2.zip --save-path models/stage3
```

### Method 2: Terrain Progression

```bash
# 1. Flat → 2. Slopes → 3. Stairs → 4. Mixed
python scripts/train_ppo.py --env flat --total-timesteps 500000 --save-path models/curr1
python scripts/train_ppo.py --env slopes --total-timesteps 500000 --policy models/curr1.zip --save-path models/curr2
python scripts/train_ppo.py --env stairs --total-timesteps 500000 --policy models/curr2.zip --save-path models/curr3
python scripts/train_ppo.py --env mixed --total-timesteps 1000000 --policy models/curr3.zip --save-path models/final
```

## CBF Integration

### Training Without CBF

CBF is **not used during training** - this allows the policy to learn natural, efficient gaits.

```bash
python scripts/train_ppo.py --env mixed --total-timesteps 2000000
```

### Testing With CBF

CBF is applied **only at test time** as a safety filter:

```bash
python scripts/evaluate_safe.py --policy models/ppo_mixed.zip --alpha 1.0
```

### CBF Parameter Tuning

The `alpha` parameter controls safety vs performance:

```bash
# More aggressive (faster but less safe)
python scripts/evaluate_safe.py --policy models/ppo_mixed.zip --alpha 0.5

# Balanced (recommended)
python scripts/evaluate_safe.py --policy models/ppo_mixed.zip --alpha 1.0

# Conservative (safer but slower)
python scripts/evaluate_safe.py --policy models/ppo_mixed.zip --alpha 2.0
```

## Monitoring Training

### TensorBoard (if enabled)

```bash
tensorboard --logdir logs/
```

View:
- Episode rewards
- Policy loss
- Value loss
- Entropy

### Command Line Output

Watch for:
- Increasing mean reward
- Decreasing policy loss
- Stable value loss

## Troubleshooting

### Training Not Converging

1. **Reduce learning rate**: `--learning-rate 1e-4`
2. **Increase rollout length**: Edit `n_steps` in config
3. **Simplify terrain**: Start with `--difficulty 0.3`

### Policy Falls Frequently

1. **Increase training time**: More timesteps needed
2. **Adjust reward weights**: Increase `survival_bonus` and `fall_penalty`
3. **Check terrain difficulty**: May be too hard

### CBF Too Conservative

1. **Decrease alpha**: `--alpha 0.5`
2. **Reduce safety margins**: Edit `config/cbf_config.yaml`
3. **Use fewer barriers**: Disable some in config

### QP Solver Failures

1. **Increase slack penalty**: Edit `cbf_config.yaml`
2. **Relax constraints**: Increase margins
3. **Check state bounds**: Ensure realistic values

## Best Practices

1. **Start simple**: Train on flat terrain first
2. **Use curriculum**: Gradually increase difficulty
3. **Monitor closely**: Check training metrics regularly
4. **Save frequently**: Use checkpoints every 50k steps
5. **Test both ways**: Compare with and without CBF
6. **Tune incrementally**: Change one parameter at a time

## Performance Benchmarks

### Training Time (CPU)

- Flat terrain: 1-2 hours (1M steps)
- Mixed terrain: 4-6 hours (2M steps)

### Expected Results

| Terrain | Reward (no CBF) | Reward (with CBF) | Falls (no CBF) | Falls (with CBF) |
|---------|----------------|-------------------|----------------|------------------|
| Flat    | 80-100        | 75-95            | 5%            | 0%              |
| Rough   | 60-80         | 55-75            | 15%           | 0%              |
| Mixed   | 50-70         | 45-65            | 25%           | 0-1%            |

## Next Steps

After successful training:

1. **Visualize behavior**: `scripts/visualize_cbf.py`
2. **Analyze safety**: Check intervention rates
3. **Test edge cases**: Try extreme terrain difficulties
4. **Deploy**: Use policy in your application
