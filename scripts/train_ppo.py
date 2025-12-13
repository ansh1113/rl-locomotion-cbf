#!/usr/bin/env python3
"""Train PPO policy for quadruped locomotion."""
import argparse
import os
from rl_locomotion_cbf import create_quadruped_env, train_ppo


def main():
    """Main training function."""
    parser = argparse.ArgumentParser(description='Train PPO policy for quadruped')
    parser.add_argument('--env', type=str, default='flat',
                       choices=['flat', 'mixed', 'slopes', 'stairs', 'rough'],
                       help='Terrain type')
    parser.add_argument('--difficulty', type=float, default=0.5,
                       help='Terrain difficulty (0-1)')
    parser.add_argument('--total-timesteps', type=int, default=1000000,
                       help='Total training timesteps')
    parser.add_argument('--learning-rate', type=float, default=3e-4,
                       help='Learning rate')
    parser.add_argument('--save-path', type=str, default='models/ppo_policy',
                       help='Path to save trained model')
    parser.add_argument('--render', action='store_true',
                       help='Enable rendering during training')
    args = parser.parse_args()
    
    # Create environment
    print(f"Creating {args.env} terrain environment (difficulty={args.difficulty})")
    env = create_quadruped_env(
        terrain_type=args.env,
        terrain_difficulty=args.difficulty,
        render=args.render
    )
    
    # Train policy
    print(f"Training PPO policy for {args.total_timesteps} timesteps")
    policy = train_ppo(
        env=env,
        total_timesteps=args.total_timesteps,
        learning_rate=args.learning_rate,
        save_path=args.save_path,
        verbose=1
    )
    
    print(f"Training complete! Model saved to {args.save_path}")
    
    # Cleanup
    env.close()


if __name__ == "__main__":
    main()

