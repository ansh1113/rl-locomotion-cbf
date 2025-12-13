#!/usr/bin/env python3
"""Evaluate PPO policy with CBF safety filter."""
import argparse
import numpy as np
from rl_locomotion_cbf import (
    create_quadruped_env,
    PPOPolicy,
    CBFSafetyFilter
)


def main():
    """Main evaluation function."""
    parser = argparse.ArgumentParser(description='Evaluate safe PPO policy')
    parser.add_argument('--policy', type=str, required=True,
                       help='Path to trained policy')
    parser.add_argument('--episodes', type=int, default=10,
                       help='Number of evaluation episodes')
    parser.add_argument('--env', type=str, default='mixed',
                       choices=['flat', 'mixed', 'slopes', 'stairs', 'rough'],
                       help='Terrain type')
    parser.add_argument('--difficulty', type=float, default=0.5,
                       help='Terrain difficulty (0-1)')
    parser.add_argument('--render', action='store_true',
                       help='Enable rendering')
    parser.add_argument('--alpha', type=float, default=1.0,
                       help='CBF alpha parameter')
    args = parser.parse_args()
    
    # Create environment
    print(f"Creating {args.env} terrain environment")
    env = create_quadruped_env(
        terrain_type=args.env,
        terrain_difficulty=args.difficulty,
        render=args.render
    )
    
    # Load policy
    print(f"Loading policy from {args.policy}")
    policy = PPOPolicy.load(args.policy, env=env)
    
    # Create safety filter
    print(f"Creating CBF safety filter (alpha={args.alpha})")
    safety_filter = CBFSafetyFilter(alpha=args.alpha)
    
    # Evaluate
    print(f"\nEvaluating for {args.episodes} episodes with safety filter")
    
    episode_rewards = []
    episode_lengths = []
    num_falls = 0
    
    for episode in range(args.episodes):
        obs = env.reset()
        done = False
        episode_reward = 0
        episode_length = 0
        episode_interventions = 0
        
        while not done:
            # Get action from policy
            action_raw, _ = policy.predict(obs, deterministic=True)
            
            # Filter through CBF safety layer
            dynamics = env.get_dynamics()
            action_safe = safety_filter.filter(obs, action_raw, dynamics)
            
            # Check if CBF intervened
            if np.linalg.norm(action_safe - action_raw) > 1e-3:
                episode_interventions += 1
            
            # Execute action
            obs, reward, done, info = env.step(action_safe)
            episode_reward += reward
            episode_length += 1
            
            if done and info.get('fell', False):
                num_falls += 1
        
        episode_rewards.append(episode_reward)
        episode_lengths.append(episode_length)
        
        print(f"Episode {episode+1}: Reward={episode_reward:.2f}, "
              f"Length={episode_length}, Interventions={episode_interventions}")
    
    # Print statistics
    print("\n" + "="*60)
    print("EVALUATION RESULTS (WITH SAFETY)")
    print("="*60)
    print(f"Episodes: {args.episodes}")
    print(f"Mean reward: {np.mean(episode_rewards):.2f} ± {np.std(episode_rewards):.2f}")
    print(f"Mean length: {np.mean(episode_lengths):.1f} ± {np.std(episode_lengths):.1f}")
    print(f"Falls: {num_falls} ({100*num_falls/args.episodes:.1f}%)")
    
    # CBF statistics
    stats = safety_filter.get_statistics()
    print(f"\nCBF Statistics:")
    print(f"  Intervention rate: {100*stats['intervention_rate']:.1f}%")
    print(f"  Average correction: {stats['avg_correction']:.3f}")
    print(f"  QP failures: {stats['qp_failures']}")
    print("="*60)
    
    # Cleanup
    env.close()


if __name__ == "__main__":
    main()
