#!/usr/bin/env python3
"""Visualize CBF safety interventions."""
import argparse
import numpy as np
import matplotlib.pyplot as plt
from collections import deque
from rl_locomotion_cbf import (
    create_quadruped_env,
    PPOPolicy,
    CBFSafetyFilter
)


def main():
    """Main visualization function."""
    parser = argparse.ArgumentParser(description='Visualize CBF interventions')
    parser.add_argument('--policy', type=str, required=True,
                       help='Path to trained policy')
    parser.add_argument('--terrain', type=str, default='stairs',
                       choices=['flat', 'mixed', 'slopes', 'stairs', 'rough'],
                       help='Terrain type')
    parser.add_argument('--alpha', type=float, default=1.0,
                       help='CBF alpha parameter')
    parser.add_argument('--max-steps', type=int, default=500,
                       help='Maximum steps to visualize')
    args = parser.parse_args()
    
    # Create environment with rendering
    print(f"Creating {args.terrain} terrain environment")
    env = create_quadruped_env(
        terrain_type=args.terrain,
        terrain_difficulty=0.7,
        render=True
    )
    
    # Load policy
    print(f"Loading policy from {args.policy}")
    policy = PPOPolicy.load(args.policy, env=env)
    
    # Create safety filter
    safety_filter = CBFSafetyFilter(alpha=args.alpha)
    
    # Data collection
    barrier_values = []
    intervention_flags = []
    corrections = []
    heights = []
    velocities = []
    
    # Run episode
    print(f"\nRunning episode with visualization...")
    obs = env.reset()
    done = False
    step = 0
    
    while not done and step < args.max_steps:
        # Get action
        action_raw, _ = policy.predict(obs, deterministic=True)
        
        # Filter through CBF
        dynamics = env.get_dynamics()
        action_safe = safety_filter.filter(obs, action_raw, dynamics)
        
        # Record data
        correction = np.linalg.norm(action_safe - action_raw)
        corrections.append(correction)
        intervention_flags.append(1 if correction > 1e-3 else 0)
        
        # Extract state info
        if len(obs) >= 3:
            heights.append(obs[2])  # z position
        if len(obs) >= 6:
            velocities.append(np.linalg.norm(obs[3:6]))  # velocity magnitude
        
        # Execute action
        obs, reward, done, info = env.step(action_safe)
        step += 1
        
        if step % 50 == 0:
            print(f"Step {step}, Interventions: {sum(intervention_flags)}")
    
    print(f"\nEpisode complete: {step} steps")
    print(f"Total CBF interventions: {sum(intervention_flags)}")
    print(f"Intervention rate: {100*sum(intervention_flags)/len(intervention_flags):.1f}%")
    
    # Plot results
    plot_results(corrections, intervention_flags, heights, velocities)
    
    # Cleanup
    env.close()


def plot_results(corrections, interventions, heights, velocities):
    """Plot visualization results."""
    fig, axes = plt.subplots(4, 1, figsize=(12, 10))
    steps = range(len(corrections))
    
    # Plot 1: Corrections
    axes[0].plot(steps, corrections, 'b-', linewidth=1)
    axes[0].set_ylabel('Action Correction')
    axes[0].set_title('CBF Action Corrections Over Time')
    axes[0].grid(True, alpha=0.3)
    
    # Plot 2: Interventions
    axes[1].plot(steps, interventions, 'r-', linewidth=2)
    axes[1].fill_between(steps, 0, interventions, alpha=0.3, color='red')
    axes[1].set_ylabel('Intervention')
    axes[1].set_title('CBF Active Interventions (1 = Active, 0 = Inactive)')
    axes[1].set_ylim([-0.1, 1.1])
    axes[1].grid(True, alpha=0.3)
    
    # Plot 3: Height
    if len(heights) > 0:
        axes[2].plot(steps[:len(heights)], heights, 'g-', linewidth=1.5)
        axes[2].axhline(y=0.2, color='r', linestyle='--', label='Min safe height')
        axes[2].set_ylabel('Height (m)')
        axes[2].set_title('Robot Body Height')
        axes[2].legend()
        axes[2].grid(True, alpha=0.3)
    
    # Plot 4: Velocity
    if len(velocities) > 0:
        axes[3].plot(steps[:len(velocities)], velocities, 'm-', linewidth=1.5)
        axes[3].set_ylabel('Velocity (m/s)')
        axes[3].set_xlabel('Step')
        axes[3].set_title('Robot Velocity')
        axes[3].grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('cbf_visualization.png', dpi=150)
    print("\nVisualization saved to cbf_visualization.png")
    plt.show()


if __name__ == "__main__":
    main()
