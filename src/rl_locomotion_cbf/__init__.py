"""RL Locomotion with Control Barrier Functions."""

# Core modules
from .envs import QuadrupedEnv, TerrainGenerator, create_quadruped_env
from .policies import PPOPolicy, train_ppo, ActorCriticNetwork
from .safety import CBFSafetyFilter, create_barrier_set
from .dynamics import QuadrupedDynamics, linearize_dynamics

__version__ = "0.1.0"

__all__ = [
    # Environment
    'QuadrupedEnv',
    'TerrainGenerator',
    'create_quadruped_env',
    
    # Policies
    'PPOPolicy',
    'train_ppo',
    'ActorCriticNetwork',
    
    # Safety
    'CBFSafetyFilter',
    'create_barrier_set',
    
    # Dynamics
    'QuadrupedDynamics',
    'linearize_dynamics',
]

