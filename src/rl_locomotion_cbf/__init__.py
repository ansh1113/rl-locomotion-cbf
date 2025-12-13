"""RL Locomotion with Control Barrier Functions."""

__version__ = "0.1.0"

# Lazy imports to avoid requiring all dependencies
def _import_envs():
    """Lazy import of environment modules."""
    from .envs import QuadrupedEnv, TerrainGenerator, create_quadruped_env
    return QuadrupedEnv, TerrainGenerator, create_quadruped_env

def _import_policies():
    """Lazy import of policy modules."""
    from .policies import PPOPolicy, train_ppo, ActorCriticNetwork
    return PPOPolicy, train_ppo, ActorCriticNetwork

def _import_safety():
    """Lazy import of safety modules."""
    from .safety import CBFSafetyFilter, create_barrier_set
    return CBFSafetyFilter, create_barrier_set

def _import_dynamics():
    """Lazy import of dynamics modules."""
    from .dynamics import QuadrupedDynamics, linearize_dynamics
    return QuadrupedDynamics, linearize_dynamics

# Provide access via __getattr__ for convenience
def __getattr__(name):
    """Lazy loading of modules."""
    if name in ('QuadrupedEnv', 'TerrainGenerator', 'create_quadruped_env'):
        QuadrupedEnv, TerrainGenerator, create_quadruped_env = _import_envs()
        globals().update({
            'QuadrupedEnv': QuadrupedEnv,
            'TerrainGenerator': TerrainGenerator,
            'create_quadruped_env': create_quadruped_env
        })
        return globals()[name]
    elif name in ('PPOPolicy', 'train_ppo', 'ActorCriticNetwork'):
        PPOPolicy, train_ppo, ActorCriticNetwork = _import_policies()
        globals().update({
            'PPOPolicy': PPOPolicy,
            'train_ppo': train_ppo,
            'ActorCriticNetwork': ActorCriticNetwork
        })
        return globals()[name]
    elif name in ('CBFSafetyFilter', 'create_barrier_set'):
        CBFSafetyFilter, create_barrier_set = _import_safety()
        globals().update({
            'CBFSafetyFilter': CBFSafetyFilter,
            'create_barrier_set': create_barrier_set
        })
        return globals()[name]
    elif name in ('QuadrupedDynamics', 'linearize_dynamics'):
        QuadrupedDynamics, linearize_dynamics = _import_dynamics()
        globals().update({
            'QuadrupedDynamics': QuadrupedDynamics,
            'linearize_dynamics': linearize_dynamics
        })
        return globals()[name]
    raise AttributeError(f"module '{__name__}' has no attribute '{name}'")

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


