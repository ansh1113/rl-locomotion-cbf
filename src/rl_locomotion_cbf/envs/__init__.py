"""Environment modules for quadruped locomotion."""
from .quadruped_env import QuadrupedEnv
from .terrain_generator import TerrainGenerator

__all__ = ['QuadrupedEnv', 'TerrainGenerator']
