"""Terrain generation for varied locomotion environments."""
import numpy as np
from typing import Tuple, Optional
import pybullet as p


class TerrainGenerator:
    """
    Generate various terrain types for quadruped locomotion testing.
    
    Supports:
    - Flat terrain
    - Random heightfield (rough terrain)
    - Slopes
    - Stairs
    - Mixed terrain with obstacles
    """
    
    def __init__(self, terrain_type: str = 'flat', difficulty: float = 0.5):
        """
        Initialize terrain generator.
        
        Args:
            terrain_type: Type of terrain to generate
            difficulty: Difficulty level 0 (easy) to 1 (hard)
        """
        self.terrain_type = terrain_type
        self.difficulty = np.clip(difficulty, 0.0, 1.0)
        
    def generate(self, physics_client: int) -> int:
        """
        Generate terrain in PyBullet simulation.
        
        Args:
            physics_client: PyBullet client ID
            
        Returns:
            Terrain body ID
        """
        if self.terrain_type == 'flat':
            return self._generate_flat(physics_client)
        elif self.terrain_type == 'slopes':
            return self._generate_slopes(physics_client)
        elif self.terrain_type == 'stairs':
            return self._generate_stairs(physics_client)
        elif self.terrain_type == 'rough':
            return self._generate_rough(physics_client)
        elif self.terrain_type == 'mixed':
            return self._generate_mixed(physics_client)
        else:
            return self._generate_flat(physics_client)
    
    def _generate_flat(self, physics_client: int) -> int:
        """Generate flat ground plane."""
        import pybullet_data
        p.setAdditionalSearchPath(pybullet_data.getDataPath())
        return p.loadURDF("plane.urdf", physicsClientId=physics_client)
    
    def _generate_rough(self, physics_client: int) -> int:
        """Generate rough terrain with random heightfield."""
        terrain_size = 256
        mesh_scale = [0.05, 0.05, 1.0]
        
        # Generate random heights
        heightfield = self._create_random_heightfield(
            size=terrain_size,
            max_height=0.1 * self.difficulty
        )
        
        # Create collision shape
        terrain_shape = p.createCollisionShape(
            shapeType=p.GEOM_HEIGHTFIELD,
            meshScale=mesh_scale,
            heightfieldTextureScaling=terrain_size // 2,
            heightfieldData=heightfield.flatten(),
            numHeightfieldRows=terrain_size,
            numHeightfieldColumns=terrain_size,
            physicsClientId=physics_client
        )
        
        # Create terrain body
        terrain_id = p.createMultiBody(
            baseMass=0,
            baseCollisionShapeIndex=terrain_shape,
            basePosition=[0, 0, 0],
            physicsClientId=physics_client
        )
        
        # Set friction
        p.changeDynamics(
            terrain_id,
            -1,
            lateralFriction=1.0,
            physicsClientId=physics_client
        )
        
        return terrain_id
    
    def _generate_slopes(self, physics_client: int) -> int:
        """Generate terrain with slopes."""
        terrain_size = 256
        mesh_scale = [0.05, 0.05, 1.0]
        
        # Create sloped heightfield
        heightfield = self._create_slope_heightfield(
            size=terrain_size,
            slope_angle=0.2 * self.difficulty  # Max ~11 degrees
        )
        
        # Create collision shape
        terrain_shape = p.createCollisionShape(
            shapeType=p.GEOM_HEIGHTFIELD,
            meshScale=mesh_scale,
            heightfieldTextureScaling=terrain_size // 2,
            heightfieldData=heightfield.flatten(),
            numHeightfieldRows=terrain_size,
            numHeightfieldColumns=terrain_size,
            physicsClientId=physics_client
        )
        
        terrain_id = p.createMultiBody(
            baseMass=0,
            baseCollisionShapeIndex=terrain_shape,
            physicsClientId=physics_client
        )
        
        return terrain_id
    
    def _generate_stairs(self, physics_client: int) -> int:
        """Generate stairs terrain."""
        # Base plane
        base_id = self._generate_flat(physics_client)
        
        # Add stair steps
        step_width = 0.3
        step_height = 0.05 * (1 + self.difficulty)
        step_depth = 0.4
        num_steps = int(5 + 5 * self.difficulty)
        
        for i in range(num_steps):
            step_collision = p.createCollisionShape(
                p.GEOM_BOX,
                halfExtents=[step_depth / 2, step_width / 2, step_height / 2],
                physicsClientId=physics_client
            )
            
            step_position = [
                i * step_depth,
                0,
                i * step_height + step_height / 2
            ]
            
            p.createMultiBody(
                baseMass=0,
                baseCollisionShapeIndex=step_collision,
                basePosition=step_position,
                physicsClientId=physics_client
            )
        
        return base_id
    
    def _generate_mixed(self, physics_client: int) -> int:
        """Generate mixed terrain with various features."""
        # Start with rough terrain
        terrain_id = self._generate_rough(physics_client)
        
        # Add some obstacles (boxes)
        num_obstacles = int(3 + 7 * self.difficulty)
        for _ in range(num_obstacles):
            size = np.random.uniform(0.1, 0.3)
            collision = p.createCollisionShape(
                p.GEOM_BOX,
                halfExtents=[size, size, size],
                physicsClientId=physics_client
            )
            
            position = [
                np.random.uniform(-2, 2),
                np.random.uniform(-2, 2),
                size
            ]
            
            p.createMultiBody(
                baseMass=0,
                baseCollisionShapeIndex=collision,
                basePosition=position,
                physicsClientId=physics_client
            )
        
        return terrain_id
    
    def _create_random_heightfield(
        self,
        size: int,
        max_height: float
    ) -> np.ndarray:
        """Create random heightfield with smoothing."""
        # Generate random noise
        heightfield = np.random.uniform(-max_height, max_height, (size, size))
        
        # Apply Gaussian smoothing
        try:
            from scipy import ndimage
            heightfield = ndimage.gaussian_filter(heightfield, sigma=3.0)
        except ImportError:
            # Fallback to simple box filter if scipy not available
            kernel_size = 5
            heightfield = self._box_filter(heightfield, kernel_size)
        
        return heightfield
    
    def _create_slope_heightfield(
        self,
        size: int,
        slope_angle: float
    ) -> np.ndarray:
        """Create heightfield with slopes."""
        heightfield = np.zeros((size, size))
        
        # Create linear slope in x direction
        for i in range(size):
            height = i * slope_angle / size
            heightfield[i, :] = height
        
        # Add some random variation
        noise = np.random.uniform(-0.01, 0.01, (size, size))
        heightfield += noise
        
        return heightfield
    
    def _box_filter(self, array: np.ndarray, size: int) -> np.ndarray:
        """Simple box filter for smoothing."""
        padded = np.pad(array, size // 2, mode='edge')
        result = np.zeros_like(array)
        
        for i in range(array.shape[0]):
            for j in range(array.shape[1]):
                window = padded[i:i+size, j:j+size]
                result[i, j] = np.mean(window)
        
        return result
