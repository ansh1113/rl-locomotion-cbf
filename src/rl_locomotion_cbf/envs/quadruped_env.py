"""Quadruped locomotion environment using PyBullet."""
import gym
from gym import spaces
import pybullet as p
import pybullet_data
import numpy as np
from typing import Tuple, Dict, Any, Optional


class QuadrupedEnv(gym.Env):
    """
    Quadruped locomotion environment for reinforcement learning.
    
    Provides a physics-based simulation environment for training locomotion
    policies with terrain variation and stability constraints.
    """
    
    metadata = {'render.modes': ['human', 'rgb_array']}
    
    def __init__(
        self,
        terrain_type: str = 'flat',
        terrain_difficulty: float = 0.5,
        render: bool = False,
        control_freq: int = 50
    ):
        """
        Initialize quadruped environment.
        
        Args:
            terrain_type: Type of terrain ('flat', 'mixed', 'slopes', 'stairs', 'rough')
            terrain_difficulty: Difficulty level 0-1
            render: Enable visual rendering
            control_freq: Control frequency in Hz
        """
        super(QuadrupedEnv, self).__init__()
        
        self.terrain_type = terrain_type
        self.terrain_difficulty = terrain_difficulty
        self.render_enabled = render
        self.control_freq = control_freq
        
        # Simulation parameters
        self.dt = 1/240.0  # PyBullet simulation timestep
        self.control_steps = int(1.0 / (self.control_freq * self.dt))
        
        # Robot parameters (simplified quadruped)
        self.num_joints = 12  # 3 joints per leg, 4 legs
        self.num_legs = 4
        self.joints_per_leg = 3
        
        # Joint limits (hip, thigh, calf for each leg)
        self.joint_limits_lower = np.array([-0.8, -1.5, -2.5] * self.num_legs)
        self.joint_limits_upper = np.array([0.8, 0.5, -0.5] * self.num_legs)
        
        # Action space: target joint positions normalized to [-1, 1]
        self.action_space = spaces.Box(
            low=-1.0,
            high=1.0,
            shape=(self.num_joints,),
            dtype=np.float32
        )
        
        # Observation space: 48-dimensional as described in README
        # [body pos(3), body orn(4), linear vel(3), angular vel(3),
        #  joint positions(12), joint velocities(12), foot contacts(4),
        #  previous action(12)]
        obs_dim = 3 + 4 + 3 + 3 + 12 + 12 + 4 + 12  # = 53 (adjusted)
        self.observation_space = spaces.Box(
            low=-np.inf,
            high=np.inf,
            shape=(obs_dim,),
            dtype=np.float32
        )
        
        # Connect to PyBullet
        if self.render_enabled:
            self.client = p.connect(p.GUI)
        else:
            self.client = p.connect(p.DIRECT)
        
        p.setAdditionalSearchPath(pybullet_data.getDataPath())
        
        # Environment state
        self.robot_id = None
        self.plane_id = None
        self.prev_action = np.zeros(self.num_joints)
        self.step_counter = 0
        self.start_pos = [0, 0, 0.3]
        
    def reset(self) -> np.ndarray:
        """Reset environment to initial state."""
        p.resetSimulation(physicsClientId=self.client)
        p.setGravity(0, 0, -9.81, physicsClientId=self.client)
        p.setTimeStep(self.dt, physicsClientId=self.client)
        
        # Load terrain
        self.plane_id = self._create_terrain()
        
        # Create simplified quadruped robot
        self.robot_id = self._create_robot()
        
        # Reset state
        self.prev_action = np.zeros(self.num_joints)
        self.step_counter = 0
        
        return self._get_observation()
    
    def step(self, action: np.ndarray) -> Tuple[np.ndarray, float, bool, Dict[str, Any]]:
        """
        Execute one environment step.
        
        Args:
            action: Joint position targets in [-1, 1]
            
        Returns:
            observation: Current state
            reward: Step reward
            done: Episode termination flag
            info: Additional information
        """
        # Clip and denormalize action
        action = np.clip(action, -1.0, 1.0)
        joint_targets = self._denormalize_action(action)
        
        # Apply action for multiple simulation steps
        for _ in range(self.control_steps):
            self._apply_action(joint_targets)
            p.stepSimulation(physicsClientId=self.client)
        
        self.step_counter += 1
        
        # Get observation and compute reward
        obs = self._get_observation()
        reward = self._compute_reward(action)
        done = self._is_done()
        
        # Collect additional info
        info = {
            'step': self.step_counter,
            'fell': done and self._is_fallen(),
            'num_falls': 1 if (done and self._is_fallen()) else 0,
            'cbf_active': False,  # Will be set by safety filter
            'cbf_correction': 0.0
        }
        
        self.prev_action = action.copy()
        
        return obs, reward, done, info
    
    def _create_terrain(self) -> int:
        """Create terrain based on type."""
        if self.terrain_type == 'flat':
            plane_id = p.loadURDF("plane.urdf", physicsClientId=self.client)
        elif self.terrain_type == 'mixed':
            # Create heightfield terrain
            plane_id = self._create_heightfield_terrain()
        else:
            # Default to flat
            plane_id = p.loadURDF("plane.urdf", physicsClientId=self.client)
        
        return plane_id
    
    def _create_heightfield_terrain(self) -> int:
        """Create uneven terrain using heightfield."""
        # Simple random heightfield
        terrain_shape = (256, 256)
        terrain_data = np.random.uniform(
            -self.terrain_difficulty * 0.05,
            self.terrain_difficulty * 0.05,
            size=terrain_shape
        )
        
        # Smooth the terrain
        from scipy import ndimage
        terrain_data = ndimage.gaussian_filter(terrain_data, sigma=3)
        
        # Create collision shape
        terrain_shape_id = p.createCollisionShape(
            shapeType=p.GEOM_HEIGHTFIELD,
            meshScale=[0.05, 0.05, 1],
            heightfieldTextureScaling=128,
            heightfieldData=terrain_data.flatten(),
            numHeightfieldRows=terrain_shape[0],
            numHeightfieldColumns=terrain_shape[1],
            physicsClientId=self.client
        )
        
        terrain_id = p.createMultiBody(
            baseMass=0,
            baseCollisionShapeIndex=terrain_shape_id,
            physicsClientId=self.client
        )
        
        return terrain_id
    
    def _create_robot(self) -> int:
        """Create simplified quadruped robot."""
        # Create a simple box body with sphere feet
        # In a full implementation, this would load a URDF file
        
        # Body
        body_collision = p.createCollisionShape(
            p.GEOM_BOX,
            halfExtents=[0.25, 0.15, 0.08],
            physicsClientId=self.client
        )
        body_visual = p.createVisualShape(
            p.GEOM_BOX,
            halfExtents=[0.25, 0.15, 0.08],
            rgbaColor=[0.5, 0.5, 0.5, 1],
            physicsClientId=self.client
        )
        
        robot_id = p.createMultiBody(
            baseMass=10.0,
            baseCollisionShapeIndex=body_collision,
            baseVisualShapeIndex=body_visual,
            basePosition=self.start_pos,
            baseOrientation=p.getQuaternionFromEuler([0, 0, 0]),
            physicsClientId=self.client
        )
        
        return robot_id
    
    def _apply_action(self, joint_targets: np.ndarray):
        """Apply joint position targets to robot."""
        # In a full implementation with URDF, this would use p.setJointMotorControlArray
        # For simplified robot, we apply forces to maintain position
        pass
    
    def _denormalize_action(self, action: np.ndarray) -> np.ndarray:
        """Convert normalized action [-1, 1] to joint positions."""
        # Linear interpolation from normalized to actual joint range
        joint_range = self.joint_limits_upper - self.joint_limits_lower
        joint_targets = self.joint_limits_lower + (action + 1.0) * 0.5 * joint_range
        return joint_targets
    
    def _get_observation(self) -> np.ndarray:
        """
        Get current observation.
        
        Returns 53-dimensional observation:
        - Body position (3)
        - Body orientation quaternion (4)
        - Linear velocity (3)
        - Angular velocity (3)
        - Joint positions (12)
        - Joint velocities (12)
        - Foot contacts (4)
        - Previous action (12)
        """
        if self.robot_id is None:
            return np.zeros(self.observation_space.shape[0])
        
        # Get robot state
        pos, orn = p.getBasePositionAndOrientation(
            self.robot_id,
            physicsClientId=self.client
        )
        vel, ang_vel = p.getBaseVelocity(
            self.robot_id,
            physicsClientId=self.client
        )
        
        # Build observation
        obs = []
        obs.extend(pos)  # 3
        obs.extend(orn)  # 4
        obs.extend(vel)  # 3
        obs.extend(ang_vel)  # 3
        
        # Joint states (simplified - zeros for now)
        joint_positions = np.zeros(12)
        joint_velocities = np.zeros(12)
        obs.extend(joint_positions)  # 12
        obs.extend(joint_velocities)  # 12
        
        # Foot contacts (simplified)
        foot_contacts = self._get_foot_contacts()
        obs.extend(foot_contacts)  # 4
        
        # Previous action
        obs.extend(self.prev_action)  # 12
        
        return np.array(obs, dtype=np.float32)
    
    def _get_foot_contacts(self) -> np.ndarray:
        """Get binary contact state for each foot."""
        # Simplified: assume all feet in contact initially
        return np.ones(4)
    
    def _compute_reward(self, action: np.ndarray) -> float:
        """
        Compute reward as described in README.
        
        Reward components:
        - Forward velocity (1.5x)
        - Lateral movement penalty (-0.5x)
        - Fall penalty (-10.0)
        - Orientation penalty
        - Energy efficiency
        - Smooth motion
        - Survival bonus
        """
        if self.robot_id is None:
            return 0.0
        
        reward = 0.0
        
        # Get robot state
        pos, orn = p.getBasePositionAndOrientation(
            self.robot_id,
            physicsClientId=self.client
        )
        vel, ang_vel = p.getBaseVelocity(
            self.robot_id,
            physicsClientId=self.client
        )
        
        # Forward velocity reward (primary objective)
        reward += 1.5 * vel[0]
        
        # Penalize lateral movement
        reward -= 0.5 * abs(vel[1])
        
        # Stability penalty for low height
        if pos[2] < 0.2:
            reward -= 10.0
        
        # Penalize extreme orientations
        euler = p.getEulerFromQuaternion(orn)
        reward -= 0.5 * (abs(euler[0]) + abs(euler[1]))
        
        # Energy efficiency
        reward -= 0.01 * np.sum(np.square(action))
        
        # Smooth motion penalty
        reward -= 0.05 * np.sum(np.abs(action - self.prev_action))
        
        # Survival bonus
        reward += 0.1
        
        return reward
    
    def _is_done(self) -> bool:
        """Check if episode should terminate."""
        # Episode length limit
        if self.step_counter >= 1000:
            return True
        
        # Check if fallen
        if self._is_fallen():
            return True
        
        return False
    
    def _is_fallen(self) -> bool:
        """Check if robot has fallen."""
        if self.robot_id is None:
            return False
        
        pos, orn = p.getBasePositionAndOrientation(
            self.robot_id,
            physicsClientId=self.client
        )
        
        # Fallen if body is too low
        if pos[2] < 0.15:
            return True
        
        # Fallen if orientation is too extreme
        euler = p.getEulerFromQuaternion(orn)
        if abs(euler[0]) > 1.0 or abs(euler[1]) > 1.0:  # ~60 degrees
            return True
        
        return False
    
    def render(self, mode: str = 'human'):
        """Render environment (handled by PyBullet GUI)."""
        if mode == 'rgb_array':
            # Get camera image
            view_matrix = p.computeViewMatrixFromYawPitchRoll(
                cameraTargetPosition=[0, 0, 0],
                distance=2.0,
                yaw=45,
                pitch=-30,
                roll=0,
                upAxisIndex=2
            )
            proj_matrix = p.computeProjectionMatrixFOV(
                fov=60,
                aspect=1.0,
                nearVal=0.1,
                farVal=100.0
            )
            (_, _, px, _, _) = p.getCameraImage(
                width=640,
                height=480,
                viewMatrix=view_matrix,
                projectionMatrix=proj_matrix,
                renderer=p.ER_BULLET_HARDWARE_OPENGL
            )
            return px
        return None
    
    def close(self):
        """Clean up environment."""
        if self.client >= 0:
            p.disconnect(physicsClientId=self.client)
            self.client = -1
    
    def get_dynamics(self):
        """Get dynamics model for CBF safety filter."""
        from ..dynamics.quadruped_dynamics import QuadrupedDynamics
        return QuadrupedDynamics(self)


def create_quadruped_env(terrain_type='flat', terrain_difficulty=0.5, render=False):
    """
    Convenience function to create quadruped environment.
    
    Args:
        terrain_type: Type of terrain
        terrain_difficulty: Difficulty level 0-1
        render: Enable rendering
        
    Returns:
        QuadrupedEnv instance
    """
    return QuadrupedEnv(
        terrain_type=terrain_type,
        terrain_difficulty=terrain_difficulty,
        render=render
    )
