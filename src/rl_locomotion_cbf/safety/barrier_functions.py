"""Barrier functions for safety constraints."""
import numpy as np
from typing import Tuple, Optional


class BarrierFunction:
    """Base class for control barrier functions."""
    
    def __init__(self, alpha: float = 1.0):
        """
        Initialize barrier function.
        
        Args:
            alpha: Class-K function parameter
        """
        self.alpha = alpha
    
    def evaluate(self, state: np.ndarray) -> float:
        """
        Evaluate barrier function h(x).
        
        Args:
            state: Current state
            
        Returns:
            h: Barrier value (h ≥ 0 means safe)
        """
        raise NotImplementedError
    
    def class_k_function(self, h: float) -> float:
        """
        Class-K function α(h).
        
        Args:
            h: Barrier function value
            
        Returns:
            α(h): Typically λ*h for linear class-K
        """
        return self.alpha * h


class StabilityBarrier(BarrierFunction):
    """Barrier function ensuring CoM stays within support polygon."""
    
    def __init__(self, alpha: float = 1.0, margin: float = 0.05):
        """
        Initialize stability barrier.
        
        Args:
            alpha: Class-K parameter
            margin: Safety margin from polygon boundary (meters)
        """
        super().__init__(alpha)
        self.margin = margin
    
    def evaluate(self, state: np.ndarray) -> float:
        """
        Compute stability barrier value.
        
        h(x) = distance from CoM to support polygon boundary - margin
        h ≥ 0 means CoM is safely within support polygon
        
        Args:
            state: Robot state
            
        Returns:
            h: Barrier value
        """
        # Extract CoM position (assume first 3 elements are position)
        if len(state) < 3:
            return 0.0
        
        com = state[0:3]
        com_xy = com[0:2]
        
        # Simplified support polygon (square)
        # In full implementation, compute from foot contacts
        support_size = 0.3  # meters
        
        # Distance to boundary (Manhattan distance for simplicity)
        dist_x = support_size - abs(com_xy[0])
        dist_y = support_size - abs(com_xy[1])
        distance_to_boundary = min(dist_x, dist_y)
        
        # Barrier value
        h = distance_to_boundary - self.margin
        
        return h


class HeightBarrier(BarrierFunction):
    """Barrier function ensuring minimum body height."""
    
    def __init__(self, alpha: float = 1.0, min_height: float = 0.2):
        """
        Initialize height barrier.
        
        Args:
            alpha: Class-K parameter
            min_height: Minimum safe height (meters)
        """
        super().__init__(alpha)
        self.min_height = min_height
    
    def evaluate(self, state: np.ndarray) -> float:
        """
        Compute height barrier value.
        
        h(x) = z - z_min
        h ≥ 0 means robot is above minimum height
        
        Args:
            state: Robot state
            
        Returns:
            h: Barrier value
        """
        if len(state) < 3:
            return 0.0
        
        z = state[2]
        h = z - self.min_height
        
        return h


class JointLimitBarrier(BarrierFunction):
    """Barrier function for joint limit constraints."""
    
    def __init__(
        self,
        alpha: float = 1.0,
        joint_limits_lower: Optional[np.ndarray] = None,
        joint_limits_upper: Optional[np.ndarray] = None,
        margin: float = 0.1
    ):
        """
        Initialize joint limit barrier.
        
        Args:
            alpha: Class-K parameter
            joint_limits_lower: Lower joint limits (radians)
            joint_limits_upper: Upper joint limits (radians)
            margin: Safety margin from limits (radians)
        """
        super().__init__(alpha)
        self.margin = margin
        
        # Default limits if not provided
        if joint_limits_lower is None:
            self.joint_limits_lower = np.array([-0.8, -1.5, -2.5] * 4)
        else:
            self.joint_limits_lower = joint_limits_lower
            
        if joint_limits_upper is None:
            self.joint_limits_upper = np.array([0.8, 0.5, -0.5] * 4)
        else:
            self.joint_limits_upper = joint_limits_upper
    
    def evaluate(self, state: np.ndarray) -> float:
        """
        Compute joint limit barrier value.
        
        Uses minimum barrier over all joints:
        h(x) = min(q - q_min - margin, q_max - q - margin)
        
        Args:
            state: Robot state
            
        Returns:
            h: Minimum barrier value over all joints
        """
        # Assume joint positions start after body state (pos, vel, etc.)
        # For simplified model, use elements that represent joint positions
        
        num_joints = min(len(self.joint_limits_lower), len(state))
        
        h_min = np.inf
        
        for i in range(num_joints):
            if i < len(state):
                q = state[i]
                
                # Lower limit barrier
                h_lower = q - self.joint_limits_lower[i] - self.margin
                
                # Upper limit barrier
                h_upper = self.joint_limits_upper[i] - q - self.margin
                
                # Take minimum
                h_min = min(h_min, h_lower, h_upper)
        
        return h_min if h_min != np.inf else 0.0


class VelocityBarrier(BarrierFunction):
    """Barrier function for velocity limits."""
    
    def __init__(self, alpha: float = 1.0, max_velocity: float = 2.0):
        """
        Initialize velocity barrier.
        
        Args:
            alpha: Class-K parameter
            max_velocity: Maximum safe velocity (m/s)
        """
        super().__init__(alpha)
        self.max_velocity = max_velocity
    
    def evaluate(self, state: np.ndarray) -> float:
        """
        Compute velocity barrier value.
        
        h(x) = v_max^2 - ||v||^2
        h ≥ 0 means velocity is within limits
        
        Args:
            state: Robot state
            
        Returns:
            h: Barrier value
        """
        # Assume velocity is in state elements 3:6
        if len(state) < 6:
            return self.max_velocity ** 2
        
        velocity = state[3:6]
        v_norm_sq = np.sum(velocity ** 2)
        
        h = self.max_velocity ** 2 - v_norm_sq
        
        return h


class OrientationBarrier(BarrierFunction):
    """Barrier function for orientation constraints."""
    
    def __init__(self, alpha: float = 1.0, max_tilt: float = 0.5):
        """
        Initialize orientation barrier.
        
        Args:
            alpha: Class-K parameter
            max_tilt: Maximum tilt angle (radians)
        """
        super().__init__(alpha)
        self.max_tilt = max_tilt
    
    def evaluate(self, state: np.ndarray) -> float:
        """
        Compute orientation barrier value.
        
        h(x) = max_tilt^2 - (roll^2 + pitch^2)
        h ≥ 0 means orientation is within limits
        
        Args:
            state: Robot state
            
        Returns:
            h: Barrier value
        """
        # Simplified: assume orientation angles in state
        # In full implementation, extract from quaternion
        
        # For now, use a simple heuristic
        h = self.max_tilt ** 2
        
        return h


def create_barrier_set(
    include_stability: bool = True,
    include_height: bool = True,
    include_joints: bool = True,
    include_velocity: bool = False,
    include_orientation: bool = False,
    alpha: float = 1.0
) -> list:
    """
    Create a set of barrier functions.
    
    Args:
        include_stability: Include stability barrier
        include_height: Include height barrier
        include_joints: Include joint limit barriers
        include_velocity: Include velocity barrier
        include_orientation: Include orientation barrier
        alpha: Class-K parameter for all barriers
        
    Returns:
        barriers: List of barrier function instances
    """
    barriers = []
    
    if include_stability:
        barriers.append(StabilityBarrier(alpha=alpha))
    
    if include_height:
        barriers.append(HeightBarrier(alpha=alpha))
    
    if include_joints:
        barriers.append(JointLimitBarrier(alpha=alpha))
    
    if include_velocity:
        barriers.append(VelocityBarrier(alpha=alpha))
    
    if include_orientation:
        barriers.append(OrientationBarrier(alpha=alpha))
    
    return barriers
